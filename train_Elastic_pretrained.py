import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss

from utils import losses_Elastic
from config.config import config as cfg
from utils.dataset_Elastic import MXFaceDataset, DataLoaderX
from utils.utils_callbacks_Elastic import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging_Elastic import AverageMeter, init_logging

from backbones.iresnet_Elastic import iresnet100, iresnet50


torch.backends.cudnn.benchmark = True

def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if not os.path.exists(cfg.output) and rank == 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)

    trainset = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)

    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)

    # load model
    if cfg.network == "iresnet100":
        backbone = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "iresnet50":
        backbone = iresnet50(dropout=0.4,num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    else:
        backbone = None
        logging.info("load backbone failed!")
        exit()

    backbone_pth = '/kaggle/input/epoch0y1/pytorch/default/1/22744backbone.pth'
    state_dict = torch.load(backbone_pth, map_location=torch.device(local_rank), weights_only=True)
    backbone.load_state_dict(state_dict)
    if rank == 0:
        logging.info(f"Backbone loaded successfully from {backbone_pth}")

    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    backbone = DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    # get header
    if cfg.loss == "ElasticArcFace":
        header = losses_Elastic.ElasticArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,std=cfg.std).to(local_rank)
    elif cfg.loss == "ElasticArcFacePlus":
        header = losses_Elastic.ElasticArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,
                                       std=cfg.std, plus=True).to(local_rank)
    elif cfg.loss == "ElasticCosFace":
        header = losses_Elastic.ElasticCosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,std=cfg.std).to(local_rank)
    elif cfg.loss == "ElasticCosFacePlus":
        header = losses_Elastic.ElasticCosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,
                                       std=cfg.std, plus=True).to(local_rank)
    elif cfg.loss == "ArcFace":
        header = losses_Elastic.ArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(local_rank)
    elif cfg.loss == "CosFace":
        header = losses_Elastic.CosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(
            local_rank)
    else:
        print("Header not implemented")
    if args.resume:
        try:
            # header_pth = os.path.join(cfg.output, str(cfg.global_step) + "header.pth")
            header_pth = "/kaggle/input/epoch0y1/pytorch/default/1/22744header.pth"
            header.load_state_dict(torch.load(header_pth, map_location=torch.device(local_rank), weights_only=True))

            if rank == 0:
                logging.info("header resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("header resume init, failed!")
    
    header = DistributedDataParallel(
        module=header, broadcast_buffers=False, device_ids=[local_rank])
    header.train()

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_header = torch.optim.SGD(
        params=[{'params': header.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_header, lr_lambda=cfg.lr_func)

    criterion = CrossEntropyLoss()

    # Initialize AMP
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0: logging.info("Total Step is: %d" % total_step)

    if args.resume:
        rem_steps = (total_step - cfg.global_step)
        cur_epoch = cfg.num_epoch - int(cfg.num_epoch / total_step * rem_steps)
        logging.info("resume from estimated epoch {}".format(cur_epoch))
        logging.info("remaining steps {}".format(rem_steps))
        
        start_epoch = cur_epoch
        scheduler_backbone.last_epoch = cur_epoch
        scheduler_header.last_epoch = cur_epoch

        opt_backbone.param_groups[0]['lr'] = scheduler_backbone.get_last_lr()[0]
        opt_header.param_groups[0]['lr'] = scheduler_header.get_last_lr()[0]

    callback_verification = CallBackVerification(cfg.eval_step, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    global_step = cfg.global_step
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for _, (img, label) in enumerate(train_loader):
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)

            # Use AMP for forward pass
            with torch.cuda.amp.autocast():
                features = F.normalize(backbone(img))
                thetas = header(features, label)
                loss_v = criterion(thetas, label)

            # Check for NaN values in the loss
            if torch.any(torch.isnan(loss_v)):
                logging.error(f"NaN detected in loss at step {global_step}, skipping step")
                continue

            # Backward pass with scaled gradients
            scaler.scale(loss_v).backward()

            # Clip gradients
            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)

            # Optimizer step
            scaler.step(opt_backbone)
            scaler.step(opt_header)

            # Update the scale for the next iteration
            scaler.update()

            # Zero the gradients
            opt_backbone.zero_grad()
            opt_header.zero_grad()

            loss.update(loss_v.item(), 1)

            callback_logging(global_step, loss, epoch)
            callback_verification(global_step, backbone)

        scheduler_backbone.step()
        scheduler_header.step()

        callback_checkpoint(global_step, backbone, header)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss  training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    args_ = parser.parse_args()
    main(args_)
