# Graduation-thesis


## 1. Overview
ElasticArcFace is a powerful face recognition model that combines the advantages of ArcFace and ElasticNet regularization. It uses an advanced neural network architecture to provide highly accurate facial recognition results. This model leverages the latest advances in deep learning to achieve high performance in both accuracy and efficiency.

## 2. Installation:
To install the required dependencies, run the following command:

```shell
pip install -r requirement.txt
```

## 3. Installation at Kaggle:
To install the required dependencies at Kaggle notebook, run the following command:

```shell
!pip install tensorboard
!pip install easydict
!pip install scikit-learn
!pip install matplotlib
!pip install pandas
!pip install scikit-image
!pip install menpo
!pip install prettytable
!pip install mxnet
!pip install torch-summary
!pip install opencv-python
```
or
```shell
!pip install -r requirement.txt
```

## 4. Usage
ElasticArcFace can be used for both training and inference. After installation, you can train the model on your dataset or use a pre-trained model for face recognition tasks. Detailed steps are provided in the "Train Model" section below.

## 5. Dataset
ElasticArcFace is compatible with various face recognition datasets such as GLINT360K, MS1M, and VGGFace2. You can prepare your dataset in a suitable format (e.g., image folders or a CSV file with image paths and labels) and specify the dataset path in the configuration file (e.g., configs/glint360k_r100.py).

##6. Train model
### 6.1. Train model
```shell
!CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_Elastic.py 
```

### 6.2. Using pretrained model to continue training
```shell
!CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_Elastic_pretrained.py --resume 1
```




