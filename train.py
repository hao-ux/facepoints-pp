
from datasets import FaceKeyPointsDatasets
import paddle
from paddle.static import InputSpec
from net.backbone import FaceKeyPointsNetBody
from utils.metric import NME
import paddle.nn as nn
from utils.callback import LossHistory, ModelCheckpoint
from utils.optimizer import create_optimzer
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

paddle.set_device('gpu') # 使用gpu


# ------------------------------------ #
# -----------参数设置------------------ # 
# batch_size -> 批次
# 主干网络 -> 'mobilenetv1', 'mobilenetv2', 'resnet50'
# epochs -> 轮次
# model_path训练权重，会自动下载预训练权重，可以不用设置
# input_shape -> 输入图片大小

batch_size = 16
backbone = 'mobilenetv2'
epochs = 100
model_path = ''
input_shape = [224,224]


if __name__ == '__main__':
    df = pd.read_csv('./datasets/training_frames_keypoints.csv')

    labels = df.values[:,1:]
    data_mean = labels.mean()
    data_std = labels.std()

    train_datasets = FaceKeyPointsDatasets('./datasets/training_frames_keypoints.csv', './datasets/training', data_mean, data_std)
    valid_datasets = FaceKeyPointsDatasets('./datasets/test_frames_keypoints.csv', './datasets/test', data_mean, data_std)

    step_each_epoch = len(train_datasets) // batch_size

    model = paddle.Model(FaceKeyPointsNetBody(68, backbone=backbone),inputs=[InputSpec(shape=[3, input_shape[0], input_shape[1]], dtype='float32', name='image')])
    if model_path!='':
        model.load(model_path)
        print('导入模型成功！！！')

    loss = nn.SmoothL1Loss()

    metric = NME()

    model.prepare(create_optimzer(model.parameters(), step_each_epoch, epochs), loss=loss, metrics=metric)

    visualdl = paddle.callbacks.VisualDL(log_dir='./logs1')
    EarlyStopping = paddle.callbacks.EarlyStopping(save_best_model=False,patience=15)
    LRScheduler = paddle.callbacks.LRScheduler(by_epoch=True, by_step=False)
    loss_history = LossHistory('./metric')
    modelcheckpoint = ModelCheckpoint(save_dir='./logs')
    
    model.fit(train_datasets,
            valid_datasets,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[visualdl, EarlyStopping, modelcheckpoint, LRScheduler, loss_history])
    