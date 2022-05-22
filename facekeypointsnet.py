
import paddle
from paddle.static import InputSpec
from utils.transforms import Resize, ToCHW, GrayNormalize, RandomCrop
import numpy as np
from net.backbone import FaceKeyPointsNetBody
import paddle.vision.transforms as T
import pandas as pd
from utils.utils import decode_show
import time

class FaceKeyPointsNet(object):
    _defaults = {
        "model_path": "./model_data/models", # 权重路径
        "input_shape": [224, 224],  # 输入图片大小
        "backbone": "mobilenetv2", # 网络结构
    }
    
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
        
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.df = pd.read_csv('C:\\Users\\豪豪\\Envs\\tensorflow\\cv\\人脸关键点检测\\facekeypoints-tf2\\datasets\\training_frames_keypoints.csv')
        key_pts_values = self.df.values[:,1:]
        self.data_mean = key_pts_values.mean() # 计算均值
        self.data_std = key_pts_values.std()   # 计算标准差
        
        # 使用gpu
        paddle.device.set_device('gpu')
        self.transforms = T.Compose([
            Resize(256), RandomCrop(224)
        ])
        self.norm = GrayNormalize(mean=self.data_mean, std=self.data_std)
        self.to_chw = ToCHW()
        self.generate()
        print("导入模型成功！！！")
        
    def generate(self):
        self.model = FaceKeyPointsNetBody(backbone=self.backbone)
        self.model = paddle.Model(self.model, inputs=[InputSpec(shape=[3, self.input_shape[0], self.input_shape[1]], dtype='float32', name='image')])
        self.model.load(self.model_path)
        self.model.prepare()
        
    def detect_image(self, img):
        img = np.array(img)
        if img.shape[2] == 4:
            img =img[:,:,:3]
        kpt = np.ones((136, 1))
        rgb_img, kpt = self.transforms([img, kpt])
        img, kpt = self.norm([rgb_img, kpt])
        img, kpt = self.to_chw([img, kpt])
        img = np.array([img], dtype='float32')
        out = self.model.predict_batch([img])
        print(np.array(out).shape)
        out = out[0].reshape((out[0].shape[0], 136, -1))
        print(np.array(out).shape)
        decode_show(rgb_img, out, self.data_mean, self.data_std)
        
    def fps(self, img, n=100):
        start = time.time()
        img = np.array(img)
        if img.shape[2] == 4:
            img =img[:,:,:3]
        kpt = np.ones((136, 1))
        rgb_img, kpt = self.transforms([img, kpt])
        img, kpt = self.norm([rgb_img, kpt])
        img, kpt = self.to_chw([img, kpt])
        img = np.array([img], dtype='float32')
        for _ in range(n):
            out =self.model.predict_batch([img])
        end = time.time()
        avg_time = (end - start)/n
        return avg_time
        
