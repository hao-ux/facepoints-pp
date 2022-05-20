
import paddle
import pandas as pd
import os
import matplotlib.image as mpimg
import numpy as np
from utils.transforms import Resize, ToCHW, GrayNormalize, RandomCrop


class FaceKeyPointsDatasets(paddle.io.Dataset):
    def __init__(self, csv_file_path, img_path, data_mean, data_std):
        
        self.df = pd.read_csv(csv_file_path)
        self.img_path = img_path
        self.transform = paddle.vision.transforms.Compose([
            Resize(256),RandomCrop(224) ,GrayNormalize(mean=data_mean, std=data_std), ToCHW()
        ])
    
    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, self.df.iloc[index, 0])
        img = mpimg.imread(img_name)
        if img.shape[2] == 4:
            img =img[:,:,0:3]
        kpt = self.df.iloc[index, 1:].values
        kpt = kpt.astype('float').reshape(-1)
        img, kpt = self.transform([img, kpt])
        img = np.array(img, dtype='float32')
        kpt = np.array(kpt, dtype='float32')
        return img, kpt
    
    def __len__(self):
        return len(self.df)
    

    
    

        

        

        