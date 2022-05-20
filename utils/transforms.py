import numpy as np
import paddle
import random



class GrayNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, data):
        img, kpt = data[0], data[1]
        img_copy = np.copy(img)
        kpt_copy = np.copy(kpt)
        
        # 灰度化
        gray_c = paddle.vision.transforms.Grayscale(num_output_channels=3)
        img_copy = gray_c(img_copy)

        img_copy = img_copy / 255.0
        
        # 坐标点缩放到-1，1
        kpt_copy = (kpt_copy - self.mean) / self.std
        
        
        return img_copy, kpt_copy
    
class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))

        self.output_size = output_size
        
    def __call__(self, data):
        img, kpt = data[0], data[1]

        img_copy = np.copy(img)
        kpt_copy = np.copy(kpt)
        
        h, w = img_copy.shape[:2] # 500 200
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = paddle.vision.transforms.resize(img_copy, (new_h, new_w))
        kpt_copy[0::2] = kpt_copy[0::2] * new_w / w
        kpt_copy[1::2] = kpt_copy[1::2] * new_h / h
        return img, kpt_copy
    
class RandomFilp(object):
    def __call__(self, data):
        img, kpt = data[0], data[1]
        img_copy = np.copy(img)
        kpt_copy = np.copy(kpt)
        h, w,_ = img_copy.shape
        mode = random.randint(1, 10)
        if mode in [1,2,3,4,5,6,7]:
            img_copy = img_copy[:,::-1,:]
            kpt_copy[::2] = w- kpt_copy[::2]
        
    
        return img_copy, kpt_copy
    
class RandomCrop(object):
    # 随机位置裁剪输入的图像

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        image = data[0]
        key_pts = data[1]

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        h, w = image_copy.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image_copy = image_copy[top: top + new_h,
                      left: left + new_w]

        key_pts_copy[::2] = key_pts_copy[::2] - left
        key_pts_copy[1::2] = key_pts_copy[1::2] - top

        return image_copy, key_pts_copy
class ToCHW(object):
    # HWC -> CHW
    def __call__(self, data):
        img, kpt = data[0], data[1]
        transpose = paddle.vision.transforms.Transpose((2, 0, 1))
        img = transpose(img)
        return img, kpt