

import paddle
from paddle.distributed import ParallelEnv
import matplotlib.pyplot as plt
import scipy.signal
import os
import matplotlib
import numpy as np
matplotlib.use('Agg')

class LossHistory(paddle.callbacks.Callback):
    """
    保存loss可视化
    logs_dir：保存路径
    """
    def __init__(self, logs_dir):
        super(LossHistory, self).__init__()
        self.train_losses = []
        self.eval_losses = []
        self.logs_dir = logs_dir
    
    def on_train_begin(self, logs=None):
        self.train_losses = []
        self.eval_losses = []
    
    def on_epoch_end(self, epoch, logs=None):
        
        self.train_losses.append(logs.get('loss'))
        
        
    def on_eval_end(self, logs=None):
        self.eval_losses.append(logs.get('loss'))
        iters = range(len(self.train_losses))
        plt.figure()
        plt.plot(iters, self.train_losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.eval_losses, 'coral', linewidth = 2, label='val loss')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.logs_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")
        
class ModelCheckpoint(paddle.callbacks.Callback):
    """
    保存模型
    save_freq：保存频率，每一个epoch
    save_dir：保存路径
    """
    def __init__(self, save_freq=1, save_dir=None):
        self.save_freq = save_freq
        self.save_dir = save_dir

    def on_epoch_begin(self, epoch=None, logs=None):
        self.epoch = epoch

    def _is_save(self):
        return self.model and self.save_dir and ParallelEnv().local_rank == 0

    def on_epoch_end(self, epoch, logs=None):
        if self._is_save() and self.epoch % self.save_freq == 0:
            self.path = '{}/{}_loss_{:.4f}'.format(self.save_dir, epoch, logs.get('loss')[0])

    
    def on_eval_end(self, logs=None):
        if self._is_save() and self.epoch % self.save_freq == 0:
            self.path = '{}_val_loss_{:.4f}'.format(self.path, logs.get('loss')[0])
            print('save checkpoint at {}'.format(os.path.abspath(self.path)))
            self.model.save(self.path)
        

