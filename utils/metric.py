from paddle.metric import Metric
import numpy as np

class NME(Metric):
    """
    1. 继承paddle.metric.Metric
    """
    def __init__(self, name='nme', *args, **kwargs):
        """
        2. 构造函数实现，自定义参数即可
        """
        super(NME, self).__init__(*args, **kwargs)
        self._name = name
        self.rmse = 0
        self.sample_num = 0
    
    def name(self):
        """
        3. 实现name方法，返回定义的评估指标名字
        """
        return self._name
    
    def update(self, preds, labels):
        """
        4. 实现update方法，用于单个batch训练时进行评估指标计算。
        - 当`compute`类函数未实现时，会将模型的计算输出和标签数据的展平作为`update`的参数传入。
        """
        N = preds.shape[0]

        preds = preds.reshape((N, -1, 2))
        labels = labels.reshape((N, -1, 2))

        self.rmse = 0
        
        for i in range(N):
            pts_pred, pts_gt = preds[i, ], labels[i, ]
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])

            self.rmse += np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * preds.shape[1])
            self.sample_num += 1

        return self.rmse / N
    
    def accumulate(self):
        """
        5. 实现accumulate方法，返回历史batch训练积累后计算得到的评价指标值。
        每次`update`调用时进行数据积累，`accumulate`计算时对积累的所有数据进行计算并返回。
        结算结果会在`fit`接口的训练日志中呈现。
        """
        return self.rmse / self.sample_num
    
    def reset(self):
        """
        6. 实现reset方法，每个Epoch结束后进行评估指标的重置，这样下个Epoch可以重新进行计算。
        """
        self.rmse = 0
        self.sample_num = 0