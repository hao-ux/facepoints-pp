import paddle
import paddle.nn as nn
from paddle.vision.models import mobilenet_v1, mobilenet_v2, resnet50

class FaceKeyPointsNetBody(nn.Layer):
    def __init__(self, kpt_num=68, backbone='mobilenetv1'):
        if backbone not in ['mobilenetv1', 'mobilenetv2', 'resnet50']: assert "Backbone definition error"
        super(FaceKeyPointsNetBody, self).__init__()
        if backbone == 'mobilenetv1':
            self.backbone = mobilenet_v1(pretrained=True, scale=1.0)
        elif backbone == 'mobilenetv2':
            self.backbone = mobilenet_v2(pretrained=True, scale=1.0)
        else:
            self.backbone = resnet50(pretrained=True)
        self.linear1 = nn.Linear(1000, out_features=512)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(512, kpt_num*2)
    def forward(self, x):
        x = self.backbone(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x