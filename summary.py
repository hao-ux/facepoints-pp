import paddle
from net.backbone import FaceKeyPointsNetBody
# 输出网络结构

if __name__ == '__main__':
    model = paddle.Model(FaceKeyPointsNetBody(68, backbone='mobilenetv2'))
    model.summary((-1, 3, 224, 244))
