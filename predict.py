from facekeypointsnet import FaceKeyPointsNet
import matplotlib.image as mpimg

# 预测图片

facekeypointnet = FaceKeyPointsNet()



if __name__ == '__main__':

    while True:
        img = input('Input image filename:')
        try:
            image = mpimg.imread(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            facekeypointnet.detect_image(image)


    