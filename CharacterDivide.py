import cv2
import numpy as np
# =========================================================================
HIOG = 50
VIOG = 3
Position = []
save_dir = r'C:/Users/Administrator/Desktop/bishe/MyDesign/read/'
base_dir = r'C:/Users/Administrator/Desktop/bishe/MyDesign/test/'

def getHProjection(image):
    hProjection = np.zeros(image.shape, np.uint8)
    # 获取图像大小
    (h, w) = image.shape
    # 统计像素个数
    h_ = [0] * h
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    # 绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y, x] = 255
    return h_

def getVProjection(image):
    vProjection = np.zeros(image.shape, np.uint8);
    (h, w) = image.shape
    w_ = [0] * w
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
    for x in range(w):
        for y in range(h - w_[x], h):
            vProjection[y, x] = 255
    return w_

def scan(vProjection, iog, pos=0):
    start = 0
    V_start = []
    V_end = []
    for i in range(len(vProjection)):
        if vProjection[i] > iog and start == 0:
            V_start.append(i)
            start = 1
        if vProjection[i] <= iog and start == 1:
            if i - V_start[-1] < pos:
                continue
            V_end.append(i)
            start = 0
    return V_start, V_end


def divide(url):
    # 读入原始图像
    origineImage = cv2.imread(url)
    origineImage = cv2.resize(origineImage,(400,900))#多次尝试400,900最优
    # 图像灰度化
    image = cv2.cvtColor(origineImage, cv2.COLOR_BGR2GRAY)
    # 将图片二值化
    retval, img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    (h, w) = img.shape
    # 垂直投影
    V = getVProjection(img)
    # 对垂直投影水平分割
    V_start, V_end = scan(V, HIOG)
    if len(V_start) > len(V_end):
        V_end.append(w - 5)
    # 分割行，分割之后再进行列分割并保存分割位置
    for i in range(len(V_end)):
        # 获取行图像
        if V_end[i] - V_start[i] < 15:
            continue
        cropImg = img[0:h, V_start[i]:V_end[i]]
        # 对行图像进行垂直投影
        H = getHProjection(cropImg)
        H_start, H_end = scan(H, VIOG, 40)
        if len(H_start) > len(H_end):
            H_end.append(h - 5)
        for pos in range(len(H_start)):
            # 再进行一次列扫描
            DcropImg = cropImg[H_start[pos]:H_end[pos], 0:w]
            d_h, d_w = DcropImg.shape
            sec_V = getVProjection(DcropImg)
            c1, c2 = scan(sec_V, 0)

            if len(c1) > len(c2):
                c2.append(d_w)
            x = 1
            while x < len(c1):
                if c1[x] - c2[x - 1] < 8:
                    c2.pop(x - 1)
                    c1.pop(x)
                    x -= 1
                x += 1
            if len(c1) == 1:
                Position.append([V_start[i], H_start[pos], V_end[i], H_end[pos]])
                crop = origineImage[H_start[pos]: H_end[pos], V_start[i]: V_end[i]]
                row = str(10 + int(pos))
                column = str(i)
                cv2.imwrite(save_dir + row + '_' + column + '.jpg', crop)
    return int(row),int(column)
            # else:
            #     for x in range(len(c1)):
            #         Position.append([V_start[i] + c1[x], H_start[pos], V_start[i] + c2[x], H_end[pos]])
            #         crop = origineImage[ H_start[pos]-2:H_end[pos]+2,V_start[i] + c1[x]-2: V_start[i] + c2[x]+2]
            #         cv2.imwrite(save_dir + str(i) + '_' + str(pos) + '.jpg', crop)
    # 根据确定的位置顯示分割字符
    # for m in range(len(Position)):
    #     cv2.rectangle(origineImage, (Position[m][0] , Position[m][1] ), (Position[m][2] , Position[m][3]),
    #                   (0, 255, 0), 1)
    # cv2.namedWindow('image',0)
    # cv2.resizeWindow('image',400,500)
    # cv2.imshow('image', cv2.resize(origineImage, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))#局部像素重采樣
    # cv2.waitKey(0)

