# python
# 边缘检测 Robertes cross算子=[[-1,-1],[1,1]] 图像增强锐化
# 一种斜向偏差分的梯度计算，梯度的大小代表边缘的强度，梯度的方向与边缘的走向垂直
# @Zivid 2021/8/16


# 改进的快速边缘检测算法. 如果你的算法用到了cv2.canny 效率不够的话,可以用这个算法替代试试!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!目前效率最快的是Robert03!!!!!!!!!!!
import cv2
import numpy as np
# from skimage import filters


# 方法一：自定义
def Robert01(pic):
    pic_f = np.copy(pic)
    pic_f = pic_f.astype("float")
    Roberts = np.zeros((row, column))
    for x in range(row - 1):  #========这些可以numpy 加速.
        for y in range(column - 1):
            gx = abs(pic_f[x + 1, y + 1] - pic_f[x, y])
            gy = abs(pic_f[x + 1, y] - pic_f[x, y + 1])
            Roberts[x, y] = gx + gy

    sharp = pic_f + Roberts
    sharp = np.clip(sharp,0,255)
    sharp = sharp.astype("uint8")
    sharp[sharp<180]=0 #直接过滤
    return sharp

# 方法二：自定义 这个比较好 ↓
def Robert02(pic):
    ro = [[-1, -1], [1, 1]]
    for i in range(row):
        for j in range(column):
            if(j+2<column) and (i+2<=row):
                process_img = pic[i:i+2, j:j+2]
                list_robert = ro * process_img
                pic[i,j] = abs(list_robert.sum())
    return pic

# 方法三：自定义
def Robert03(pic):
    x_kernel = np.array([[-1,0],[0,1]], dtype=int)
    y_kernel = np.array([[0,-1],[1,0]], dtype=int)
    x = cv2.filter2D(pic, cv2.CV_16S, x_kernel)
    y = cv2.filter2D(pic, cv2.CV_16S, y_kernel)
    absX = np.abs(x)
    absY = np.abs(y)
    Prewitt = cv2.addWeighted(absX, 1, absY, 1, 0)+pic
    sharp=Prewitt
    sharp = np.clip(sharp,0,255)
    sharp = sharp.astype("uint8")
    sharp[sharp<180]=0 #直接过滤
    return sharp
def Robert04(pic):

    Prewitt = cv2.Canny(pic,100,200)+pic
    sharp=Prewitt
    sharp = np.clip(sharp,0,255)
    sharp = sharp.astype("uint8")
    sharp[sharp<180]=0 #直接过滤
    return sharp


# 方法四 直接调用 skimage库内的函数
# def sys_robret(pic):
#     edge_pic = filters.roberts(pic)
#     return edge_pic
#     edge_pic = filters.roberts(pic)
#     return edge_pic
    
if __name__ == "__main__":
    import time

    pic = cv2.imread('test999.bmp', 0)
    row, column = pic.shape
    cv2.imwrite("original.bmp", pic)
    aaa=time.time()
    img01 = Robert01(pic)
    print(time.time()-aaa)
    cv2.imwrite("Robertd-01.bmp", img01)
    
    # img02 = Robert02(pic)
    # cv2.imwrite("Roborts-02.bmp", img02)
    aaa=time.time()
    img03 = Robert03(pic)
    print(time.time()-aaa)
    cv2.imwrite("Roborts-03.bmp", img03)
    aaa=time.time()
    img03 = Robert04(pic)
    print(time.time()-aaa)
    cv2.imwrite("Roborts-04.bmp", img03)
    print('很明显roberts03的方法最快!!!!!!!!!!!')
    # # img04 = sys_robret(pic)
    # # cv2.imshow("Roborts-04", img04)
    # # cv2.waitKey(0)

