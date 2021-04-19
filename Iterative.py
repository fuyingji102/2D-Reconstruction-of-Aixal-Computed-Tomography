import numpy as np
from PIL import Image
from scipy import fftpack,ndimage
import time
import math
import numba
import matplotlib.pyplot as plt
start = time.perf_counter()


fs = open(r'D:\file\CT\CT Reconstruction\shepp_logan3.txt','r')
f = fs
channel =512
length=512
idata= []
for line in f:
    idata.append(line.strip('\n').split(','))
for i in range(channel):
    for j in range(length):
        idata[i][j] = eval(idata[i][j])             #读取投影数据
data = np.array(idata)                      #N个投影，每个投影有512个数据
data = data.T
data = ((data-np.min(data))/(np.max(data)-np.min(data))).astype('float64')

img_set = np.zeros((512,512)).astype('float64')
erro ,prj_set = np.zeros(512).astype('float64'),np.zeros(512).astype('float64')
for i in range(channel):
    print(i)
    img_set = ndimage.rotate(img_set,-1*180/channel,reshape=False,order=2)
    prj_set = sum(img_set.T)
    erro = (data[i] - prj_set)/512
    for i in range(512):
        img_set[i] = img_set[i] + erro[i]

img = img_set.copy()
img = ndimage.rotate(img_set,-90,reshape=False,order=2)
img = ((img-np.min(img))/(np.max(img)-np.min(img))*255).astype('uint8')
image = Image.fromarray(img)
image.show()











