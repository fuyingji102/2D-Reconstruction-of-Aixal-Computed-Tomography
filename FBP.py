import numpy as np
from PIL import Image
from scipy import fftpack
import time
import math
import numba
import matplotlib.pyplot as plt
start = time.perf_counter()


fs = open(r'D:\file\CT\CT Reconstruction\shepp_logan3.txt','r')



f = fs

num =512
lenth=512

idata= []
for line in f:
    idata.append(line.strip('\n').split(','))

#for i in range(lenth):
#   idata[i].pop()

for i in range(num):
    for j in range(lenth):
        idata[i][j] = eval(idata[i][j])             #读取投影数据

data = np.array(idata)                      #N个投影，每个投影有512个数据
data = data.T




ramp_win = np.zeros(lenth)
for i in range(int(lenth/2)):               #Ramp窗
    ramp_win[i] = i
for i in range(int(lenth/2)):
    ramp_win[511-i] = ramp_win[i+1]
ramp_win[256] = (ramp_win[255]+ramp_win[257])/2

ham_win = np.zeros(lenth)                   #hamming窗
for i in range(int(lenth/2)):
    ham_win[i] = i*(0.54+0.46*np.cos(2*np.pi*i/512))
for i in range(int(lenth/2)):
    ham_win[511-i] =ham_win[i+1]
ham_win[256] = (ham_win[255]+ham_win[257])/2


shepp_win = np.zeros(512)                   #shepp_logan窗
for i in range(256):
    shepp_win[i] = 512*0.5*np.sin(2*np.pi*i/1024)
for i in range(256):
    shepp_win[511-i] = shepp_win[i+1]
shepp_win[256] = (shepp_win[255]+shepp_win[257])/2

han_win = np.zeros(512)                     #Hanning窗
for i in range(256):
    han_win[i] = i*(0.5+0.5*np.cos(2*np.pi*i/512))
for i in range(256):
    han_win[511-i] =han_win[i+1]
han_win[256] = (han_win[255]+han_win[257])/2



def filter_func(data,win):
    f_data = fftpack.fft(data)                      #对投影进行FFT
    for i in range(num):
        f_data[i] = f_data[i]*win       

    prj = fftpack.ifft(f_data)                  #滤波后反FFT
    prj = np.real(prj)
    return prj


@numba.jit
def back_project(prj):
    img = np.zeros((512,512))
    for k in range(512):                        #反投影
        
        for i in range(512):
            for j in range(512):
                d = (i-255)*np.cos(k/512*np.pi) + (j-255)*np.sin(k/512*np.pi)
                d = d + 255               #d为像素点到投影中心轴距离
    #            d = int(d)
                if d>511 or d<0:
                     continue
    #            img[i][j] = img[i][j] + prj[k][d]      #多次反投影累计
                d1,d2 = int(math.floor(d)),int(math.ceil(d))
                img[i][j] = img[i][j] + (1-d+d1)* prj[k][d1]+(1+d-d2)*prj[k][d2]
    return img

def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr




def make(img):
    img = (img-np.min(img))/(np.max(img)-np.min(img))*255   #标定到0~255
    img = img.astype('uint8')
    img = flip90_left(img)

    for i in range(512):                                #环形伪影去除
        for j in range(512):
            r = ((i-255)**2+(j-255)**2)**(1/2)
            if r <=255:
                pass
            else:
                img[i][j]  =  0
                
    for i in range(512):                                #环形伪影去除
        for j in range(512):
            r = ((i-255)**2+(j-255)**2)**(1/2)
            if r <=255:
                img[i][j] =  0 if img[i][j] - 25 <0 else img[i][j] - 25
            else:
                pass
    img[256][256] = (img[254][254]+img[257][257])/2
    img[255][255] = img[256][256]
    img = (img-np.min(img))/(np.max(img)-np.min(img))*255
    img = img.astype('uint8')
    
    image = Image.fromarray(img)
    return image





prj = filter_func(data,ham_win)
img = back_project(prj)
image = make(img)
image.show()


end = time.perf_counter()






