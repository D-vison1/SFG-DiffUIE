import numpy as np
import os
import cv2
import math
from skimage import io, color, filters
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
 
from functools import partial
np.seterr(divide='ignore',invalid='ignore')
 
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def isImagefile(input):
    return any(input.endswith(ext) for ext in IMG_EXTENSIONS)

def cal_ssim(im1,im2):
      assert len(im1.shape) == 2 and len(im2.shape) == 2
      assert im1.shape == im2.shape
      mu1 = im1.mean()
      mu2 = im2.mean()
      sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
      sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
      sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
      k1, k2, L = 0.01, 0.03, 255
      C1 = (k1*L) ** 2
      C2 = (k2*L) ** 2
      C3 = C2/2
      l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
      c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
      s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
      ssim = l12 * c12 * s12
      return ssim

# 计算均值
def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # 对输入部分像素值排序
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # 获取截断的范围
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    # 计算公式权重
    weight = (1/(K-T_a_L-T_a_R))
    # 根据范围求和
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    # 返回均值
    return val
# 计算方差
def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)

def _uicm(x):
    # 将图像3个分量展开
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    # 得到对应的RG和YB分量
    RG = R-G
    YB = ((R+G)/2)-B
    # 分别计算RG和YB分量的均值和方差
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    # 根据公式得到最终的结果
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)


def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # 得到图像的每个颜色分量
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    # 对每个分量使用sobel算子
    Rs = filters.sobel(R)
    Gs = filters.sobel(G)
    Bs = filters.sobel(B)
    # 将生成的边缘图与原始图像相乘得到灰度边缘图
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    # 对每个通道使用EME算法
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

def eme(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]

            blockmin = float(np.min(block))
            blockmax = float(np.max(block))

            # # old version
            # if blockmin == 0.0: eme += 0
            # elif blockmax == 0.0: eme += 0
            # else: eme += w * math.log(blockmax / blockmin)

            # new version
            if blockmin == 0: blockmin+=1
            if blockmax == 0: blockmax+=1
            eme += w * math.log(blockmax / blockmin)
    return eme

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c


def _uiconm(ch, blocksize=8):
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]
            blockmin = float(np.min(block))
            blockmax = float(np.max(block))

            top = plipsub(blockmax,blockmin)
            bottom = plipsum(blockmax,blockmin)
            if bottom == 0 :
                s+=0
            else:
                m = top/bottom
                if m ==0.:
                    s+=0
                else:
                    s += (m) * np.log(m)

    return plipmult(w,s)

def Entropy(inputdir, resultdir):

    inputdir_list = os.listdir(inputdir)
    inputdir_list.sort()

    fc = open(os.path.join(resultdir,'entropy.txt'), 'w')

    entropy_list = []
    avg_entropy = 0
    n_images = 0

    for name in inputdir_list:
        if name.endswith('.jpg'):
            n_images += 1
    print('number of images :',n_images)

    for name in inputdir_list:

        path = os.path.join(inputdir,name)
        image = cv2.imread(path, 0)

        tmp = []
        for i in range(256):
            tmp.append(0)
        val = 0
        k = 0
        res = 0

        img = np.array(image)
        for i in range(len(img)):
            for j in range(len(img[i])):
                val = img[i][j]
                tmp[val] = float(tmp[val] + 1)
                k =  float(k + 1)
        for i in range(len(tmp)):
            tmp[i] = float(tmp[i] / k)
        for i in range(len(tmp)):
            if(tmp[i] == 0):
                res = res
            else:
                res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
        
        avg_entropy += res/n_images
        print(name+':')
        print('     	entropy', res)
        entropy_list.append(res)
        fc.write('%s : %.4f\n' %(name, res))
    print('Avg. entropy: %f\n' %(avg_entropy))
    fc.write('Avg. entropy: %f\n' %(avg_entropy))
    fc.close()
 

def UIQM(inputdir, resultdir):

    inputdir_list = os.listdir(inputdir)
    inputdir_list.sort()

    fc = open(os.path.join(resultdir,'uiqm.txt'), 'w')

    uiqm_list = []
    avg_uiqm = 0
    n_images = 0
    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753

    for name in inputdir_list:
        if name.endswith('.jpg'):
            n_images += 1
    print('number of images :',n_images)
    for name in inputdir_list:
 
        path1 = os.path.join(inputdir,name)
        img = cv2.imread(path1)
        uicm = _uicm(img)
        uism = _uism(img)
        uiconm = _uiconm(img)

        print(name+':')
        uiqm = p1 * uicm + p2 * uism + p3 * uiconm
        avg_uiqm += uiqm/n_images
        print('     	uiqm', uiqm)
        uiqm_list.append(uiqm)

        fc.write('%s : %.4f\n' %(name, uiqm))
    print('Avg. uiqm: %f\n' %(avg_uiqm))
    fc.write('Avg. uiqm: %f\n' %(avg_uiqm))
    fc.close()

def UCIQE(inputdir, resultdir):
    inputdir_list = os.listdir(inputdir)
    inputdir_list.sort()

    fc = open(os.path.join(resultdir,'uciqe.txt'), 'w')

    uciqe_list = []
    avg_uci = 0
    n_images = 0

    for name in inputdir_list:
        if name.endswith('.jpg'):
            n_images += 1
    print('number of images :',n_images)
    for name in inputdir_list:

        path1 = os.path.join(inputdir,name)

        image = cv2.imread(path1)#图片路径
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # RGB转为HSV 
        H, S, V = cv2.split(hsv)
        delta = np.std(H) /180  #色度的标准差 
        mu = np.mean(S) /255  #饱和度的平均值 
        n, m = np.shape(V)
        number = math.floor(n*m/100)  #所需像素的个数  
        Maxsum, Minsum = 0, 0
        V1, V2 = V /255, V/255
        
        for i in range(1, number+1):
            Maxvalue = np.amax(np.amax(V1))
            x, y = np.where(V1 == Maxvalue)
            Maxsum = Maxsum + V1[x[0],y[0]]
            V1[x[0],y[0]] = 0
        
        top = Maxsum/number
        
        for i in range(1, number+1):
            Minvalue = np.amin(np.amin(V2))
            X, Y = np.where(V2 == Minvalue)
            Minsum = Minsum + V2[X[0],Y[0]]
            V2[X[0],Y[0]] = 1
        
        bottom = Minsum/number
        
        conl = top-bottom

        print(name+':')
        uciqe = 0.4680*delta + 0.2745*conl + 0.2576*mu
        uciqe = round(uciqe, 4)
        avg_uci += uciqe/n_images
        avg_uci = round(avg_uci, 4)
        print('     	uciqe', uciqe)
        uciqe_list.append(uciqe)

        fc.write('%s : %.4f\n' %(name, uciqe))
    print('Avg. uciqe: %f\n' %(avg_uci))
    fc.write('Avg. uciqe: %f\n' %(avg_uci))
    fc.close()
    

def quality_assess_SM3(inputdir, gtdir,resultdir):
    inputdir_list = os.listdir(inputdir)
    gtdir_list = os.listdir(gtdir)
    inputdir_list.sort()
    gtdir_list.sort()
 
    fc = open(os.path.join(resultdir,'psnr_ssim.txt'), 'w')
 
    ssim_list = []
    psnr_list = []
    n_images = 0
    avg_p = 0
    avg_s = 0
    for name in gtdir_list:
        if name.endswith('.jpg'):
            n_images += 1
    print('number of images :',n_images)
    for name in gtdir_list:
 
        path1 = os.path.join(inputdir,name)
        path2 = os.path.join(gtdir,name)
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
 
        inputImg_gry = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        GT_gry = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
 
        print(name+':')
        p = compare_psnr(img1, img2)
        avg_p += p/n_images
        print('     	psnr', p)
        psnr_list.append(p)
 
        # s = compare_ssim(inputImg_gry,GT_gry)
        s=cal_ssim(inputImg_gry,GT_gry)
        avg_s += s/n_images
        print('     	ssim', s)
        ssim_list.append(s)
 
        fc.write('%s : %.4f %.4f \n' %(name, p, s))
    print('Avg. psnr: %f ssim: %f\n' %(avg_p, avg_s))
    fc.write('Avg. psnr: %f ssim: %f\n' %(avg_p, avg_s))
    fc.close()
 
def dropext(fn): return os.path.splitext(fn)[0]
 
# gtdir ='E:\\PythonProject\\2nd_work\\Our\\UIEB\\GT'
inputdir = 'E:\\PythonProject\\2nd_work\\Our\\LSUI\\GT'
resultdir = 'E:/PythonProject/PSNR_SSIM/psnr_results/LSUI-GT'
# quality_assess_SM3(inputdir, gtdir, resultdir)
UIQM(inputdir, resultdir)
UCIQE(inputdir, resultdir)
 
print("____________________take notes:network_epoch_datasets_______________________________")