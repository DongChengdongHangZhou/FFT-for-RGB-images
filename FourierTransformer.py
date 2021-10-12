import cv2
import numpy as np


def grey_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    phase = np.angle(fshift) #取相位
    magnitude = np.abs(fshift) #取振幅
    return magnitude,phase

def grey_ifft(magnitude,phase):
    s1_real = magnitude*np.cos(phase) #取实部
    s1_imag = magnitude*np.sin(phase) #取虚部
    s2 = np.zeros(magnitude.shape,dtype=complex)
    s2.real = np.array(s1_real) 
    s2.imag = np.array(s1_imag)
    f2shift = np.fft.ifftshift(s2) #对新的进行逆变换
    img_back = np.fft.ifft2(f2shift)
    img_back = np.abs(img_back)
    return img_back

def color_fft(img):
    R_channel = img[:,:,0]
    G_channel = img[:,:,1]
    B_channel = img[:,:,2]
    R_channel_magnitude,R_channel_phase = grey_fft(R_channel)
    G_channel_magnitude,G_channel_phase = grey_fft(G_channel)
    B_channel_magnitude,B_channel_phase = grey_fft(B_channel)
    return R_channel_magnitude,G_channel_magnitude,B_channel_magnitude,R_channel_phase,G_channel_phase,B_channel_phase

def color_ifft(R_channel_magnitude,G_channel_magnitude,B_channel_magnitude,R_channel_phase,G_channel_phase,B_channel_phase):
    R_channel_img_back=grey_ifft(R_channel_magnitude,R_channel_phase)
    G_channel_img_back=grey_ifft(G_channel_magnitude,G_channel_phase)
    B_channel_img_back=grey_ifft(B_channel_magnitude,B_channel_phase)
    img_back = np.stack((R_channel_img_back,G_channel_img_back,B_channel_img_back),axis=2)
    return img_back

if __name__ =='__main__':
    img = cv2.imread('0.png')
    R_channel_magnitude,G_channel_magnitude,B_channel_magnitude,R_channel_phase,G_channel_phase,B_channel_phase = color_fft(img)
    img_back = color_ifft(R_channel_magnitude,G_channel_magnitude,B_channel_magnitude,R_channel_phase,G_channel_phase,B_channel_phase)
    cv2.imwrite('nnn.png',img_back)
