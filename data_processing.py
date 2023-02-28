from json import load
from keras.layers import Input
from SRGAN import create_dis,create_gen,combine_model,training,build_vgg
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from numpy import asarray
import numpy as np
import os
from tqdm import tqdm
from numba import cuda
import tensorflow as tf
import cv2 as cv
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
device = cuda.get_current_device()
print(device)
device.reset()



def load_data(path):
    lr_data, hr_data = list(), list()
    for filename in tqdm(os.listdir(path)):
        hr_img = load_img(os.path.join(path,filename),target_size=(128,128))
        hr_img = img_to_array(hr_img)
        lr_img= cv.resize(hr_img,(32,32))
        lr_data.append(lr_img)
        hr_data.append(hr_img)
    
    return [asarray(lr_data),asarray(hr_data)]

def preprocess_data(data):
	# load compressed arrays
	# unpack arrays
	X1, X2 = data
	# scale from [0,255] to [-1,1]
	X1 = (X1) / 255
	X2 = (X2) / 255
	return [X1, X2]


#data=load_data('data/original_images')
#data=preprocess_data(data)

hr_ip = Input(shape=(128,128,3))
lr_ip = Input(shape=(32,32,3))

dis=create_dis(hr_ip)
vgg=build_vgg((128,128,3))
vgg.trainable=False
gen=create_gen(lr_ip,16)
dis.summary()
gen.summary()
gan_model=combine_model(gen,dis,vgg,lr_ip)

vgg.summary()
gan_model.summary()

