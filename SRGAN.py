
from pyexpat import model
import numpy as np
from keras import Model
from keras.layers import Conv2D,PReLU,BatchNormalization,Flatten
from keras.layers import UpSampling2D,LeakyReLU,Dense,Input,add
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def res_block(ip):    
    resmodel=Conv2D(64,(3,3),padding="same")(ip)
    resmodel=BatchNormalization(momentum=0.05)(resmodel)
    resmodel=PReLU(shared_axes=[1,2])(resmodel)
    resmodel=Conv2D(64,(3,3),padding="same")(resmodel)
    resmodel=BatchNormalization(momentum=0.5)(resmodel)
    return add([ip,resmodel])


def upscale_block(ip):
    up_model=Conv2D(256,(3,3),padding='same')(ip)
    up_model=UpSampling2D(size=2)(up_model)
    up_model=PReLU(shared_axes=[1,2])(up_model)
    
    return up_model

def create_gen(gen_ip,num_res_block):
    ip=Conv2D(64,(9,9),padding="same")(gen_ip)
    ip=PReLU(shared_axes=[1,2])(ip)
    temp=ip
    for i in range(num_res_block):
        temp=res_block(temp)
    
    temp=Conv2D(64,(3,3),padding="same")(temp)
    temp=BatchNormalization(momentum=0.5)(temp)
    temp=add([temp,ip])
    temp=upscale_block(temp)
    temp=upscale_block(temp)
    
    temp=Conv2D(3,(9,9),padding="same")(temp)
    
    return Model(inputs=gen_ip,outputs=temp)


def discriminator_block(ip, filters, strides=1, bn=True):
    
    disc_model = Conv2D(filters, (3,3), strides = strides, padding="same")(ip)
    
    if bn:
        disc_model = BatchNormalization( momentum=0.8 )(disc_model)
    
    disc_model = LeakyReLU( alpha=0.2 )(disc_model)
    
    return disc_model
def create_dis(disc_ip):

    df = 64
    
    d1 = discriminator_block(disc_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)
    
    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)
    model=Model(disc_ip,validity)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt)

    return model

def build_vgg(hr_shape):
    
    vgg = VGG19(weights="imagenet",include_top=False, input_shape=hr_shape)
    
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)

def combine_model(gen,dis,vgg,low_image):
    gen_img=gen(low_image)
    
    gen_feature=vgg(gen_img)
    
    dis.trainable=False
    
    validity=dis(gen_img) #adversarial loss
    model=Model(inputs=low_image,outputs=[validity,gen_feature])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer=opt)
    return model

def summarize_performance(step, g_model, lr_imgs,hr_imgs, n_samples=3):
	# select a sample of input images
	lr_imgs=lr_imgs[0:n_samples]
	# generate a batch of fake samples
	hr_imgs=hr_imgs[0:n_samples]
	gen_imgs=g_model.predict(lr_imgs)
	# plot real source images
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + i)
		plt.axis('off')
		plt.imshow(hr_imgs[i])
	# plot generated target image
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples + i)
		plt.axis('off')
		plt.imshow(lr_imgs[i])
	# plot real target image
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples*2 + i)
		plt.axis('off')
		plt.imshow(gen_imgs[i])
	# save plot to file
	filename1 = 'train/plot_%06d.png' % (step+1)
	plt.savefig(filename1)
	plt.close()
	g_model.compiled_metrics==None
	# save the generator model
	filename2 = 'weight/model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))


def training(gen_model,dis_model,gan_model,vgg,dataset,batch_size=1,epoches=10,state=0):
    
    lr_data,hr_data=dataset
    num=len(lr_data)
    bat_per_epo=int(num/batch_size)
    fake_label = np.zeros((batch_size, 1)) # Assign a label of 0 to all fake (generated images)
    real_label = np.ones((batch_size,1)) # Assign a label of 1 to all real images.
    
    if(state!=0):
        gan_model.load_weights('weight/model_%06d.h5'%state)
        print('successfully update--- weight/model_%06d.h5'%state)
        
    for epoch in range (state,epoches):
        lr_data,hr_data=shuffle(lr_data,hr_data,random_state=epoch)
        totall_loss_content=0.0
        totall_loss_adver=0.0
        
        for batch in tqdm(range(bat_per_epo)):
            lr_imgs=lr_data[batch*batch_size:batch*batch_size+batch_size]
            hr_imgs=hr_data[batch*batch_size:batch*batch_size+batch_size]
            fake_imgs=gen_model.predict(lr_imgs)
             
            dis_model.trainable=True
            
            d_loss_gen=dis_model.train_on_batch(fake_imgs,fake_label)
            d_loss_real=dis_model.train_on_batch(hr_imgs,real_label)
            
            dis_model.trainable=False
            
            totall_loss_adver += 0.5* (d_loss_gen+d_loss_real)
            
            image_features=vgg.predict(hr_imgs)
            
            g_loss,_,_=gan_model.train_on_batch(lr_imgs,[real_label,image_features])
            
            totall_loss_content += g_loss
            
        print('epoch: %d, d[%.3f], g[%.3f]' % (epoch+1, totall_loss_adver/num,totall_loss_content/num, totall_loss_content/num))
        
        if((epoch+1)%10==0):
            summarize_performance(epoch, gen_model,lr_data,hr_data)
            
            
    
    