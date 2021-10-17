import numpy as np
import pydicom
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf 
from glob import glob
import os
from PIL import Image

tf.random.set_seed(42)
np.random.seed(42)

class CycleGAN_IMG:

    def __init__(self):
        self.cyclegan = keras.models.load_model('./SR/run/CycleGAN/mri_256.1.resnet_RFA/model/g_AB_model.h5')
        # self.dcgan = keras.models.load_model('C:/Users/thsdb/AppData/Local/Programs/Python/Python39/MyProject/ConnectAI/dcgan_model.h5')
    
    def predict_gan(self, gan, dataset):
        generator = gan
        g_img = []

        for X_batch in dataset: # 혹시나 한 번에 여러개의 사진을 predict 할 수도 있다. 
       
            X_batch = Image.fromarray(X_batch)
            X_batch = np.array(X_batch.resize((256, 256))) # PIL lib로 resize후 numpy배열로 반환
            
            X_batch = np.expand_dims(X_batch, axis=0) # (배치사이즈, 노이즈크기) = (1, 256*256)
            X_batch = np.expand_dims(X_batch, axis=3) # (1, 256, 256, 1)
            
            # 나중에는 노이즈를 train_x 즉, 짧은 시간 촬영한 MRI 사진을 넣어야함 
            generated_images = generator.predict(X_batch)
            generated_images = np.squeeze(generated_images, axis=0) # 편의를 위해 
            print('g_img', generated_images.shape)

            # g_img.append(generated_images) # 여러개의 사진 한번에 SR할 때 
    
        return generated_images 

    def pred_start(self, image):
        
        dataset = np.expand_dims(image, axis=0) # 한 장의 이미지를 넣기위해서 / 여러장 한 번에 올 때는 사용 X 
        print(dataset.shape) # (256, 256) -> (1, 256, 256)
        pred_img =self.predict_gan(self.cyclegan, dataset)

        return pred_img





        
        
    
