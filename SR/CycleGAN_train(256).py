from glob import glob
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.python.ops.gen_nn_ops import Relu

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import voxel
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, UpSampling2D, Dropout, Concatenate, Input
from tensorflow.keras.layers import add
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.utils import plot_model
# run fodler mkdir
SECTION = 'CycleGAN'
DATA_NAME = 'mri'
PARAMETER = '256.1.resnet_RFA'
RUN_FOLDER = f'run/{SECTION}/'
RUN_FOLDER += '_'.join([DATA_NAME, PARAMETER])
RUN_FOLDER += '/'
print(RUN_FOLDER)
mode = None

if not os.path.exists(RUN_FOLDER):
    # mkdir 함수는 해당 경로가 존재하지 않는다면(중간 경로는 필수로 있어야함) FileNotFoundError 에러가 발생.
    os.makedirs(RUN_FOLDER) # makedirs 함수는 에러 발생하지 않고 경로를 새로 만든다.
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'raws'))
    os.mkdir(os.path.join(RUN_FOLDER, 'model'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

    mode = 'bulid'
    
class CycleGAN:
    def __init__(self, Path, iterations, batch_size, sample_interval, input_dim, n_filter
                 , learning_rate
                 , lambda_validation
                 , lambda_reconstr
                 , lambda_id):
        self.PATH = Path

        self.pyVoxel = voxel.PyVoxel()

        self.pre_voxel_x = [] # len() == 환자 수
        self.pre_voxel_y = [] # len() == 환자 수

        self.iterations = iterations
        self.batch_size = batch_size
        self.sample_interval = sample_interval

        self.shape = input_dim # generator shape option
        self.pixel_shape = (batch_size, input_dim[2], input_dim[0], input_dim[1]) # = (B, C, H, W)
        self.filters = n_filter

        self.learning_rate = learning_rate

        self.lambda_validation = lambda_validation
        self.lambda_reconstr = lambda_reconstr
        self.lambda_id = lambda_id

        self.train_x = None # None -> ndarray
        self.train_y = None # None -> ndarray

        # generator, discriminator object creation
        self.g_AB, self.g_BA = self.bulid_generator_resnet(), self.bulid_generator_resnet()
        self.d_A, self.d_B = self.bulid_discriminator(), self.bulid_discriminator()

        # self.g_AB.summary()
        # self.g_BA.summary()
        # self.d_A.summary()
        # self.d_B.summary()

        # discriminator compile
        self.d_A.compile(loss='binary_crossentropy', optimizer=Adam(self.learning_rate, 0.5), metrics=['accuracy']) # loss, optimizer fucntion 수정 필요
        self.d_B.compile(loss='binary_crossentropy', optimizer=Adam(self.learning_rate, 0.5), metrics=['accuracy'])
        self.d_A.trainable = False
        self.d_B.trainable =False

        ### weights setting ###
        if mode == 'bulid': # mode가 'bulid' 즉, 처음 만들어졌다면 현재 상태 저장
            print('□□□ 초기 가중치를 저장합니다... □□□')
            self.g_AB.save_weights(os.path.join(RUN_FOLDER + 'weights/g_AB_weights.h5'))
            self.g_BA.save_weights(os.path.join(RUN_FOLDER + 'weights/g_BA_weights.h5'))
            self.d_A.save_weights(os.path.join(RUN_FOLDER + 'weights/d_A_weights.h5'))
            self.d_B.save_weights(os.path.join(RUN_FOLDER + 'weights/d_B_weights.h5'))
        else:
            print('□□□ 저장된 가중치를 불러옵니다... □□□')
            self.g_AB.load_weights(os.path.join(RUN_FOLDER + 'weights/g_AB_weights.h5'))
            self.g_BA.load_weights(os.path.join(RUN_FOLDER + 'weights/g_BA_weights.h5'))
            self.d_A.load_weights(os.path.join(RUN_FOLDER + 'weights/d_A_weights.h5'))
            self.d_B.load_weights(os.path.join(RUN_FOLDER + 'weights/d_B_weights.h5'))
        ### weights wetting ###

        img_A = Input(shape = self.shape) # noise?
        img_B = Input(shape = self.shape) # noise?

        fake_A = self.g_BA(img_B) # noise to fake?
        fake_B = self.g_AB(img_A) 

        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B) # 유효성 

        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A) # 재구성

        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B) #동일성 (배경, 큰 틀을 유지하기 위해 제약을 둠, 이미지의 변환이 필요한 부분 이외에는 바꾸지 않도록)

        self.combined = Model(inputs = [img_A, img_B], outputs = [valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id])

        self.combined.compile(loss = ['mse','mse', 'mae', 'mae', 'mae', 'mae'], 
                            loss_weights = [self.lambda_validation,
                                            self.lambda_validation,
                                            self.lambda_reconstr,
                                            self.lambda_reconstr,
                                            self.lambda_id,
                                            self.lambda_id],
                            optimizer = Adam()) 
        self.combined.summary()

        # plot_model(self.g_AB, to_file = "g_AB.png", show_shapes = True, show_layer_names = True)
        # plot_model(self.g_BA, to_file = "g_BA.png", show_shapes = True, show_layer_names = True)
        # plot_model(self.d_A, to_file = "d_A.png", show_shapes = True, show_layer_names = True)
        # plot_model(self.d_B, to_file = "d_B.png", show_shapes = True, show_layer_names = True)
                
    def read_path(self):
        path_x = self.PATH + '/train_x'
        path_y = self.PATH + '/train_y'

        raw_path_x = [ y for x in os.walk(path_x) for y in glob(os.path.join(x[0], '*.raw'))]
        raw_path_y = [ y for x in os.walk(path_y) for y in glob(os.path.join(x[0], '*.raw'))]

        print('raw_path_x', raw_path_x)
        return raw_path_x, raw_path_y

    def read_raw(self, raw_path_x, raw_path_y):
        image_x = []
        image_y = []

        # pyVoxel = voxel.PyVoxel()

        for i in range(len(raw_path_x)):
            # input = (28, 256, 256) -> (n, 28, 256, 256)
            self.pyVoxel.ReadFromRaw(raw_path_x[i]) # clip 기능 추가 완료(0~3500)
            self.pre_voxel_x.append(self.pyVoxel.m_Voxel) # MinMax전 voxel 값 저장

            self.pyVoxel.NormalizeMM() 
            image_x.append(self.pyVoxel.m_Voxel)  # MinMax 정규화 완료 된 voxel 값

        for i in range(len(raw_path_y)):
            self.pyVoxel.ReadFromRaw(raw_path_y[i])
            self.pre_voxel_y.append(self.pyVoxel.m_Voxel)

            self.pyVoxel.NormalizeMM()
            image_y.append(self.pyVoxel.m_Voxel)

        image_x = np.array(image_x)
        image_y = np.array(image_y)
        print('image_x :', image_x.shape)
        print('image_y :', image_y.shape)

        return image_x, image_y

    def noise_processing(self, noise, n_dim):
        nP = Image.fromarray(noise)
        nP = np.array(nP.resize((n_dim, n_dim)))

        nP = np.expand_dims(nP, axis=0)
        nP = np.expand_dims(nP, axis=3)

        return nP

    def psnr(self):
        pass

    def epochs_plot(self, run_folder, epoch, plot_A, plot_B):
        r, c = 3, 4 # 행, 열
        plot_A = np.array(plot_A)
        
        for img in plot_A: # /4500으로 정규화 된 상태에서 학습 되었기 때문에 원상복귀
            img *= 3500

        for img in plot_B:
            img *= 3500

        fig, axs = plt.subplots(r, c, figsize=(15, 15))

        for j in range(c): # 0 1 2 3 
            if j < 2:
                axs[0,j].imshow(plot_A[j][0], cmap='gray') # original
                axs[0,j].axis('off')
                axs[0,j].set_title('lr_img', fontsize=25)

                axs[1,j].imshow(plot_A[j][1], cmap='gray') # fake
                axs[1,j].axis('off')
                axs[1,j].set_title('fake_hr_img', fontsize=25)

                axs[2,j].imshow(plot_A[j][2], cmap='gray') # reconstr
                axs[2,j].axis('off')
                axs[2,j].set_title('reconstr_img', fontsize=25)
            else:
                axs[0,j].imshow(plot_B[j-2][0], cmap='gray') # original
                axs[0,j].axis('off')
                axs[0,j].set_title('hr_img', fontsize=25)

                axs[1,j].imshow(plot_B[j-2][1], cmap='gray') # fake
                axs[1,j].axis('off')
                axs[1,j].set_title('fake_lr_img', fontsize=25)

                axs[2,j].imshow(plot_B[j-2][2], cmap='gray') # reconstr
                axs[2,j].axis('off')
                axs[2,j].set_title('reconstr_img', fontsize=25)
        
        fig.savefig(os.path.join(run_folder, 'images/{} Epochs.png'.format(epoch)))
        plt.close 
    
    def pixel_shuffle(self, x, rx=2, ry=2):
        [B, C, H, W] = list(x.shape)
        print('1', list(x.shape))
        
        x = tf.reshape(x, shape=[B, C // (ry*rx), ry, rx, H, W])
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C // (ry*rx), H * ry, W * rx)

        print('2', list(x.shape))
        return x

    def bulid_generator_resnet(self):
        def down_sampling(input_layer, filters, k_size=3):
            d = Conv2D(filters, kernel_size=k_size, strides=2, padding='same')(input_layer)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)
            return d 
        
        def up_sampling(input_layer, filters, k_size=3, dropout_rate=0): # pixel_shuffle
            u = Conv2D(filters*2*(2**2), (3, 3), kernel_initializer='he_normal', padding='same')(input_layer)
            u = tf.nn.depth_to_space(u, 2)
            u = LeakyReLU(alpha=0.3)(u)

            u = Conv2D(filters, kernel_size=k_size, strides=1, padding='same')(u)
            u = BatchNormalization()(u)
            u = Activation('relu')(u)

            if dropout_rate:
                u = Dropout(dropout_rate)(u)
    
            # u = Concatenate()([u, skip_input])
    
            return u

        def conv_block(input_layer, filters, k_size=3):
            # conv
            x = Conv2D(filters, kernel_size=k_size, kernel_initializer='he_normal', strides=1, padding='same')(input_layer)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(filters, kernel_size=k_size, kernel_initializer='he_normal', strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(filters, kernel_size=k_size, kernel_initializer='he_normal', strides=1, padding='same')(x)
            x = BatchNormalization()(x)

            return x

        def aggregation(input_layer, filters):
            identity = input_layer

            conv1 = conv_block(input_layer, filters)
            res1 = add([conv1, input_layer])

            conv2 = conv_block(res1, filters)
            res2 = add([conv2, res1])

            conv3 = conv_block(res2, filters)
            res3 = add([conv3, res2])

            conv4 = conv_block(res3, filters)

            aggregate = Concatenate()([conv1, conv2, conv3, conv4])

            final = Conv2D(filters, (1,1), kernel_initializer='he_normal', padding='same', activation='relu')(aggregate) # why kernel_size (1,1)?

            return final + identity
        
        input = Input(shape = self.shape)

        # DownSampling
        d1 = down_sampling(input, self.filters) # 256 -> 128
        d2 = down_sampling(d1, self.filters * 2) # 128 -> 64

        # Residual
        a1 = aggregation(d2, self.filters*2)
        a2 = aggregation(a1, self.filters*2)

        # Up Sampling
        u1 = up_sampling(a2, self.filters) # 64 -> 128, skip_connection 추가하기 즉, resnet + unet
        u2 = up_sampling(u1, self.filters//2) # 128 -> 256

        output = Conv2D(1, (9,9), padding='same', activation='sigmoid')(u2)

        return Model(inputs=[input], outputs=[output])

    def bulid_generator_unet(self): 
        def down_sampling(input_layer, filters, k_size=3):
            d = Conv2D(filters, kernel_size=k_size, strides=2, padding='same')(input_layer)
            d = BatchNormalization()(d) # InstanceNormalization이 import 되지 않아 임시용
            d = Activation('relu')(d)
            return d
    
        def up_sampling(input_layer, skip_input, filters, k_size=3, dropout_rate=0):
            u = Conv2D(filters * 2 * (2 ** 2), (3, 3), kernel_initializer='he_normal', padding='SAME')(input_layer)
            u = tf.nn.depth_to_space(u, 2)
            u = LeakyReLU(alpha=0.3)(u)
            # u = UpSampling2D(size=2)(input_layer) # don't change channel range
            u = Conv2D(filters, kernel_size=k_size, strides=1, padding='same')(u)
            u = BatchNormalization()(u)
            u = Activation('relu')(u)

            if dropout_rate:
                u = Dropout(dropout_rate)(u)

            u = Concatenate()([u, skip_input])

            return u

        input = Input(shape = self.shape)

        # DownSampling
        d1 = down_sampling(input, self.filters) # 256 -> 128
        d2 = down_sampling(d1, self.filters * 2) # 128 -> 64
        d3 = down_sampling(d2, self.filters * 4) # 64 -> 32
        d4 = down_sampling(d3, self.filters * 8) # 32 -> 16
        d5 = down_sampling(d4, self.filters * 16) # 16 -> 8 8까지 줄여서 얻을 수 있는게 많이 없다.
        # Y' = G(X)' + X' 

        # UpSampling
        u1 = up_sampling(d5, d4, self.filters * 4) # 8 -> 16
        u2 = up_sampling(u1, d3, self.filters * 2) # 16 -> 32
        u3 = up_sampling(u2, d2, self.filters) # 32 -> 64
        u4 = up_sampling(u3, d1, self.filters // 2) # 64 -> 128
        u5 = Conv2D(self.filters // 2 * 2 * (2 ** 2), (3, 3), kernel_initializer='he_normal', padding='SAME')(u4)
        u5 = tf.nn.depth_to_space(u5, 2)
        u5 = LeakyReLU(alpha=0.3)(u5)
        # u5 = UpSampling2D(size=2)(u4) # 128 -> 256

        output = Conv2D(1, (9,9), activation='sigmoid')(u5)

        return Model(inputs=[input], outputs=[output])
        
    def bulid_discriminator(self):
        def classifier(input_layer, filters, strides, norm=True):
            c = Conv2D(filters, kernel_size=3, kernel_initializer='he_normal', strides=strides, padding='same')(input_layer)

            if norm:
                c = BatchNormalization()(c)
            
            c = LeakyReLU(0.2)(c)

            return c
        nKernels = 32
        input = Input(shape=(256, 256,1))

        y = classifier(input, self.filters, strides=2, norm=False) # 256 -> 128
        y = classifier(y, self.filters*2, strides=2) # 128 -> 64
        y = classifier(y, self.filters*4, strides=2) # 64 -> 32
        y = classifier(y, self.filters*8, strides=2) # 32 -> 16

        output = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(y) # 16

        return Model(input, output)
        
    def train(self):
        patch = int(self.shape[0]/2**4)
        self.disc_patch = (patch, patch, 1)

        real = np.ones((self.batch_size,) + self.disc_patch)
        fake = np.ones((self.batch_size,) + self.disc_patch)

        for iteration in range(self.iterations):
            iteration = iteration # + n # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print(f'[{iteration+1}/{self.iterations}]번째 Epoch')
            
            for n_raw, hr_imgs in enumerate(self.train_y):
                print()
                cnt = 0

                for n_channel, hr_img in enumerate(hr_imgs):
                    print(f'{n_raw+1}번째 검사자 [{self.batch_size*cnt}:{self.batch_size*(cnt+1)}] batch_size({self.batch_size})...')

                    imgs_A = self.noise_processing(self.train_x[n_raw][n_channel+4], 256)
                    imgs_B = self.noise_processing(hr_img, n_dim=256)

                    fake_B = self.g_AB.predict(imgs_A) # 256 -> 512 만드는 g
                    fake_A = self.g_BA.predict(imgs_B) # 512 -> 256 만드는 g

                    # 가짜 이미지와 진짜 이미지 배치로 각 판별자를 훈련
                    self.d_A.trainable = True
                    self.d_B.trainable = True

                    dA_loss_real = self.d_A.train_on_batch(imgs_A, real)
                    dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                    dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                    dB_loss_real = self.d_B.train_on_batch(imgs_B, real)
                    dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                    dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                    d_loss = 0.5 * np.add(dA_loss, dB_loss)

                    self.d_A.trainable = False
                    self.d_B.trainable = False

                    # generator가 학습이 안된다. 즉, 계속 같은 이미지만 출력(판별자는 학습 되므로 loss 변화)
                    g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [real, real, imgs_A, imgs_B, imgs_A, imgs_B])
                    cnt += 1

            if (iteration + 1) % self.sample_interval == 0:
                print("{} [ D 손실: {}] [ G 손실: {}]".format(iteration+1, d_loss, g_loss))
                
                temp_R = []
                temp_A, temp_B = [], []

                for i in range(2):
                    test_A = self.noise_processing(self.test_x[0][i+4], 256)
                    fake_B = self.g_AB.predict(test_A)
                    reconstr_A = self.g_BA.predict(fake_B)

                    test_A = np.squeeze(test_A, axis=0)
                    fake_B = np.squeeze(fake_B, axis=0)
                    reconstr_A = np.squeeze(reconstr_A, axis=0)

                    temp_A.append([test_A, fake_B, reconstr_A])

                for i in range(2):
                    test_B = self.noise_processing(self.test_y[0][i], 256)
                    fake_A = self.g_BA.predict(test_B)
                    reconstr_B = self.g_AB.predict(fake_A)

                    test_B = np.squeeze(test_B, axis=0)
                    fake_A = np.squeeze(fake_A, axis=0)
                    reconstr_B = np.squeeze(reconstr_B, axis=0)

                    temp_B.append([test_B, fake_A, reconstr_B])

                for i in range(20):
                    test_R = self.noise_processing(self.test_x[0][i+4], 256)
                    fake_R = self.g_AB.predict(test_R)
                    reconstr_R = np.squeeze(self.g_BA.predict(fake_R), axis=0)
                    # fake_R = np.squeeze(self.g_AB.predict(test_R), axis=0) # (256, 256, 1)
                   
                    temp_R.append(reconstr_R)   

                self.epochs_plot(RUN_FOLDER, iteration+1, temp_A, temp_B)
                self.save_raw(RUN_FOLDER, iteration+1, np.array(temp_R))
                
                self.g_AB.save_weights(os.path.join(RUN_FOLDER + 'weights/g_AB_weights_{}.h5'.format(iteration+1)))
                self.g_BA.save_weights(os.path.join(RUN_FOLDER + 'weights/g_BA_weights_{}.h5'.format(iteration+1)))
                self.d_A.save_weights(os.path.join(RUN_FOLDER + 'weights/d_A_weights_{}.h5'.format(iteration+1)))
                self.d_B.save_weights(os.path.join(RUN_FOLDER + 'weights/d_B_weights_{}.h5'.format(iteration+1)))
            
        self.g_AB.save_weights(os.path.join(RUN_FOLDER + 'weights/g_AB_weights.h5'))
        self.g_BA.save_weights(os.path.join(RUN_FOLDER + 'weights/g_BA_weights.h5'))
        self.d_A.save_weights(os.path.join(RUN_FOLDER + 'weights/d_A_weights.h5'))
        self.d_B.save_weights(os.path.join(RUN_FOLDER + 'weights/d_B_weights.h5'))
        # g_AB model을 저장함. predict에서 불러서 predict 하면 될 듯 
        self.g_AB.save(os.path.join(RUN_FOLDER + 'model/g_AB_model.h5'))

    def save_raw(self, run_folder,  epoch, images):
        path = run_folder + 'raws/' + ('%d Epochs.raw'%epoch)
        self.pyVoxel.NumpyArraytoVoxel_float(images)
        self.pyVoxel.inverse_NormalizeMM(self.pre_voxel_x[0]) # [0]의 이유는 fake를 만들 때 train_x[0]을 했기 때문
        
        self.pyVoxel.WriteToRaw(path) # 정규화 풀어준 후 저장

    def start(self): # 128, epoch, batch_size
        raw_path_x, raw_path_y = self.read_path()
        data_x, data_y = self.read_raw(raw_path_x, raw_path_y)
        self.train_x, self.test_x = data_x[:22], data_x[22:]
        self.train_y, self.test_y = data_y[:22], data_y[22:]

        self.train()
        
Path = './DCGAN_data'
test = CycleGAN(Path, 
                iterations=10, 
                batch_size=1, 
                sample_interval=1, 
                input_dim=(256,256,1),
                n_filter=32, # 256기준
                learning_rate=0.0002,
                lambda_validation=1,
                lambda_reconstr=10,
                lambda_id=2
                )

test.start()