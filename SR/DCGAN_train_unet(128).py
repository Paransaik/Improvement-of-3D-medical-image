import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from PIL import Image
import sys, os, math

from tensorflow.python.keras.layers.merge import concatenate
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import voxel

from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape, Input, concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop

# run fodler mkdir
SECTION = 'DCGAN'
DATA_NAME = 'mri'
PARAMETER = '128.10'
RUN_FOLDER = f'run/{SECTION}/'
RUN_FOLDER += '_'.join([DATA_NAME, PARAMETER])
RUN_FOLDER += '/'
print(RUN_FOLDER)
mode = None

if not os.path.exists(RUN_FOLDER):
    # mkdir 함수는 해당 경로가 존재하지 않는다면(중간 경로는 필수로 있어야함) FileNotFoundError 에러가 발생.
    os.makedirs(RUN_FOLDER) # makedirs 함수는 에러 발생하지 않고 경로를 새로 만든다.
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'model'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

    mode = 'bulid'
    
def readPath(path):

    path_x = path + '/train_x'
    path_y = path + '/train_y'

    rawPath_x = [ y for x in os.walk(path_x) for y in glob(os.path.join(x[0], '*.raw'))]
    rawPath_y = [ y for x in os.walk(path_y) for y in glob(os.path.join(x[0], '*.raw'))]

    print('rawPath_x', rawPath_x)
    return rawPath_x, rawPath_y

def readRaw(rawPath_x, rawPath_y): # 채널수 맞출 수 있는 코드 추가(질문 답변 받고)

    image_x = []
    image_y = []

    pyVoxel = voxel.PyVoxel()

    for i in range(len(rawPath_x)): # flatten해야함 noise로 쓰이기 때문 

        # input = (28, 256, 256) -> (2, 28, 256, 256)
        pyVoxel.ReadFromRaw(rawPath_x[i])
        image_x.append(pyVoxel.m_Voxel)

    for i in range(len(rawPath_y)):
        # 20 -> 21이 되더라도 정상적으로 image_y에 들어간다. 
        # input = (20, 512, 512) -> (2, 20, 512, 512)
        pyVoxel.ReadFromRaw(rawPath_y[i])
        image_y.append(pyVoxel.m_Voxel)
    
    image_x = np.array(image_x)
    image_y = np.array(image_y)
    print('train_x :', image_x.shape)
    print('train_y :', image_y.shape)
        
    return image_x, image_y

def noiseProcessing(noise, n_raw, batch_size, cnt=0): # 0 ok 1 -> index 31 is out of bounds for axis 0 with size 30
    nP = []

    for i in range(batch_size):
        temp = Image.fromarray(noise[n_raw][(cnt*batch_size) + (i+4)])
        temp = np.array(temp.resize((128, 128)))
        nP.append(temp)
    
    nP = np.array(nP)
    nP = nP / 4500
    nP = np.expand_dims(nP, axis=-1) # (20, 256, 256) -> (20, 256, 256, 1)
    # nP = np.expand_dims(nP, axis=0)

    return nP

def PSNR(hr_img, g_img):
    g_img = g_img * 4500

    mse = np.mean((hr_img - g_img) ** 2)
    if mse == 0: # MSE is zero means no noise is present in the sigmal
        return 100
    
    max_pixel = 4500.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    return psnr

def epochs_plot(run_folder, epoch, pred_imgs):
    r, c = 4, 5 # 행, 열

    pred_imgs *= 4500 # /4500으로 정규화 된 상태에서 학습 되었기 때문에 원상복귀
    fig, axs = plt.subplots(r, c, figsize=(15, 15))

    for i in range(r):
        for j in range(c):
            if i == 0:
                axs[i,j].imshow(train_y[0][j], cmap='gray')
                axs[i,j].axis('off')

            elif i == 2:
                axs[i,j].imshow(train_y[0][j+5], cmap='gray')
                axs[i,j].axis('off')    

            elif i == 1:
                axs[i,j].imshow(pred_imgs[j], cmap='gray')
                axs[i,j].axis('off')

            elif i == 3:
                axs[i,j].imshow(pred_imgs[j+5], cmap='gray')
                axs[i,j].axis('off')
    
    fig.savefig(os.path.join(run_folder, 'images/{} Epochs.png'.format(epoch)))
    plt.close

def build_generator(input_size=(128, 128, 1)):
    inputs = Input(input_size)
    depth = 32

    conv1 = Conv2D(depth, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(depth, kernel_size=3, strides=1, padding='same', activation='relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1) # 64

    conv2 = Conv2D(depth*2, kernel_size=3, strides=1, padding='same', activation='relu')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(depth*2, kernel_size=3, strides=1, padding='same', activation='relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2) # 32

    conv3 = Conv2D(depth*4, kernel_size=3, strides=1, padding='same', activation='relu')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(depth*4, kernel_size=3, strides=1, padding='same', activation='relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3) # 16

    conv4 = Conv2D(depth*8, kernel_size=3, strides=1, padding='same', activation='relu')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(depth*8, kernel_size=3, strides=1, padding='same', activation='relu')(conv4)
    conv4 = BatchNormalization()(conv4) 
    pool3 = MaxPooling2D(pool_size=(2,2))(conv4) # 8

    conv5 = Conv2D(depth*8, kernel_size=3, strides=1, padding='same', activation='relu')(pool3)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(depth*8, kernel_size=3, strides=1, padding='same', activation='relu')(conv5)
    conv5 = BatchNormalization()(conv5)  # 8

    up1 = concatenate([Conv2DTranspose(depth*4, (3,3), strides=2, padding='same')(conv5), conv4], axis=3)
    dconv1 = Conv2D(depth*4, kernel_size=3, padding='same', activation='relu')(up1)
    dconv1 = BatchNormalization()(dconv1)
    dconv1 = Conv2D(depth*4, kernel_size=3, padding='same', activation='relu')(dconv1)
    dconv1 = BatchNormalization()(dconv1) # 16

    up2 = concatenate([Conv2DTranspose(depth*2, (3,3), strides=2, padding='same')(dconv1), conv3], axis=3)
    dconv2 = Conv2D(depth*2, kernel_size=3, padding='same', activation='relu')(up2)
    dconv2 = BatchNormalization()(dconv2)
    dconv2 = Conv2D(depth*2, kernel_size=3, padding='same', activation='relu')(dconv2)
    dconv2 = BatchNormalization()(dconv2) # 32

    up3 = concatenate([Conv2DTranspose(depth, (3,3), strides=2, padding='same')(dconv2), conv2], axis=3)
    dconv3 = Conv2D(depth, kernel_size=3, padding='same', activation='relu')(up3)
    dconv3 = BatchNormalization()(dconv3)
    dconv3 = Conv2D(depth, kernel_size=3, padding='same', activation='relu')(dconv3)
    dconv3 = BatchNormalization()(dconv3) # 64

    up4 = concatenate([Conv2DTranspose(depth, (3,3), strides=2, padding='same')(dconv3), conv1], axis=3)
    dconv4 = Conv2D(depth, kernel_size=3, padding='same', activation='relu')(up4)
    dconv4 = BatchNormalization()(dconv4)
    dconv4 = Conv2D(depth, kernel_size=3, padding='same', activation='relu')(dconv4)
    dconv4 = BatchNormalization()(dconv4) # 128

    dconv5 = Conv2DTranspose(depth//2, (3,3), strides=2, padding='same')(dconv4)
    dconv5 = Conv2D(depth//2, kernel_size=3, padding='same', activation='relu')(dconv5)
    dconv5 = BatchNormalization()(dconv5)
    dconv5 = Conv2D(depth//2, kernel_size=3, padding='same', activation='relu')(dconv5)
    dconv5 = BatchNormalization()(dconv5) # 256

    dconv6 = Conv2DTranspose(depth//4, (3,3), strides=2, padding='same')(dconv5)
    dconv6 = Conv2D(depth//4, kernel_size=3, padding='same', activation='relu')(dconv6)
    dconv6 = BatchNormalization()(dconv6) # 512

    # outputs = Conv2D(1, (1, 1), activation='sigmoid')(dconv5)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(dconv6) # sigmoid 안한다면 hr_imgs와 값이 맞다.

    return Model(inputs=[inputs], outputs=[outputs])

def build_discriminator():
    
    model = Sequential()
    model.add(Conv2D(1, kernel_size=1, strides=1, padding='same', input_shape=[512, 512, 1], activation='sigmoid'))

    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same'))#, input_shape=[512, 512, 1])) # 256 256
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same')) # 128 128
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same')) # 64 64
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2D(256, kernel_size=5, strides=2, padding='same')) # 32 32
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Flatten()) # 32 * 32 * 128
    model.add(Dense(1, activation='sigmoid'))
    
    return model 

def build_gan(generator, discriminator):

    model = Sequential()

    model.add(generator) # 생성자 판별자 모델 연결 
    model.add(discriminator)

    return model

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', # mse도 생각해보기
                      optimizer=RMSprop(), metrics='accuracy')
discriminator.trainable = False

generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics='accuracy')
generator.summary()
discriminator.summary()

### weights setting ###
if mode == 'bulid': # mode가 'bulid' 즉, 처음 만들어졌다면 현재 상태 저장
    print('□□□ 초기 가중치를 저장합니다... □□□')
    generator.save_weights(os.path.join(RUN_FOLDER + 'weights/g_weights.h5'))
    discriminator.save_weights(os.path.join(RUN_FOLDER + 'weights/d_weights.h5'))
else:
    print('□□□ 저장된 가중치를 불러옵니다... □□□')
    generator.load_weights(os.path.join(RUN_FOLDER, 'weights/g_weights.h5'))
    discriminator.load_weights(os.path.join(RUN_FOLDER, 'weights/d_weights.h5'))

dcgan = build_gan(generator, discriminator)
dcgan.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics='accuracy')
### weights wetting ###

losses = [] # 그래프를 그리기 위해서
accuracies = [] # 일정 간격마다 append 
iteration_checkpoints = []

import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
def train(iterations, batch_size, sample_interval):

    real = np.ones((batch_size, 1)) # shape = (batch_size, 1), not 1 dim
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations): # Epoch라고 생각하면 될 듯 
        print("[{}/{}]번째 Epoch".format(iteration+1, iterations))

        for n_raw, hr_imgs in enumerate(train_y): # hr_imgs = (channel, 512, 512)
            hr_imgs = hr_imgs / 4500
            cnt = 0
            while(1):
                hr_img = hr_imgs[batch_size*cnt:batch_size*(cnt + 1)] #  0 : batch, batch : 2 batch
                
                if len(hr_img) != batch_size:
                    print()
                    break
                else:
                    print(f'{n_raw+1}번째 검사자 [{batch_size*cnt}:{batch_size*(cnt+1)}] batch_size({batch_size})...')

                hr_img = np.expand_dims(hr_img, axis=3) # (batch_size, 512, 512 ,1)
                noise = noiseProcessing(train_x, n_raw, batch_size, cnt)
                # z = np.random.normal(0, 1, (batch_size, 128*128)) # noise 생성 0부터 1까지의 정규 분포 생성 shape = (batch_size, 100)
                    
                gen_imgs = generator.predict(noise) # 학습단계가 아닌 예측을 내보내는 단계기 떄문에 predict

                discriminator.trainable = True # trainable test 
                for i in range(2):
                    d_loss_real = discriminator.train_on_batch(hr_img, real) # 판별자 훈련, trainable = False로 default값이 되어있는데 괜찮은건가 , loss랑 acc를 반환함 (metircs'accurcy'경우)
                    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake) # (x, y)
                    d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake) # 판별자의 손실은 real_loss와 gen_loss로 이루어진다 real이미지와의 결과가 1에 가까울 수록 loss가 줄어들고 gen이미지와의 결과가 0에 가까울 수록 loss가 줄어들도록한다., 0.5를 곱하는 것은 손실값을 줄여 학습 파라미터가 크게 날뛰는 걸 방지하는 용도
                discriminator.trainable = False 

                noise = noiseProcessing(train_x, n_raw, batch_size, cnt) # 여기서 오류
                    
                g_loss = dcgan.train_on_batch(noise, real) # 생성자 훈련, 판별자를 속인것으로 만들어서 generator가 학습할 수 있도록 
                cnt += 1

        if (iteration + 1 ) % sample_interval == 0:

            losses.append((d_loss, g_loss)) 
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            print("{} [ D 손실: {}, 정확도: {}%] [ G 손실: {}\n".format(iteration + 1, d_loss, 100.0 * accuracy, g_loss))
            
            generator.save_weights(os.path.join(RUN_FOLDER + 'weights/g_weights_{}.h5'.format(iteration+1)))
            discriminator.save_weights(os.path.join(RUN_FOLDER + 'weights/d_weights_{}.h5'.format(iteration+1)))
            
            noise = noiseProcessing(train_x, 0, 10)
            pred_imgs = generator.predict(noise) # (10, 512, 512, 1)
            epochs_plot(RUN_FOLDER, iteration+1, pred_imgs)

    generator.save_weights(os.path.join(RUN_FOLDER + 'weights/g_weights.h5'))
    discriminator.save_weights(os.path.join(RUN_FOLDER + 'weights/d_weights.h5'))
    dcgan.save(os.path.join(RUN_FOLDER + 'model/dcgan_model.h5'))
        
Path = './DCGAN_data'

rawPath_x, rawPath_y = readPath(Path)
train_x, train_y = readRaw(rawPath_x, rawPath_y)

train(iterations=5, batch_size=10, sample_interval=2) # batch_size는 20으로 고정해야함


    


