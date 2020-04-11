import numpy as np
import keras
import matplotlib.pyplot as plt
import csv
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

folder = './data/'
path = folder + 'driving_log.csv'

rows = 66
cols = 200


batch_size = 32
nb_epoch = 50

log_tokens = []
with open(path,'rt') as f:
    reader = csv.reader(f)
    for line in reader:
        log_tokens.append(line)
log_labels = log_tokens.pop(0)

###########################################################    

# Resize the training images to remove irrelevant data from the top portion and the Car bonnet from bottom
def crop_image(image):
    #New sizes for image, 
    col, row = 200,66
    
    shape = image.shape
    
    #Cut off the sky and tree portion from the original picture
    crop_up = int(shape[0]/5)
    
    #Cut off the Bottom front of the car
    crop_down = int(shape[0]-25)

    image = image[crop_up:crop_down, 0:shape[1]]
    image = cv2.resize(image,(col,row), interpolation=cv2.INTER_AREA)    
    return image
########### End of Crop image function ################

##### Image file preprocessing ############
def preprocess_color(image):
  # Preprocessing image files and augmenting
  
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    return image
########### End of preprocess_image_file_color() ###############

##### Image file preprocessing ############
def preprocess_flipYaxis(image):
    # flip the image on Y-axis for augumentation
    
    image = cv2.flip(image, 1)
    
    return image
########### End of preprocess_flipYaxis() ###############

##### Image file preprocessing and Augumentation Brightness ############
def preprocess_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

########### End of preprocess_Brightness ###############

##### Image file preprocessing and Augumentation Horizontal and Vertical Shifts ############
def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr,steer_ang
########### End of preprocess_Horizontal and Vertical shifts ###############

##### Image file preprocessing and Augumentation Random Shadow ############

def preprocess_random_shadow(image):
    top_y = 200*np.random.uniform()
    top_x = 0
    bot_x = 66
    bot_y = 200*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

########### End of preprocess_Random Shadow ###############

def data_loading(imgs,steering,folder,correction=0.08):
# Loading log tokens    
    log_tokens = []
    with open(path,'rt') as f:
        reader = csv.reader(f)
        for line in reader:
            log_tokens.append(line)
    log_labels = log_tokens.pop(0)

# Using for loop for loading and appending centre images with steering angles
    for i in range(len(log_tokens)):
        img_path = log_tokens[i][0]
        img_path = folder+'IMG'+(img_path.split('IMG')[1]).strip()
        img = plt.imread(img_path)
        
        #Call crop function to remove irrelevant part from image
        img = crop_image(img)        
        imgs.append(img)
        steering.append(float(log_tokens[i][3]))
        
     
        img = preprocess_color(img)
        imgs.append(img)
        steering.append(float(log_tokens[i][3]))
        
        img = preprocess_brightness_camera_images(img)
        imgs.append(img)
        steering.append(float(log_tokens[i][3]))
        
        img = preprocess_random_shadow(img)
        imgs.append(img)
        steering.append(float(log_tokens[i][3]))
        
        
# Using for loop for loading and appending left images with steering angles and adding a little correction
    for i in range(len(log_tokens)):
        img_path = log_tokens[i][1]
        img_path = folder+'IMG'+(img_path.split('IMG')[1]).strip()
        img = plt.imread(img_path)
        
        #Call crop function to remove irrelevant part from image
        img = crop_image(img)
        
        imgs.append(img)
        steering.append(float(log_tokens[i][3]) + correction)
        
        img = preprocess_flipYaxis(img)
        imgs.append(img)
        steering.append(-1.*float(log_tokens[i][3]))
        
        img = preprocess_brightness_camera_images(img)
        imgs.append(img)
        steering.append(float(log_tokens[i][3]) + correction)
        
        img = preprocess_random_shadow(img)
        imgs.append(img)
        steering.append(float(log_tokens[i][3]))
        
# Using for loop for loading and appending right images with steering angles and subtracting a little correction
    for i in range(len(log_tokens)):
        img_path = log_tokens[i][2]
        img_path = folder+'IMG'+(img_path.split('IMG')[1]).strip()
        img = plt.imread(img_path)
        
        #Call crop function to remove irrelevant part from image
        img = crop_image(img)
        
        imgs.append(img)
        steering.append(float(log_tokens[i][3]) - correction)
        
        img = preprocess_flipYaxis(img)
        imgs.append(img)
        steering.append(-1.*float(log_tokens[i][3]))
        
        img = preprocess_brightness_camera_images(img)
        imgs.append(img)
        steering.append(float(log_tokens[i][3]) + correction)
        
        img = preprocess_random_shadow(img)
        imgs.append(img)
        steering.append(float(log_tokens[i][3]))
        
def main():
    # Loading data    
    images_train = np.array(data['Images']).astype('float32')
    steering_train = np.array(data['Steering']).astype('float32')
    
    # shuffling the data for avoiding overfit
    X_train, y_train = shuffle(images_train, steering_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.2)
    
    # reshaping the shape of the images to the ones we want to input here it is 16X16
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 3)
    X_val = X_val.reshape(X_val.shape[0], rows, cols, 3)
    
    ################## Referred from KERAS Documentation - https://keras.io/preprocessing/image/ ################
    train_datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.02,
        fill_mode='nearest')    
    
    train_generator = train_datagen.flow(X_train, y_train, batch_size)  

    valid_datagen = ImageDataGenerator()

    validation_generator = valid_datagen.flow(X_val, y_val, batch_size)
   
    # the model is referred from NVIDIA's "End to End Learning for Self-Driving Cars" paper

    model = Sequential()
    model.add(Lambda(lambda x:x/127.5 -1., input_shape = (rows,cols,3)))   
    model.add(Cropping2D(cropping=((70,25),(0,0))))   # trim image to only see section with road
    model.add(Conv2D(24,(5,5),input_shape=(rows,cols,3),padding = 'valid' ))
    model.add(ELU())
    model.add(Conv2D(36,(5,5), padding ='valid',strides =(2,2)))
    model.add(ELU())
    model.add(Conv2D(48,(5,5), padding ='valid',strides=(2,2)))
    model.add(ELU())
    model.add(Conv2D(64,(3,3), padding ='valid',strides=(1,1)))
    model.add(ELU())
    model.add(Conv2D(64,(3,3), padding ='valid',strides=(1,1)))
    model.add(ELU()) #Change -1
    model.add(Flatten())
    model.add(Dropout(0.5)) #Change -2 : Add dropout
    model.add(Dense(1164,kernel_initializer ='he_normal'))
    model.add(ELU())
    model.add(Dense(100,kernel_initializer='he_normal'))
    model.add(ELU())
    model.add(Dense(50,kernel_initializer ='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5)) #Change -3 : Add dropout
    model.add(Dense(10,kernel_initializer ='he_normal'))
    model.add(ELU())
    model.add(Dense(1,name='output',kernel_initializer = 'he_normal'))
    
    model.summary()
    # loss-mse and optimizer-adam
    model.compile(loss='mean_squared_error',optimizer='adam')
        
    #### Model fit generator ##########
    # train the network
    
       
    model.fit_generator(
        train_generator,
        steps_per_epoch = len(X_train)/batch_size,
        epochs = nb_epoch,
        validation_data = validation_generator,
        validation_steps = batch_size)
    
      
    #saving the model
    model.save("model.h5")
    print("Model saved.")
    
if __name__ == '__main__':
    
    data={}
    data['Images'] = []
    data['Steering'] = []

    data_loading(data['Images'], data['Steering'],folder,0.3)
    main()