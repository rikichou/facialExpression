import sys,os,shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from matplotlib.pyplot import imsave
from matplotlib import cm

# ead data csv
data_file_path = '/media/ruiming/data/workspace/pro/facialExpression/data/fer2013/icml_face_data.csv'
data = pd.read_csv(data_file_path)
print(data.head())

emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
cols = ["emotion","subset","pixels"]
data.columns = cols
print(data.columns)
print(data.subset.value_counts())
print(data.emotion.value_counts())

# get train and test data
train_data = data[data["subset"]=="Training"]
val_data = data[data["subset"]=="PublicTest"]
test_data = data[data["subset"]=="PrivateTest"]
# train_data = train_data[:10]
# test_data = test_data[:10]
# val_data = val_data[:10]
print(train_data.shape,test_data.shape,val_data.shape)

# elp function
# def toPixels(pixels):
#     arr = np.array(pixels.split(),"float64")
#     return arr
# def reshapetoImage(data):
#     Images = np.reshape(data["pixels"].to_list(),(data.shape[0],48,48,1))
#     return Images

def toPixels(pixels):
    arr = np.array(pixels.split(),"uint8")
    return arr

def reshapetoImage(data):
    Images = np.reshape(data["pixels"].to_list(),(data.shape[0],48,48))
    return Images

def plotImages(x,y):
    plt.figure(figsize=[20,12])
    for i in range(5):
        plt.subplot(5,5,i+1)
        plt.imshow(x[i],cmap="gray")
        plt.title(emotions[y[i]])
        plt.axis("off")
    plt.show()

train_data["pixels"] = train_data["pixels"].apply(toPixels)
val_data["pixels"] = val_data["pixels"].apply(toPixels)
test_data["pixels"] = test_data["pixels"].apply(toPixels)

train_images = reshapetoImage(train_data)
val_images = reshapetoImage(val_data)
test_images = reshapetoImage(test_data)

train_labels = list(train_data["emotion"])
val_labels = list(val_data["emotion"])
test_labels = list(test_data["emotion"])

def save_images(out_base_dir, images, labels):
    # check and makedir
    if not os.path.exists(out_base_dir):
        os.makedirs(out_base_dir)
    # save images
    for idx,img in enumerate(images):
        # get img save path
        out_img_dir = os.path.join(out_base_dir, emotions[labels[idx]])
        if not os.path.exists(out_img_dir):
            os.mkdir(out_img_dir)
        out_img_name = '{:0>6d}'.format(idx)+'.jpg'
        cv2.imwrite(os.path.join(out_img_dir, out_img_name), images[idx])

out_dir = '/media/ruiming/data/workspace/pro/facialExpression/data/fer2013/images'
save_images(os.path.join(out_dir, 'train'), train_images, train_labels)
save_images(os.path.join(out_dir, 'val'), val_images, val_labels)
save_images(os.path.join(out_dir, 'test'), test_images, test_labels)

# i = 0
# cv2.imshow(emotions[train_labels[i]], train_images[i].reshape((48,48)).astype(np.uint8))
# cv2.imwrite('./after.jpg', train_images[i])


