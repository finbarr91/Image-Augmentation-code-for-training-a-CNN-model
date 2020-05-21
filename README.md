# Image-Augmentation-code-for-training-a-CNN-model
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from sklearn.externals._pilutil import toimage

(X_train, y_train), (X_test,y_test) = cifar10.load_data()



# converting the dataset into a float format

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Just to obtain 8 images fro  the dataset
n = 8
X_train_Sample = X_train[:n]
print(X_train_Sample.shape)

fig = plt.figure

print(X_train.shape)

# doing the main data augmentation using image generator
dataget_train = ImageDataGenerator(rotation_range=90,vertical_flip=True,height_shift_range=0.5,brightness_range=(0,1))

dataget_train.fit(X_train_Sample)

fig = plt.figure(figsize=(20,2))
for x_batch in dataget_train.flow(X_train_Sample, batch_size=n):
    for i in range(0,n):
        ax = fig.add_subplot(1,n,i+1)
        ax.imshow(toimage(x_batch[i]))
        plt.axis('off')
    fig.suptitle('Augmented images (rotated 90 degrees)')
    plt.show()
    break
