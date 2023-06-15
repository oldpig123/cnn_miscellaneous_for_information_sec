import tensorflow as tf
import random
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split


def loaddata(dir):

    data = tf.keras.utils.image_dataset_from_directory(dir,batch_size=10)
    

    
    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)

    return (train, val,test)
def load_data_without_test(dir):
    (train,val)=tf.keras.utils.image_dataset_from_directory(dir,batch_size=10,validation_split=0.2,subset="both",seed=np.random.randint(0,100))
    return(train,val)

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
(train, val,test) = loaddata("./atnt_dp1")
# (train, val) = load_data_without_test("./total_image88_dp")
print(train)
test = tf.keras.utils.image_dataset_from_directory("./atnt_png")
# (m,x,test) = loaddata("./atnt_dp8")
# # Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0

# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']

# print(train_images[1])

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     # The CIFAR labels happen to be arrays,
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

model = models.Sequential()

model.add(layers.Rescaling(1.0/255, input_shape=(256,256,3)))
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))


model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(len(train.class_names)))
model.add(layers.Dense(40))
# model.add(layers.Conv2D(16, (3, 3), 1,
#           activation='relu', input_shape=(256, 256, 3)))
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(32, (3, 3), 1, activation='relu'))
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(16, (3, 3), 1, activation='relu'))
# model.add(layers.MaxPooling2D())
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(40, activation='sigmoid'))

# model.add(layers.Rescaling(1.0/255, input_shape=(256,256,3)))
# model.add(layers.Conv2D(16,3,padding='same',activation='relu'))
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(32,3,padding='same',activation='relu'))
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(64,3,padding='same',activation='relu'))
# model.add(layers.MaxPooling2D())
# model.add(layers.Dense(128,activation='relu'))
# model.add


model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # loss='mse',
              metrics=['accuracy'])

history = model.fit(train, epochs=10,batch_size=10,
                    validation_data=val)
print(history.history['val_accuracy'])
print(history.history['val_loss'])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.xticks(range(1,11))
plt.legend(loc='lower right')
# plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
# plt.ylim([0, 4])
plt.xticks(range(1,11))
plt.legend(loc='lower right')
# plt.show()

test_loss, test_acc = model.evaluate(test, verbose=2)

print(test_acc)

