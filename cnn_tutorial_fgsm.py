import tensorflow as tf
import random
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split
import os

def loaddata(dir):

    data = tf.keras.utils.image_dataset_from_directory(dir,batch_size=10)
    

    
    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)

    return (train, val,test)
def load_data(dir):
    seed = np.random.randint(0,100)
    train =tf.keras.utils.image_dataset_from_directory(dir,batch_size=10,validation_split=0.3,subset="training",seed=seed)
    val = tf.keras.utils.image_dataset_from_directory(dir,batch_size=10,validation_split=0.2,subset="validation",seed=seed)
    return(train,val)
def load_data_without_test(dir):
    (train,val)=tf.keras.utils.image_dataset_from_directory(dir,batch_size=10,validation_split=0.2,subset="both",seed=np.random.randint(0,100))
    return(train,val)

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# (train, val,test) = loaddata("./total_image88_dp")
dir = "./total_image1616"
#(train, val) = load_data_without_test(dir)
(train,val) = load_data(dir)
print(train.class_names)


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
model.add(layers.Dense(40,activation='sigmoid'))
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
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
# plt.ylim([0, 4])
plt.xticks(range(1,11))
plt.legend(loc='lower right')
plt.show()



loss_object = tf.keras.losses.CategoricalCrossentropy()
def create_perturbation(single_image, single_label):  #(image,預測結果)
  with tf.GradientTape() as tape:   #用以計算梯度
    tape.watch(single_image)    #想要查看x=single_image時的梯度
    prediction = model(single_image)
    # print(prediction)
    loss = loss_object(single_label, prediction)
  # print(loss)  
  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, single_image)   #在函數為loss的時候，求x=image_1位置的梯度。
  #  print(gradient)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)   #回傳梯度的正負值(可以看做是雜訊)
  return signed_grad    #回傳雜訊

advis = tf.keras.utils.image_dataset_from_directory(dir,batch_size=1)
# print(advis.class_names)
label_list = advis.class_names
# print(train.class_names)

epsilon = [0.1,0.01,0.03]
counter = [0]*40
dir = "./total_image1616"
try:
    os.mkdir(dir+"p1")
    os.mkdir(dir+"p01")
    os.mkdir(dir+"p03")
except:
    pass
try:
    os.mkdir(dir+"_origin")
except:
    pass
for image,labels in advis.take(-1):

        # prob = model.predict(image)
        # print(advis.class_names[(int)(prob.argmax(axis=-1))])
        # plt.imshow(image[0].numpy().astype("uint8"))
        # plt.show()
        # print(labels[0])
        # print(image[0].shape)
        try:
            os.mkdir(dir+"p1/"+str(advis.class_names[int(labels[0])]))
            os.mkdir(dir+"p01/"+str(advis.class_names[int(labels[0])]))
            os.mkdir(dir+"p03/"+str(advis.class_names[int(labels[0])]))
        except:
            pass
        try:
            os.mkdir(dir+"_origin/"+str(advis.class_names[int(labels[0])]))
        except:
            pass
        counter[int(labels[0])] = counter[int(labels[0])]+1
        newfile01 = "{}.png".format(dir+"p1/"+str(advis.class_names[int(labels[0])])+"/"+str(counter[int(labels[0])]))
        newfile001 = "{}.png".format(dir+"p01/"+str(advis.class_names[int(labels[0])])+"/"+str(counter[int(labels[0])]))
        newfile003 = "{}.png".format(dir+"p03/"+str(advis.class_names[int(labels[0])])+"/"+str(counter[int(labels[0])]))
        new_origin_file = "{}.png".format(dir+"_origin/"+str(advis.class_names[int(labels[0])])+"/"+str(counter[int(labels[0])]))
        temp_image = tf.cast(image[0],tf.float32)
        temp_image = image[0][None,...]
        # print(temp_image.shape)
        label = tf.one_hot(labels[0],40)
        label = tf.reshape(label,(1,40))
        # print(label)
        # print(temp_image)
        p = create_perturbation(temp_image,label)
        # print(p)
        perturbation = ((p[0]*0.5+0.5)*255)
        # plt.imshow(perturbation)
        # plt.show()
        # print(perturbation)
        
        # print(adv)
        # plt.imshow(image[0].numpy().astype('uint8'))
        # plt.show()
        # plt.imshow(adv)
        # plt.show()
        adv = image[0]+(p[0]*255)*epsilon[0]
        adv = tf.clip_by_value(adv,0,255)
        cv.imwrite(newfile01,adv.numpy())
        adv = image[0]+(p[0]*255)*epsilon[1]
        adv = tf.clip_by_value(adv,0,255)
        cv.imwrite(newfile001,adv.numpy())
        adv = image[0]+(p[0]*255)*epsilon[2]
        adv = tf.clip_by_value(adv,0,255)
        cv.imwrite(newfile003,adv.numpy())
        cv.imwrite(new_origin_file,image[0].numpy())
        
        # print(image.numpy()[i])
        # print(labels[i])
        # temp_image = image[i][None, ...]
        # perturbations = create_perturbation(temp_image, tf.reshape(labels[i],(1,40)))
        # plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]
        
print("**************************************")
test = tf.keras.utils.image_dataset_from_directory("./total_image1616")
test_loss, test_acc = model.evaluate(test, verbose=2)

print(test_acc)
test = tf.keras.utils.image_dataset_from_directory(dir+"p1")
test_loss, test_acc = model.evaluate(test, verbose=2)

print(test_acc)
test = tf.keras.utils.image_dataset_from_directory(dir+"p03")
test_loss, test_acc = model.evaluate(test, verbose=2)

print(test_acc)
test = tf.keras.utils.image_dataset_from_directory(dir+"p01")
test_loss, test_acc = model.evaluate(test, verbose=2)

print(test_acc)

