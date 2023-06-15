import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

def loaddata(dir):
    data = tf.keras.utils.image_dataset_from_directory(dir)
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])
    plt.show()
    return 


if __name__ == '__main__':
    loaddata("./atnt")
    