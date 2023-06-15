import os
from PIL import Image
os.mkdir("./atnt_png")
for i in range(40):
    os.mkdir("./atnt_png"+"/"+"s"+str(i+1))
    for j in range(10):
        path = "./atnt"+"/"+"s"+str(i+1)+"/"+str(j+1)+".pgm"
        print(path)
        new_file = "{}.png".format("./atnt_png"+"/"+"s"+str(i+1)+"/"+str(j+1))
        im = Image.open(path)
        im = im.save(new_file)
