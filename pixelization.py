import cv2
import os
from PIL import Image
# os.mkdir("./atnt_16x16")
for i in range(40):
    # os.mkdir("./atnt_16x16"+"/"+"s"+str(i+1))
    for j in range(20):
        path = "./total_image_ver3"+"/"+str(i+1)+"/"+str(j+1)+".jpg"
        print(path)
        new_file = "{}.jpg".format("./total_image_ver3"+"/"+str(i+1)+"/"+str(j+1)+"_16x16")
        input = cv2.imread(path)
        height, width = input.shape[:2]
        w, h = ((int)(width/16), (int)(height/16))
        temp = cv2.resize(input, (w, h), interpolation=cv2.INTER_LINEAR)
        output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)


        cv2.imwrite(new_file,output)


