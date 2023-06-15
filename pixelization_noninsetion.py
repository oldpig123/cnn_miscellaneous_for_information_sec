import cv2
import os
import numpy
from PIL import Image

dest_dir = "./atnt_dp8"
try:
    os.mkdir(dest_dir)
except:
    pass
pixelization_para = 1 #this is the "b" parameter
b=1
m = 1
epsilon = 8
scale = 127
for i in range(40):
    try:
        os.mkdir(dest_dir+"/s"+str(i+1))
    except:
        pass
    for j in range(10):
        path = "./atnt_png"+"/s"+str(i+1)+"/"+str(j+1)+".png"
        print(path)
        new_file = "{}.png".format(dest_dir+"/s"+str(i+1)+"/"+str(j+1)+"_"+str(pixelization_para)+"x"+str(pixelization_para))
        input = cv2.imread(path)
        output = input
        height, width = input.shape[:2]
        w, h = ((int)(width/pixelization_para)+1, (int)(height/pixelization_para)+1)
        ystep = numpy.linspace(0,width,w,dtype="int")
        xstep = numpy.linspace(0,height,h,dtype="int")
        # print(ystep)
        # print(xstep)
        # print(len(ystep)-1)
        scale = (255*m/(b*b))/epsilon
        
        print(scale)
        for y_index in range(1,len(ystep)):
            for x_index in range(1,len(xstep)):
                # print(y_index)
                # print(x_index)
                # print(ystep[y_index])
                # print(xstep[x_index])
                startx,endx = [xstep[x_index-1],xstep[x_index]]
                starty,endy = [ystep[y_index-1],ystep[y_index]]
                # print([xstep[x_index-1],xstep[x_index]])
                # print([ystep[y_index-1],ystep[y_index]])
                # print(input[starty:endy,startx:endx])
                roi = input[ystep[y_index-1]:ystep[y_index],xstep[x_index-1]:xstep[x_index]]
                # print(roi)
                dp_value = numpy.random.laplace(0.0,scale)
                if pixelization_para != 1:
                    # print(numpy.random.laplace(0.0,(255*m/(pixelization_para*pixelization_para))/epsilon))
                    
                    
                    cv2.rectangle(output,(startx,starty),(endx,endy),[int(x+dp_value) for x in cv2.mean(roi)[:3]],-1)
                else:
                    output[startx,starty] = input[startx,starty]+dp_value


        cv2.imwrite(new_file,output)


