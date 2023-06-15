import os
import glob

dir = glob.glob("./archive/*")

for j in range(40):
    file = glob.glob("./archive/"+str(j+1)+"/*")
    k = 1
    for f in file:
        os.rename(f,"./archive/"+str(j+1)+"/"+str(k)+".jpeg")
        k = k+1