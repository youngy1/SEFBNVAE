import numpy as np
import tifffile
size=80
#with open("./berea/TI_paragan.txt","r") as f:
with open("E:/paperdata/vae/data//snes/snesim9", "r") as f:
    Array1D=f.readlines()[3:]
for no,i in enumerate(Array1D):
    Array1D[no]=i[0]
Array3D=[]
for i in range(size):
    page=Array1D[i*size*size:(i+1)*size*size]
    deal_page=[]
    for j in range(size):
        line=page[j*size:(j+1)*size]
        deal_page.append(line)
    Array3D.append(deal_page)
TI_Array=np.array(Array3D,np.float32)
tifffile.imsave("E:/paperdata/vae/data/snes/snesim9.tiff",TI_Array)
