Python implementation code for the paper titled,

Title: 3D reconstruction of porous media using a batch normalized variational auto-encoder

Authors: Ting Zhang1, Yi Yang1, Anqin Zhang1, *

1.College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

(*corresponding author, E-mail: shiep2021@yeah.net)

Ting Zhang Email: tingzh@shiep.edu.cn, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Yi Yang: young_y1@126.com, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Anqin Zhang Email: shiep2021@yeah.net, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China


## SE-FBN-VAE
1.requirements

Tensorflow-gpu == 1.15.1

To run the code, an NVIDIA  GPU video card with 8GB video memory is required.

Software development environment should be any Python integrated development environment used on an NVIDIA video card.

Programming language: Python 3.6.


2.Usage

First, prepare training image: Cut the porous media slice into 80 * 80 * 80 size pictures or a .tif file. Then, for serial slices, use pre/image_to_data.py to convert the serial images into .txt format. For .tif format file, use pre/tif2txt.py to convert the .tif file into .txt format. (we have given an example in /save/ti.txt)

Secondly, set the network parameters such as data path, learning rate and storage location. The path of executable .py file  is: code/bn_vae_att.py. After configuring the parameters and environment, you can run directly: python bn_vae_att.py

Finally, in /save/savepoint, the trained model is stored here. Find the  .txt format of the porous media three-dimensional structure images in /save/output. Use pro/txtTOtiff to convert .txt to .tif format (if need) or use .txt directly for later analysis visualization and processing.
