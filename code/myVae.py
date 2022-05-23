import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from PIL import Image
import os
from mpl_toolkits.mplot3d import Axes3D
import math

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
alpha = 0.8

def get_input(path):
    f = open(path, 'r')
    lines = f.readlines()
    data = np.zeros(shape=(80, 80, 80), dtype=np.float)
    x = 0
    y = 0
    z = 0
    for i in lines:
        data[x, y, z] = float(i)
        x = x + 1
        if x == 80:
            y = y + 1
            x = 0
        if y == 80:
            z = z + 1
            y = 0
    input = []
    lx = 0
    ly = 0
    lz = 0
    step = 40
    max  = 80
    while lz+40<=max:
        ly = 0
        while ly+40<=max:
            lx = 0
            while lx+40<=max:
                input.append(data[lx:lx+step,ly:ly+step,lz:lz+step])
                lx = lx+step
            ly = ly+step
        lz = lz+step
    input = np.array(input)
    return input



def encode(input,istrain=True):
    with tf.variable_scope("encoder",reuse=(not istrain)):
        layer1 = tf.layers.conv3d(inputs=input,filters=64,kernel_size=(3,3,3),strides = 1,padding='valid')
        layer1 = tf.layers.batch_normalization(layer1, training=istrain)
        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob=0.7)
        layer1 = tf.nn.relu(layer1)

        layer2 = tf.layers.conv3d(inputs=layer1, filters=128, kernel_size=(5, 5, 5), strides=2, padding='valid')
        layer2 = tf.layers.batch_normalization(layer2, training=istrain)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.7)
        layer2 = tf.nn.relu(layer2)

        layer3 = tf.layers.conv3d(inputs=layer2, filters=256, kernel_size=(5, 5, 5), strides=2, padding='valid')
        layer3 = tf.layers.batch_normalization(layer3, training=istrain)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.7)
        layer3 = tf.nn.relu(layer3)

        layer4 = tf.layers.conv3d(inputs=layer3, filters=512, kernel_size=(6, 6, 6), strides=2, padding='valid')
        layer4 = tf.layers.batch_normalization(layer4, training=istrain)
        layer4 = tf.maximum(alpha * layer4, layer4)
        layer4 = tf.nn.dropout(layer4, keep_prob=0.7)
        layer4 = tf.nn.relu(layer4)

        layer4 = tf.layers.flatten(layer4)
        layer5 = tf.layers.dense(layer4,256)
        layer5 = tf.layers.batch_normalization(layer5, training=istrain)
        layer5 = tf.maximum(alpha * layer5, layer5)
        mean = tf.layers.dense(layer5,20)
        logvar =  tf.layers.dense(layer5,20)
        return mean,logvar


def decoder(x,istrain=True):
    with tf.variable_scope("decoder", reuse=(not istrain)):
        layer5 = tf.layers.dense(x,256)
        layer5 = tf.maximum(alpha * layer5,layer5)

        layer4 = tf.layers.dense(layer5,512)
        layer4 = tf.maximum(alpha * layer4, layer4)
        layer4 = tf.reshape(layer4,shape=(8,1,1,1,512))

        # layer4 = tf.layers.conv3d_transpose(inputs=layer4, filters=1024, kernel_size=(6, 6, 6), strides=2, padding='valid')
        # #layer4 = tf.layers.batch_normalization(layer4, training=istrain)
        # layer4 = tf.maximum(alpha * layer4, layer4)
        # layer4 = tf.nn.dropout(layer4, keep_prob=0.8)

        layer3 = tf.layers.conv3d_transpose(inputs=layer4, filters=256, kernel_size=(5, 5, 5), strides=2, padding='valid')
        layer3 = tf.layers.batch_normalization(layer3, training=istrain)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.7)
        layer3 = tf.nn.relu(layer3)

        layer2 = tf.layers.conv3d_transpose(inputs=layer3, filters=128, kernel_size=(5, 5, 5), strides=2,
                                            padding='valid')
        layer2 = tf.layers.batch_normalization(layer2, training=istrain)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.7)
        layer2 = tf.nn.relu(layer2)

        layer1 = tf.layers.conv3d_transpose(inputs=layer2, filters=64, kernel_size=(5, 5, 5), strides=1,
                                            padding='valid')
        layer1 = tf.layers.batch_normalization(layer1, training=istrain)
        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob=0.7)
        layer1 = tf.nn.relu(layer1)

        out1 = tf.layers.conv3d_transpose(inputs=layer1, filters=32, kernel_size=(5, 5, 5), strides=2, padding='valid')
        out1 = tf.layers.batch_normalization(out1, training=istrain)
        out1 = tf.maximum(alpha * out1, out1)
        out1 = tf.nn.dropout(out1, keep_prob=0.7)
        out1 = tf.nn.relu(out1)

        out = tf.layers.conv3d_transpose(inputs=out1, filters=1, kernel_size=(4, 4, 4), strides=1, padding='valid')
        out = tf.layers.batch_normalization(out, training=istrain)
        out = tf.maximum(alpha * out, out)
        # out = tf.nn.dropout(out, keep_prob=0.8)
        out = tf.nn.tanh(out)
        return out

def sampling(mean,logvar):
    epsilon = tf.random_normal(shape=(1,20),mean=0.,dtype=tf.float32)
    return mean + tf.exp(logvar/2)*epsilon

# def gram_matrix(x):
#     b,w,h,ch = x.get_shape().as_list()
#     features = tf.reshape(x,[b,h*w,ch])
#     #[h*w,ch] matrix ->[ch,h*w] * [h*w,ch] =[ch,ch]
#     gram = tf.matmul(features,features,adjoint_a=True)/tf.constant(ch*w*h,tf.float32)
#     return gram
print(os.getcwd())
x = tf.placeholder(np.float,shape=(8,40,40,40,1))
mean,logvar = encode(x)

z1 = sampling(mean,logvar)
z = decoder(z1)
reconstr_loss = tf.reduce_sum(tf.pow(tf.subtract(z,x),2.0))
latent_loss = -0.5*tf.reduce_sum(1+logvar-tf.square(mean)-tf.exp(logvar),1)
cost = tf.reduce_mean(reconstr_loss+latent_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

path = '../save/TISHALE.txt'

os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU:0'



gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
tfconfig = tf.ConfigProto(gpu_options=gpu_options)
#tfconfig.gpu_options.allow_growth = True
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu,enable=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    input = get_input(path)
    input = np.reshape(input,newshape=(8,40,40,40,1))
    input = input *2-1
    if len(os.listdir("../save/New_Folder")) > 0:
        saver.restore(sess, "../save/New_Folder/save_model.ckpt")
    else:

        for epoch in range(1500):

            _, c = sess.run([optimizer, cost], feed_dict={x: input})

            if epoch % 100== 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
                saver.save(sess, "../save/New_Folder/save_model.ckpt")

        print("ok!")

    f1 = open('../save/output/New_Folder2.txt', 'w+',encoding='UTF-8')
    out = sess.run(z,feed_dict={x:input})
    cubeIndex = 0
    xIndex = 0
    yIndex = 0
    zIndex = 0

    for i in range(80*80*80):
        z = math.floor(i / (80 * 80))
        y = math.floor((i - z * 80 * 80) / 80)
        x = i - y * 80 - z * 80 * 80
        if x in range(0,40) and y in range(0,40) and z in range(0,40):
            cubeIndex = 0
            f1.write(str(1 if out[cubeIndex, x, y, z, 0]>0 else 0)+'\n')
            continue
        if x in range(40,80) and y in range(0,40) and z in range(0,40):
            cubeIndex = 1
            x = x-40
            f1.write(str(1 if out[cubeIndex, x, y, z, 0]>0 else 0) + '\n')
            continue
        if x in range(0,40) and y in range(40,80) and z in range(0,40):
            cubeIndex = 2
            y = y-40
            f1.write(str(1 if out[cubeIndex, x, y, z, 0] > 0 else 0) + '\n')

        if x in range(40,80) and y in range(40,80) and z in range(0,40):
            cubeIndex = 3
            x = x-40
            y = y-40
            f1.write(str(1 if out[cubeIndex, x, y, z, 0] > 0 else 0) + '\n')
            continue

        if x in range(0,40) and y in range(0,40) and z in range(40,80):
            cubeIndex = 4
            z = z-40
            f1.write(str(1 if out[cubeIndex, x, y, z, 0] > 0 else 0) + '\n')
            continue

        if x in range(40,80) and y in range(0,40) and z in range(40,80):
            cubeIndex = 5
            x = x-40
            z = z-40
            f1.write(str(1 if out[cubeIndex, x, y, z, 0] > 0 else 0) + '\n')
            continue

        if x in range(0,40) and y in range(40,80) and z in range(40,80):
            cubeIndex = 6
            z = z-40
            y = y-40
            f1.write(str(1 if out[cubeIndex, x, y, z, 0] > 0 else 0) + '\n')
            continue

        if x in range(40,80) and y in range(40,80) and z in range(40,80):
            cubeIndex = 7
            x = x-40
            y = y-40
            z = z-40
            f1.write(str(1 if out[cubeIndex, x, y, z, 0] > 0 else 0) + '\n')
            continue

    print('执行完毕')

   #  f1 = open('../save/a.txt', 'w+')
   #  for i in range(out.shape[1]):
   #      for row in range(out.shape[2]):
   #          for col in range(out.shape[3]):
   #              if out[0,i,row,col,0]>0:
   #                  f1.write('1\n')
   #              else:
   #                  f1.write('0\n')






# if __name__ == '__main__':
#     y = tf.placeholder(np.float,shape=(1,2))
#     epsilon = np.ones(shape=(1,2))
#     p = decoder(y)
#     input  =  get_input(path)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         res1 = sess.run(p,feed_dict={y:epsilon})
#         print(res1.shape)
