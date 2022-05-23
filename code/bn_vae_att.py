import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from PIL import Image
import os
from mpl_toolkits.mplot3d import Axes3D
import math

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

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
    with tf.variable_scope("encoder",reuse=tf.AUTO_REUSE):
        layer1 = tf.layers.conv3d(inputs=input,filters=64,kernel_size=(3,3,3),strides = 1,padding='valid')
        layer1 = tf.layers.batch_normalization(layer1, training=istrain)
        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)

        layer2 = tf.layers.conv3d(inputs=layer1, filters=128, kernel_size=(5, 5, 5), strides=2, padding='valid')
        layer2 = tf.layers.batch_normalization(layer2, training=istrain)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)

        layer3 = tf.layers.conv3d(inputs=layer2, filters=256, kernel_size=(5, 5, 5), strides=2, padding='valid')
        layer3 = tf.layers.batch_normalization(layer3, training=istrain)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)

        layer4 = tf.layers.conv3d(inputs=layer3, filters=512, kernel_size=(6, 6, 6), strides=2, padding='valid')
        layer4 = tf.layers.batch_normalization(layer4, training=istrain)
        layer4 = tf.maximum(alpha * layer4, layer4)
        layer4 = tf.nn.dropout(layer4, keep_prob=0.8)

        layer4 = se_layer(inputs_tensor=layer4, ratio=16, num=1)

        layer4 = tf.layers.flatten(layer4)
        layer5 = tf.layers.dense(layer4,256)
        layer5 = tf.layers.batch_normalization(layer5, training=istrain)
        layer5 = tf.maximum(alpha * layer5, layer5)
        mean = tf.layers.dense(layer5,20)
        logvar =  tf.layers.dense(layer5,20)
        return mean,logvar


def decoder(x,istrain=True):
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
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
        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)

        layer2 = tf.layers.conv3d_transpose(inputs=layer3, filters=128, kernel_size=(5, 5, 5), strides=2,
                                            padding='valid')
        layer2 = tf.layers.batch_normalization(layer2, training=istrain)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)

        layer1 = tf.layers.conv3d_transpose(inputs=layer2, filters=64, kernel_size=(5, 5, 5), strides=1,
                                            padding='valid')
        layer1 = tf.layers.batch_normalization(layer1, training=istrain)
        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)

        out1 = tf.layers.conv3d_transpose(inputs=layer1, filters=32, kernel_size=(5, 5, 5), strides=2, padding='valid')
        out1 = tf.layers.batch_normalization(out1, training=istrain)
        out1 = tf.maximum(alpha * out1, out1)
        out1 = tf.nn.dropout(out1, keep_prob=0.8)

        out = tf.layers.conv3d_transpose(inputs=out1, filters=1, kernel_size=(4, 4, 4), strides=1, padding='valid')
        out = tf.layers.batch_normalization(out, training=istrain)
        out = tf.maximum(alpha * out, out)
        #out = tf.nn.dropout(out, keep_prob=0.8)
        out = tf.nn.tanh(out)
        return out


def se_layer(inputs_tensor=None,ratio=None,num=None,**kwargs):
    """
    SE-NET
    :param inputs_tensor:input_tensor.shape=[batchsize,h,w,channels]
    :param ratio: Number of output channels for excitation intermediate operation
    :param num:
    :return:
    """
    #    channels = inputs_tensor.get_shape()[-1]
    channels = 512
    x = tf.keras.layers.GlobalAveragePooling3D()(inputs_tensor)
    x = tf.keras.layers.Reshape((1, 1, channels))(x)
    x = tf.keras.layers.Conv2D(channels//ratio, (1, 1), strides=1, name="se_conv1_"+str(num), padding="valid")(x)
    x = tf.keras.layers.Activation('relu', name='se_conv1_relu_'+str(num))(x)
    x = tf.keras.layers.Conv2D(channels, (1, 1), strides=1, name="se_conv2_"+str(num), padding="valid")(x)
    x = tf.keras.layers.Activation('sigmoid', name='se_conv2_relu_'+str(num))(x)
    output = tf.keras.layers.multiply([inputs_tensor, x])
    return output



class Scaler(tf.keras.layers.Layer ):
    """特殊的scale层
    """
    def __init__(self,tau=1):
#        super(Scaler, self).__init__(**kwargs)
        super(Scaler, self).__init__()
        self.tau = tau


    def build(self, input_shape):
        super(Scaler, self).build(input_shape)
        self.scale = self.add_weight(
            name='scale', shape=(input_shape[-1],),initializer='zeros'
        )

    def call(self, inputs, mode):
        if mode=='T':
            scale = self.tau + (1 - self.tau) * tf.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * tf.sigmoid(-self.scale)
        return inputs * tf.sqrt(scale)

    # def get_config(self):
    #     config = {'tau': self.tau }
    #     base_config = super(Scaler, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

# def sampling(mean,logvar):
#     epsilon = tf.random_normal(shape=(1,20),mean=0.,dtype=tf.float32)
#     return mean + tf.exp(logvar/2)*epsilon
def sampling(mean,logvar):
    """重参数采样 """
    noise = tf.random_normal(shape=tf.shape(mean))
    return mean + logvar * noise
print(os.getcwd())

x = tf.placeholder(np.float,shape=(8,40,40,40,1))

mean,logvar = encode(x)
#增加部分
scaler=Scaler()
mean = tf.layers.batch_normalization(mean, scale=False, center=False, epsilon=1e-8)
z_mean = scaler(mean, mode='T')
logvar = tf.layers.batch_normalization(logvar, scale=False, center=False, epsilon=1e-8)
z_logvar = scaler(logvar,mode='F')
#print(type(z_logvar))
z1 = sampling(z_mean,z_logvar)
z = decoder(z1)
reconstr_loss = tf.reduce_sum(tf.pow(tf.subtract(z,x),2.0))
latent_loss = -0.5*tf.reduce_sum(1+z_logvar-tf.square(z_mean)-tf.exp(z_logvar),1)
cost = tf.reduce_mean(reconstr_loss+latent_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


path = '../save/ti.txt'

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
    if len(os.listdir("../save/testbeta_model")) > 0:
        saver.restore(sess, "../save/testbeta_model/save_model.ckpt")
    else:

        for epoch in range(2000):

            _, c,b = sess.run([optimizer, cost,latent_loss], feed_dict={x: input})
            y= tf.abs(tf.subtract(2*b/256,1))
            beta = tf.reduce_mean(tf.sqrt(y))
            #if epoch % 100== 0:
            if epoch :
                #print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
                print("Epoch:", '%04d' % (epoch + 1),sess.run(beta))
                #saver.save(sess, "../save/testbeta_model/save_model.ckpt")

        print("ok!")

    # f1 = open('../save/output/bn_att_1000.txt', 'w+',encoding='UTF-8')
    # out = sess.run(z,feed_dict={x:input})
    # cubeIndex = 0
    # xIndex = 0
    # yIndex = 0
    # zIndex = 0
    #
    # for i in range(80*80*80):
    #     z = math.floor(i / (80 * 80))
    #     y = math.floor((i - z * 80 * 80) / 80)
    #     x = i - y * 80 - z * 80 * 80
    #     if x in range(0,40) and y in range(0,40) and z in range(0,40):
    #         cubeIndex = 0
    #         f1.write(str(1 if out[cubeIndex, x, y, z, 0]>0 else 0)+'\n')
    #         continue
    #     if x in range(40,80) and y in range(0,40) and z in range(0,40):
    #         cubeIndex = 1
    #         x = x-40
    #         f1.write(str(1 if out[cubeIndex, x, y, z, 0]>0 else 0) + '\n')
    #         continue
    #     if x in range(0,40) and y in range(40,80) and z in range(0,40):
    #         cubeIndex = 2
    #         y = y-40
    #         f1.write(str(1 if out[cubeIndex, x, y, z, 0] > 0 else 0) + '\n')
    #
    #     if x in range(40,80) and y in range(40,80) and z in range(0,40):
    #         cubeIndex = 3
    #         x = x-40
    #         y = y-40
    #         f1.write(str(1 if out[cubeIndex, x, y, z, 0] > 0 else 0) + '\n')
    #         continue
    #
    #     if x in range(0,40) and y in range(0,40) and z in range(40,80):
    #         cubeIndex = 4
    #         z = z-40
    #         f1.write(str(1 if out[cubeIndex, x, y, z, 0] > 0 else 0) + '\n')
    #         continue
    #
    #     if x in range(40,80) and y in range(0,40) and z in range(40,80):
    #         cubeIndex = 5
    #         x = x-40
    #         z = z-40
    #         f1.write(str(1 if out[cubeIndex, x, y, z, 0] > 0 else 0) + '\n')
    #         continue
    #
    #     if x in range(0,40) and y in range(40,80) and z in range(40,80):
    #         cubeIndex = 6
    #         z = z-40
    #         y = y-40
    #         f1.write(str(1 if out[cubeIndex, x, y, z, 0] > 0 else 0) + '\n')
    #         continue
    #
    #     if x in range(40,80) and y in range(40,80) and z in range(40,80):
    #         cubeIndex = 7
    #         x = x-40
    #         y = y-40
    #         z = z-40
    #         f1.write(str(1 if out[cubeIndex, x, y, z, 0] > 0 else 0) + '\n')
    #         continue

    print('执行完毕')







