import tifffile
import numpy as np

def text_save_sgems(filename, data, d, h, w):
    # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    file.write(str(d) + ' ' + str(h) + ' ' + str(w) + '\n')
    file.write('1' + '\n')
    file.write('data' + '\n')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")

# sample = 19539
# # while 20<sample<31:
# while sample==19539:
    #img_sys_path = '/home/lab209-3/Alldata/morooi/SinGAN/PorousMediaGan-master/code/pytorch/postprocess'
#img_sys_path = '../postprocess'
# im_in = tifffile.imread('%s/yy_fake_TI_postprocessed_example%d.tiff' % (img_sys_path,sample))
#im_in = tifffile.imread('E:\\Experiment_Data\\SR\\test.tif')

im_in = tifffile.imread('E:\pycharmprojects\PyTorch-SRGAN\myshale.tif')
real_255 = np.ones((im_in.shape[0], im_in.shape[1], im_in.shape[2])) * 255
data = ((real_255 - im_in) / 255).astype(np.int32).reshape(80 * 80 * 80)
#data = im_in.astype(np.int32).reshape(80 * 80 * 80)
# txt_path = '%s/yy_fake_TI_postprocessed_example%d.txt' % (img_sys_path, sample)
#txt_path='E:\\Experiment_Data\\SR\\test.txt'
txt_path='E:\pycharmprojects\PyTorch-SRGAN\myshale.txt'
text_save_sgems(txt_path, data, 80, 80, 80)

#sample += 1

