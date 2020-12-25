import os
import math
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

# ============================================================================
# -----------------生成图片路径和标签的List------------------------------------
train_dir = r"C:\Users\Administrator\PycharmProjects\bishe/MyDesign/train_00/"
#创建对应的列表，用于存放图片和labels
chan1891=[]
label_chan1891=[]
chen1928=[]
label_chen1928=[]
cun2071=[]
label_cun2071=[]
di2156=[]
label_di2156=[]
fa2308=[]
label_fa2308=[]
gai2439=[]
label_gai2439=[]
gan2441=[]
label_gan2441=[]
gan2448=[]
label_gan2448=[]
hong2673=[]
label_hong2673=[]
hua2715=[]
label_hua2715=[]
huang2742=[]
label_huang2742=[]
huang2745=[]
label_huang2745=[]
ji2790=[]
label_ji2790=[]
ji2811=[]
label_ji2811=[]
jiao2944=[]
label_jiao2944=[]
ling3373=[]
label_ling3373=[]
miao3578=[]
label_miao3578=[]
ming3587=[]
label_ming3587=[]
nian3678=[]
label_nian3678=[]
shang4147=[]
label_shang4147=[]
shu4264=[]
label_shu4264=[]
tian4476=[]
label_tian4476=[]
tong4508=[]
label_tong4508=[]
wen4637=[]
label_wen4637=[]
xiang4783=[]
label_xiang4783=[]
xin4836=[]
label_xin4836=[]
xuan4894=[]
label_xuan4894=[]
yang4984=[]
label_yang4984=[]
ye7639=[]
label_ye7639=[]
yin5085=[]
label_yin5085=[]
yu5185=[]
label_yu5185=[]
yuan5210=[]
label_yuan5210=[]
zai5255=[]
label_zai5255=[]
zan5262=[]
label_zan5262=[]
zao5276=[]
label_zao5276=[]
zhen5370=[]
label_zhen5370=[]

# ============================================================================
# step1：获取r"C:\Users\Administrator\Desktop\ProjectDemo\MyDesign\train_00"下所有的图片路径名，存放到
# 对应的列表中，同时贴上标签，存放到label列表中。
def get_files(file_dir, ratio):
    for file in os.listdir(file_dir + '/chan1891'):
        chan1891.append(file_dir + '/chan1891' + '/' + file)
        label_chan1891.append(0)
    for file in os.listdir(file_dir + '/chen1928'):
        chen1928.append(file_dir + '/chen1928' + '/' + file)
        label_chen1928.append(1)
    for file in os.listdir(file_dir + '/cun2071'):
        cun2071.append(file_dir + '/cun2071' + '/' + file)
        label_cun2071.append(2)
    for file in os.listdir(file_dir + '/di2156'):
        di2156.append(file_dir + '/di2156' + '/' + file)
        label_di2156.append(3)
    for file in os.listdir(file_dir + '/fa2308'):
        fa2308.append(file_dir + '/fa2308' + '/' + file)
        label_fa2308.append(4)
    for file in os.listdir(file_dir + '/gai2439'):
        gai2439.append(file_dir + '/gai2439' + '/' + file)
        label_gai2439.append(5)
    for file in os.listdir(file_dir + '/gan2441'):
        gan2441.append(file_dir + '/gan2441' + '/' + file)
        label_gan2441.append(6)
    for file in os.listdir(file_dir + '/gan2448'):
        gan2448.append(file_dir + '/gan2448' + '/' + file)
        label_gan2448.append(7)
    for file in os.listdir(file_dir + '/hong2673'):
        hong2673.append(file_dir + '/hong2673' + '/' + file)
        label_hong2673.append(8)
    for file in os.listdir(file_dir + '/hua2715'):
        hua2715.append(file_dir + '/hua2715' + '/' + file)
        label_hua2715.append(9)
    for file in os.listdir(file_dir + '/huang2742'):
        huang2742.append(file_dir + '/huang2742' + '/' + file)
        label_huang2742.append(10)
    for file in os.listdir(file_dir + '/huang2745'):
        huang2745.append(file_dir + '/huang2745' + '/' + file)
        label_huang2745.append(11)
    for file in os.listdir(file_dir + '/ji2811'):
        ji2811.append(file_dir + '/ji2811' + '/' + file)
        label_ji2811.append(12)
    for file in os.listdir(file_dir + '/ji2790'):
        ji2790.append(file_dir + '/ji2790' + '/' + file)
        label_ji2790.append(13)
    for file in os.listdir(file_dir + '/jiao2944'):
        jiao2944.append(file_dir + '/jiao2944' + '/' + file)
        label_jiao2944.append(14)
    for file in os.listdir(file_dir + '/ling3373'):
        ling3373.append(file_dir + '/ling3373' + '/' + file)
        label_ling3373.append(15)
    for file in os.listdir(file_dir + '/miao3578'):
        miao3578.append(file_dir + '/miao3578' + '/' + file)
        label_miao3578.append(16)
    for file in os.listdir(file_dir + '/ming3587'):
        ming3587.append(file_dir + '/ming3587' + '/' + file)
        label_ming3587.append(17)
    for file in os.listdir(file_dir + '/nian3678'):
        nian3678.append(file_dir + '/nian3678' + '/' + file)
        label_nian3678.append(18)
    for file in os.listdir(file_dir + '/shang4147'):
        shang4147.append(file_dir + '/shang4147' + '/' + file)
        label_shang4147.append(19)
    for file in os.listdir(file_dir + '/shu4264'):
        shu4264.append(file_dir + '/shu4264' + '/' + file)
        label_shu4264.append(20)
    for file in os.listdir(file_dir + '/tian4476'):
        tian4476.append(file_dir + 'tian4476' + '/' + file)
        label_tian4476.append(21)
    for file in os.listdir(file_dir + '/tong4508'):
        tong4508.append(file_dir + '/tong4508' + '/' + file)
        label_tong4508.append(22)
    for file in os.listdir(file_dir + '/wen4637'):
        wen4637.append(file_dir + '/wen4637' + '/' + file)
        label_wen4637.append(23)
    for file in os.listdir(file_dir + '/xiang4783'):
        xiang4783.append(file_dir + '/xiang4783' + '/' + file)
        label_xiang4783.append(24)
    for file in os.listdir(file_dir + '/xin4836'):
        xin4836.append(file_dir + '/xin4836' + '/' + file)
        label_xin4836.append(25)
    for file in os.listdir(file_dir + '/xuan4894'):
        xuan4894.append(file_dir + '/xuan4894' + '/' + file)
        label_xuan4894.append(26)
    for file in os.listdir(file_dir + '/yang4984'):
        yang4984.append(file_dir + '/yang4984' + '/' + file)
        label_yang4984.append(27)
    for file in os.listdir(file_dir + '/ye7639'):
        ye7639.append(file_dir + '/ye7639' + '/' + file)
        label_ye7639.append(28)
    for file in os.listdir(file_dir + '/yin5085'):
        yin5085.append(file_dir + '/yin5085' + '/' + file)
        label_yin5085.append(29)
    for file in os.listdir(file_dir + '/yu5185'):
        yu5185.append(file_dir + '/yu5185' + '/' + file)
        label_yu5185.append(30)
    for file in os.listdir(file_dir + '/yuan5210'):
        yuan5210.append(file_dir + '/yuan5210' + '/' + file)
        label_yuan5210.append(31)
    for file in os.listdir(file_dir + '/zai5255'):
        zai5255.append(file_dir + '/zai5255' + '/' + file)
        label_zai5255.append(32)
    for file in os.listdir(file_dir + '/zan5262'):
        zan5262.append(file_dir + '/zan5262' + '/' + file)
        label_zan5262.append(33)
    for file in os.listdir(file_dir + '/zao5276'):
        zao5276.append(file_dir + '/zao5276' + '/' + file)
        label_zao5276.append(34)
    for file in os.listdir(file_dir + '/zhen5370'):
        zhen5370.append(file_dir + '/zhen5370' + '/' + file)
        label_zhen5370.append(35)

    #print("There are "+len(zhen5370)+"zhen5370\nThere are "+len(zao5276)+ "zao5276\nThere are "+len(zan5262)+" zan5262\n")
    # step2：对生成的图片路径和标签List做打乱处理,合起来组成一个list（img和lab）
    image_list = np.hstack((chan1891,chen1928,cun2071,di2156,fa2308,gai2439,
                            gan2441,gan2448,hong2673,hua2715,huang2742,huang2745,
                            ji2811,ji2790,jiao2944,ling3373,miao3578,ming3587,
                            nian3678,shang4147,shu4264,tian4476,tong4508,wen4637,
                            xiang4783,xin4836,xuan4894,yang4984,ye7639,yin5085,
                            yu5185,yuan5210,zai5255,zan5262,zao5276,zhen5370))
    label_list = np.hstack((label_chan1891, label_chen1928, label_cun2071, label_di2156,label_fa2308,label_gai2439,
                            label_gan2441,label_gan2448,label_hong2673,label_hua2715,label_huang2742,label_huang2745,
                            label_ji2790,label_ji2811,label_jiao2944,label_ling3373,label_miao3578,label_ming3587,
                            label_nian3678,label_shang4147,label_shu4264,label_tian4476,label_tong4508,label_wen4637,
                            label_xiang4783,label_xin4836,label_xuan4894,label_yang4984,label_ye7639,label_yin5085,
                            label_yu5185,label_yuan5210,label_zai5255,label_zan5262,label_zao5276,label_zhen5370))
    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()     #转秩
    np.random.shuffle(temp)

    # 从打乱的temp中再取出list（img和lab）
    # image_list = list(temp[:, 0])
    # label_list = list(temp[:, 1])
    # label_list = [int(i) for i in label_list]
    # return image_list,label_list

    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    # 将所得List分为两部分，一部分用来训练tra，一部分用来测试val
    # ratio是测试集的比例
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))  # 测试样本数
    n_train = n_sample - n_val  # 训练样本数
    print(n_sample,n_val,n_train)

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels


# ---------------------------------------------------------------------------
# --------------------生成Batch----------------------------------------------

# step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab
# 是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像
#   image_W, image_H, ：设置好固定的图像高度和宽度
#   设置batch_size：每个batch要放多少张图片
#   capacity：一个队列最大多少
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 转换类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # read img from a queue

    # step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    # step4：生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

    # ========================================================================

def PreWork():

    # 对预处理的数据进行可视化，查看预处理的效果
    IMG_W = 85
    IMG_H = 85
    BATCH_SIZE = 16
    CAPACITY = 64
    image_list, label_list, val_images, val_labels = get_files(train_dir,0.5)
    # image_list, label_list = get_files(train_dir,0.5)
    image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    print(label_batch.shape)
    lists = ('chan1891','chen1928','cun2071','di2156','fa2308','gai2439',
         'gan2441','gan2448','hong2673','hua2715','huang2742','huang2745',
         'ji2790','ji2811','jiao2944','ling3373','miao3578','ming3587',
         'nian3678','shang4147','shu4264','tian4476','tong4508','wen4637',
         'xiang4783','xin4836','xuan4894','yang4984','ye7639','yin5085',
         'yu5185','yuan5210','zai5255','zan5262','zao5276','zhen5370')
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()  # 创建一个线程协调器，用来管理之后在Session中启动的所有线程
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i < 1:
    # 提取出两个batch的图片并可视化。
                img, label = sess.run([image_batch, label_batch])  # 在会话中取出img和label
    # img = tf.cast(img, tf.uint8)

                for j in np.arange(BATCH_SIZE):
    # np.arange()函数返回一个有终点和起点的固定步长的排列
                    print('label: %d' % label[j])
                    plt.imshow(img[j, :, :, :])
                    title = lists[int(label[j])]
                    plt.title(title)
                    plt.show()
                i += 1
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)
if __name__ == '__main__':
    PreWork()
