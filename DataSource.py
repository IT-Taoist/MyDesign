import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image  #注意Image,后面会用到

# 原始图片的存储位置
orig_picture = r"C:\Users\Administrator\PycharmProjects\bishe\database\train\\"

# 生成图片的存储位置
gen_picture = r"C:\Users\Administrator\PycharmProjects\bishe\MyDesign\train_00/"

# 需要的识别类型
classes = {'chan1891','chen1928','cun2071','di2156','fa2308','gai2439',
         'gan2441','gan2448','hong2673','hua2715','huang2742','huang2745',
         'ji2790','ji2811','jiao2944','ling3373','miao3578','ming3587',
         'nian3678','shang4147','shu4264','tian4476','tong4508','wen4637',
         'xiang4783','xin4836','xuan4894','yang4984','ye7639','yin5085',
         'yu5185','yuan5210','zai5255','zan5262','zao5276','zhen5370'}

# 样本总数
num_samples = 396


# 制作TFRecords数据
def create_record():
    writer = tf.python_io.TFRecordWriter("train.tfrecords")#创建一个tfrecords文件
    for index, name in enumerate(classes):                 #enumerate函数用于字典的枚举，index为位置信息，name用于获取对应label
        class_path = orig_picture + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((85, 85))  # 设置需要转换的图片大小
            img_raw = img.tobytes()     # 将图片转化为原生bytes
            #print(index, img_raw)
            example = tf.train.Example( #将图片bytes转换成2进制并与对应label写入文件，提升I/O效率
                features=tf.train.Features(feature={    #设置feature
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())
    writer.close()


# =======================================================================================
def read_and_decode(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # 从文件队列里创建一个reader
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        })
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [85, 85, 3])
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)
    return img, label


# =======================================================================================
if __name__ == '__main__':
    create_record()
    batch = read_and_decode('train.tfrecords')
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #将传入参数的操作进行分组
    with tf.Session() as sess:  # 开始一个会话
        sess.run(init_op)       #启动数据出列并执行计算
        coord = tf.train.Coordinator()#线程管理器：tensor中用于管理Session多线程，与runner往往连用
        threads = tf.train.start_queue_runners(coord=coord)#入队线程，返回线程的列表

        for i in range(num_samples):
            example, lab = sess.run(batch)  # 在会话中取出image和label
            img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
            img.save(gen_picture + '/' + str(i) + 'samples' + str(lab) + '.JPG')  # 存下图片;注意cwd后边加上‘/’
            print(example, lab)
        coord.request_stop()                #终止所有线程
        coord.join(threads)                 #将线程加入主线程
        sess.close()

        # ========================================================================================