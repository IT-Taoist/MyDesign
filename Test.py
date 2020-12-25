import os
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import MyDesign.CNNModel as model
# ======================================================================

N_CLASSES = 36

img_dir = r"C:/Users/Administrator/PycharmProjects/bishe/MyDesign/test/"
log_dir = r"C:/Users/Administrator/PycharmProjects/bishe/MyDesign/train_00"
lists = ['chan1891','chen1928','cun2071','di2156','fa2308','gai2439',
         'gan2441','gan2448','hong2673','hua2715','huang2742','huang2745',
         'ji2790','ji2811','jiao2944','ling3373','miao3578','ming3587',
         'nian3678','shang4147','shu4264','tian4476','tong4508','wen4637',
         'xiang4783','xin4836','xuan4894','yang4984','ye7639','yin5085',
         'yu5185','yuan5210','zai5255','zan5262','zao5276','zhen5370']

def get_one_image(img_dir):
    imgs = os.listdir(img_dir)
    img_num = len(imgs)
    idn = np.random.randint(0,img_num)
    image = imgs[idn]
    image_dir = img_dir + image
    image = Image.open(image_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([85,85])
    image_arr = np.array(image)
    return image_arr

#將標籤對應漢字進行輸出
def changeToChinese(strLabel):
    if strLabel == 'chan1891':
        strLabel = '闡'
    elif strLabel == 'chen1928':
        strLabel = '臣'
    elif strLabel == 'cun2071':
        strLabel = '寸'
    elif strLabel == 'di2156':
        strLabel = '地'
    elif strLabel == 'fa2308':
        strLabel = '法'
    elif strLabel == 'gai2439':
        strLabel = '蓋'
    elif strLabel == 'gan2441':
        strLabel = '干'
    elif strLabel == 'gan2448':
        strLabel = '感'
    elif strLabel == 'chen1928':
        strLabel = '臣'
    elif strLabel == 'hong2673':
        strLabel = '洪'
    elif strLabel == 'hua2715':
        strLabel = '化'
    elif strLabel == 'huang2742':
        strLabel = '皇'
    elif strLabel == 'huang2745':
        strLabel = '煌'
    elif strLabel == 'ji2790':
        strLabel = '機'
    elif strLabel == 'ji2811':
        strLabel = '極'
    elif strLabel == 'jiao2944':
        strLabel = '教'
    elif strLabel == 'ling3373':
        strLabel = '靈'
    elif strLabel == 'miao3578':
        strLabel = '妙'
    elif strLabel == 'ming3587':
        strLabel = '明'
    elif strLabel == 'nian3678':
        strLabel = '念'
    elif strLabel == 'shang4147':
        strLabel = '上'
    elif strLabel == 'shu4264':
        strLabel = '樞'
    elif strLabel == 'tian4476':
        strLabel = '天'
    elif strLabel == 'tong4508':
        strLabel = '通'
    elif strLabel == 'wen4637':
        strLabel = '聞'
    elif strLabel == 'xiang4783':
        strLabel = '象'
    elif strLabel == 'xin4836':
        strLabel = '心'
    elif strLabel == 'xuan4894':
        strLabel = '玄'
    elif strLabel == 'yang4984':
        strLabel = '陽'
    elif strLabel == 'ye7639':
        strLabel = '燁'
    elif strLabel == 'yin5085':
        strLabel = '陰'
    elif strLabel == 'yu5185':
        strLabel = '於'
    elif strLabel == 'yuan5210':
        strLabel = '元'
    elif strLabel == 'zai5255':
        strLabel = '宰'
    elif strLabel == 'zan5262':
        strLabel = '贊'
    elif strLabel == 'zao5276':
        strLabel = '造'
    elif strLabel == 'zhen5370':
        strLabel = '真'
    return strLabel

def test(image_arr):
    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 85, 85, 3])
        p = model.inference(image, 1, N_CLASSES)
        logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32, shape=[85, 85, 3])
        saver = tf.train.Saver()
        with tf.Session() as sess :
            print("Reading checkpoint...")
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            # 调用saver.restore()函数，加载训练好的网络模型
                print('Loading success')
            prediction = sess.run(logits, feed_dict={x: image_arr})
            max_index = np.argmax(prediction)
            lists[max_index] = changeToChinese(lists[max_index])
            print('預測的標籤序號為：' , max_index )
            print('檢測為漢字：' , lists[max_index])
            print('預測的結果與比較可能性為：%.2f'  %prediction[:,max_index])

if __name__ == '__main__':
    img = get_one_image(img_dir)
    test(img)