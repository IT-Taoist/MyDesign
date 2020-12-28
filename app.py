import tkinter.messagebox
from selenium import webdriver
from selenium.webdriver.common.alert import Alert
from flask import render_template, Flask, request
import MyDesign.Test as meThod
import MyDesign.CharacterDivide as DV
from opencc import OpenCC

app = Flask(__name__)

ctrl = 0
save_dir = r'C:/Users/Administrator/Desktop/bishe/MyDesign/read/'
base_dir = r'C:/Users/Administrator/Desktop/bishe/MyDesign/test/'

@app.route('/',methods=['GET','POST'])
def init():
    if request.method == 'GET':
        return render_template('ImgCV.html',info = '',img='',simple = '')
    if request.method == 'POST' :
        #获取标单提交的图片
        f = request.form['img_file']
        if f != '':#判断是否获取到值
            row,column = DV.divide(base_dir + f)#执行分割图片的程序
            info = ''
            for i in range(10,row+1):
                for j in range(0,column+1):
                    img = meThod.get_one_image(save_dir + str(i) + '_' + str(j)+ '.jpg')
                    try:
                        result = meThod.test(img)
                        info = info + result
                    except Exception as e:
                        info = info + '\t'
                        print(e)
                info = info + '\n'
                simple = OpenCC('t2s').convert(info)#实现繁体字转换  t2s  繁转简  s2t   简转繁
            return render_template('ImgCV.html',info=info,img = f,ctrl = 1,simple = simple)
        else :
            # tkinter.Tk().withdraw()
            tkinter.Tk().wm_attributes('-topmost', 1)
            tkinter.messagebox.showwarning('錯誤信息','請上傳圖片！')
            return render_template('ImgCV.html', info='   ', img='')


if __name__ == '__main__':
    app.run()
