from flask import render_template, Flask, request

app = Flask(__name__)

@app.route('/')
def init():

    return render_template('index.html')

@app.route('/loadImg',methods=['GET','POST'])
def loadImg():
    img_file = request.form['imgFile']

    return render_template('index.html',img_file=img_file)

if __name__ == '__main__':
    app.run()