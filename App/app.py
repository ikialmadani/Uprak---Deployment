from flask import Flask, render_template, request
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np


app = Flask(__name__)

#ke halaman utama
@app.route('/')
def index():
  return render_template('index.html')

#ke halaman home
@app.route('/home')
def home():
  return render_template('index.html')

#ke halaman about
@app.route('/about')
def about():
  return render_template('about.html')

#ke halaman team
@app.route('/team')
def team():
  return render_template('team.html')

#halaman prediksi
@app.route('/prediksi', methods=['GET'])
def prediction():
  return render_template('prediksi.html')

#halaman prediksi(prediksi jenis bunga)
@app.route('/prediksi', methods=['POST'])
def prediksi():
  imagefile = request.files['flowerimage']
  image_path = "static/upload/" + imagefile.filename
  imagefile.save(image_path)

  IMAGE_SIZE = (200,200)
  BATCH_SIZE = 32

  model = keras.models.load_model('model.h5')
  img = image.load_img(image_path, target_size=IMAGE_SIZE)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
 
  images = np.vstack([x])
  classes = model.predict(images, batch_size=BATCH_SIZE)
  classes = np.argmax(classes)
  if classes==0:
    result = "daisy"
    print('Bunga Daisy')
  elif classes==1:
    result = "dandelion"
    print('Bunga Dandelion')
  elif classes==2:
    result = "rose"
    print('Bunga Rose')
  elif classes==3:
    result = "sunflower"
    print('Bunga Sunflower')
  else:
    result = "tulip"
    print('Bunga Tulip')

  return render_template('prediksi.html', prediction=result, img_file=image_path)

if __name__ == "__main__":
  app.run(debug=True)
