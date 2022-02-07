import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Conv2D, Lambda
from tensorflow.keras.models import Model
from keras.preprocessing.image import save_img
from PIL import Image
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
import shutil
from werkzeug.utils import secure_filename

def load_image(path):
    return np.array(Image.open(path))

def resolve_single(model, lr):
    return resolve(model, np.expand_dims(lr, axis=0))[0]

def resolve(model, lr_batch):
    lr_batch = lr_batch.astype("float32")
    sr_batch = model(lr_batch)
    sr_batch = np.clip(sr_batch, 0, 255)
    sr_batch = np.around(sr_batch)
    sr_batch = sr_batch.astype("uint8")
    return sr_batch

def edsr(scale= 4, num_filters=64, num_res_blocks=16, res_block_scaling=None):
    """Creates an EDSR model."""
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
    """Creates an EDSR residual block."""
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        """Sub-pixel convolution."""
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

def normalize(x):
    return (x - DIV2K_RGB_MEAN) / 127.5


def denormalize(x):
    return x * 127.5 + DIV2K_RGB_MEAN

genratorModel = edsr()
genratorModel.load_weights('weights/edsr_X4_SRGAN-24850.h5')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route("/")
def upload() :
    return render_template("index.html", pagetitle="Homepage")

@app.route("/", methods=['GET', 'POST'])
def uploading() :
    if request.method == 'POST' :
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        lr_file = request.files['file']
        if lr_file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)

        if lr_file and allowed_file(lr_file.filename):
            lr_filename = secure_filename(lr_file.filename)
            extension = os.path.splitext(lr_filename)[1]
            lr_filename = "TheLr" + extension
            lr_file.save(UPLOAD_FOLDER + lr_filename)
            lr_image = load_image(UPLOAD_FOLDER+lr_filename)
            shutil.move(UPLOAD_FOLDER + lr_filename, "/app/static/uploads/"+lr_filename)
            hr_image = resolve_single(genratorModel, lr_image)
            hr_filename = "TheHr" + extension
            save_img(UPLOAD_FOLDER+hr_filename, hr_image)
            shutil.move(UPLOAD_FOLDER + hr_filename, "/app/static/uploads/"+hr_filename)

            return render_template("uploading.html",
                                    pagetitle="result",
                                    lr_image=lr_filename,
                                    hr_image=hr_filename)
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for("..", filename = "tmp/" + filename), code=301)

if __name__ == "__main__" :
   app.run()