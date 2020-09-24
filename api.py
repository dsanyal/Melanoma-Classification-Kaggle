
import numpy as np 
import os
import tensorflow as tf
import tensorflow_addons as tfa
import efficientnet.tfkeras as efn
from flask import Flask,render_template,url_for,request,redirect
from werkzeug.utils import secure_filename

    
# augmentation params
MAX_ROT_ANGLE = 180.0
MAX_SHEAR_LEVEL = 0.1
HSHIFT, VSHIFT = 5., 5. # max. number of pixels to shift(translation) horizontally and vertically
MAX_ROT_ANGLE = np.pi*MAX_ROT_ANGLE/180 # in radians
AUGMENT_FRAC = 0.7 # probability that an image will go through the augmentation pipeline
NUM_CUTOUTS = 10 # how many cutouts to be inserted on each image
CUTOUT_SIZE = 15 # cutout square dimension (in number of pixels)
CUTOUT_FRAC = 0.8 # probability of cutout augmentation being applied to an image (if augmentation is turned on via AUGMENT_FRAC)

#Focal loss params
GAMMA = 2. # focal loss
ALPHA = 0.8  # focal loss

# whether to use test time augmentation
test_time_aug= False
TTA = 5    # number of test time augmentation

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__, template_folder='templates', static_url_path = "/static")
app.secret_key = b'_5#y2L"F1Q8z\n\xec]33'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 *5

def parse_image(image_path):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [192, 192])
    return image

def do_augmentation(image):
    image = tf.image.random_flip_left_right(image)
    # Colour op transformations
    image = tf.image.random_brightness(image, max_delta= 0.05)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, 0.05)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    # crop(central) and resize (zoom)
    image = tf.image.resize_with_crop_or_pad(image, tf.random.uniform(shape=[], minval= tf.cast(0.7*192, tf.int32), maxval=192, dtype=tf.int32), target_width= tf.random.uniform(shape=[], minval=tf.cast(0.7*192, tf.int32), maxval=192, dtype=tf.int32))
    image = tf.image.resize(image, [192,192])
    # shear
    shear_x = MAX_SHEAR_LEVEL* tf.random.uniform(shape=[],minval=-1,maxval=1)
    shear_y = MAX_SHEAR_LEVEL * tf.random.uniform(shape=[],minval=-1,maxval=1)
    image =  tfa.image.transform(image, [1.0, shear_x, 0.0, shear_y, 1.0, 0.0, 0.0, 0.0])
        # rotation
    image = tfa.image.rotate(image, MAX_ROT_ANGLE * tf.random.uniform([], dtype=tf.float32)) # rotation
        # translation
    image = tfa.image.translate(image, [HSHIFT * tf.random.uniform(shape=[],minval=-1, maxval=1), VSHIFT * tf.random.uniform(shape=[],minval=-1, maxval=1)]) # [dx dy] shift/translation
    image = tf.reshape(image, [192,192, 3])
    return image

def data_loader(filename, augment=False, repeat=False):
    dataset = tf.data.Dataset.from_tensor_slices(([filename],))
    dataset = dataset.map(parse_image)
    if repeat:
        dataset = dataset.repeat()
    if augment:
        dataset = dataset.map(do_augmentation)
    dataset = dataset.batch(1)
    return dataset

def predict(image_path):  
    if test_time_aug:
        test_data = data_loader(image_path, augment=True, repeat=True)
        ypred_test = model.predict(test_data, steps = TTA)
        ypred_test = ypred_test[:TTA].reshape((1,TTA), order = 'F') # Fortran like indexing
        ypred_test = ypred_test.mean(axis = 1)[0] 
    else:
        test_data = data_loader(image_path, augment=False, repeat=False)
        #ypred_test = model(np.expand_dims(test_data,axis=0), training=False).numpy()[0][0]
        ypred_test = model.predict(test_data)[0][0]
    return ypred_test

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/error/<int:error_code>', methods=['GET','POST'])
def error(error_code):
    return render_template('error.html',error_code = error_code)


@app.route('/', methods=['GET','POST'])
def upload_image():
    display_image = None
    image_path = None
    prediction_pct = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(url_for('error', error_code=1))
        image = request.files['file']
        display_image = request.form.get('display_image')
        if image.filename == '':
            return redirect(url_for('error', error_code=2))
        if not allowed_file(image.filename):
            return redirect(url_for('error', error_code=3))
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            prediction = predict(image_path)
            prediction_pct = np.round_(prediction*100, 2) # convert to percentage and round of to 2 decimal places
    return render_template('index.html', display_image = display_image, image_path = image_path, prediction=prediction_pct)

if __name__ == '__main__': 
    with open('models/model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights('models/fold3.h5')
    app.run(host = "0.0.0.0",debug=True)


""" 
def data_loader_old(filename, augment=False, repeat=False):
    img_raw = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(img_raw, channels=3)  #tf.io.decode_image(img_raw)

    image = tf.image.resize(image, [192,192])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def build_model(input_shape = (192,192,3), pretrained_model= efn.EfficientNetB2): 
    inp = tf.keras.layers.Input(shape=input_shape)
    base_model = pretrained_model(include_top=False, weights='imagenet', input_shape=input_shape)
    print("Using {} as the base model".format(base_model.name))
    x = base_model(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation = 'relu')(x)
    x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    model = tf.keras.Model(inputs= inp, outputs = x)
    return model

model = build_model()
model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
"""