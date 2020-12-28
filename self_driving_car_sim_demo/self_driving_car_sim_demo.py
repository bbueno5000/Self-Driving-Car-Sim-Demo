"""
TODO: docstring
"""
import argparse
import base64
import cv2
import datetime
import eventlet
import flask
import io
import keras.callbacks
import keras.models
import keras.optimizers
import matplotlib.image as mpimg
import numpy
import os
import PIL
import shutil
import sklearn.model_selection
import socketio

class Drive:
    """
    TODO: docstring
    """
    def ___init__(self):
        """
        TODO: docstring
        """
        self.sio = socketio.Server()
        self.app = flask.Flask(__name__)
        self.model = None
        prev_image_array = None
        self.MAX_SPEED = 25
        self.MIN_SPEED = 10
        self.speed_limit = self.MAX_SPEED

    def __call__(self):
        """
        TODO: docstring
        """
        parser = argparse.ArgumentParser(description='Remote Driving')
        parser.add_argument(
            'model',
            type=str,
            help='Path to model h5 file. Model should be on the same path.')
        parser.add_argument(
            'image_folder',
            type=str,
            nargs='?',
            default='',
            help='Path to image folder. This is where the images from the run will be saved.')
        args = parser.parse_args()
        model = keras.models.load_model(args.model)
        if args.image_folder != '':
            print("Creating image folder at {}".format(args.image_folder))
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                shutil.rmtree(args.image_folder)
                os.makedirs(args.image_folder)
            print("RECORDING THIS RUN ...")
        else:
            print("NOT RECORDING THIS RUN ...")
        app = socketio.Middleware(self.sio, self.app)
        eventlet.wsgi.server(eventlet.listen(('', 4567)), self.app)

    @sio.on('connect')
    def connect(self, sid, environ):
        """
        TODO: docstring
        """
        print("connect ", sid)
        self.send_control(0, 0)

    def send_control(self, steering_angle, throttle):
        """
        TODO: docstring
        """
        self.sio.emit(
            "steer",
            data={'steering_angle': steering_angle.__str__(), 'throttle': throttle.__str__()},
            skip_sid=True)

    @sio.on('telemetry')
    def telemetry(self, sid, data):
        """
        TODO: docstring
        """
        if data:
            steering_angle = float(data["steering_angle"])
            throttle = float(data["throttle"])
            speed = float(data["speed"])
            image = PIL.Image.open(io.BytesIO(base64.b64decode(data["image"])))
            try:
                image = numpy.asarray(image)
                image = Utils.preprocess(image)
                image = numpy.array([image])
                steering_angle = float(self.model.predict(image, batch_size=1))
                if speed > self.speed_limit:
                    self.speed_limit = self.MIN_SPEED
                else:
                    self.speed_limit = self.MAX_SPEED
                throttle = 1.0 - steering_angle**2 - (speed/self.speed_limit)**2
                print('{} {} {}'.format(steering_angle, throttle, speed))
                self.send_control(steering_angle, throttle)
            except Exception as e:
                print(e)
            if args.image_folder != '':
                timestamp = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                image_filename = os.path.join(args.image_folder, timestamp)
                image.save('{}.jpg'.format(image_filename))
        else:
            self.sio.emit('manual', data={}, skip_sid=True)

class Model:
    """
    TODO: docstring
    """
    def __init__(self):
        """
        TODO: docstring
        """
        numpy.random.seed(0)

    def __call__(self):
        """
        Load train/validation data set and train the model
        """
        parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
        parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='data')
        parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
        parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
        parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=10)
        parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=20000)
        parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=40)
        parser.add_argument('-o', help='save best models only', dest='save_best_only', type=self.s2b, default='true')
        parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
        args = parser.parse_args()
        print('-' * 30)
        print('Parameters')
        print('-' * 30)
        for key, value in vars(args).items():
            print('{:<20} := {}'.format(key, value))
        print('-' * 30)
        data = self.load_data(args)
        model = self.build_model(args)
        self.train_model(model, args, *data)

    def build_model(self, args):
        """
        NVIDIA model used.
        Image normalization to avoid saturation and make gradients work better.
        The convolution layers are meant to handle feature engineering
        The fully connected layer for predicting the steering angle.
        Dropout avoids overfitting.
        ELU(Exponential linear unit) function takes care of the vanishing gradient problem. 
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Lambda(lambda x: x/127.5-1.0, input_shape=Utils().INPUT_SHAPE))
        model.add(keras.layers.Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(keras.layers.Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(keras.layers.Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(keras.layers.Conv2D(64, 3, 3, activation='elu'))
        model.add(keras.layers.Conv2D(64, 3, 3, activation='elu'))
        model.add(keras.layers.Dropout(args.keep_prob))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation='elu'))
        model.add(keras.layers.Dense(50, activation='elu'))
        model.add(keras.layers.Dense(10, activation='elu'))
        model.add(keras.layers.Dense(1))
        model.summary()
        return model

    def load_data(self, args):
        """
        Load training data and split it into training and validation set.
        """
        data_df = pandas.read_csv(
            os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'),
            names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
        X = data_df[['center', 'left', 'right']].values
        y = data_df['steering'].values
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
            X, y, test_size=args.test_size, random_state=0)
        return X_train, X_valid, y_train, y_valid

    def s2b(self, s):
        """
        Converts a string to boolean value.
        """
        s = s.lower()
        return s == 'true' or s == 'yes' or s == 'y' or s == '1'

    def train_model(self, model, args, X_train, X_valid, y_train, y_valid):
        """
        Train the model.
        """
        checkpoint = keras.callbacks.ModelCheckpoint(
            'model-{epoch:03d}.h5', monitor='val_loss', verbose=0,
            save_best_only=args.save_best_only, mode='auto')
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=args.learning_rate))
        model.fit_generator(
            Utils.batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
            args.samples_per_epoch, args.nb_epoch, max_q_size=1,
            validation_data=Utils.batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
            nb_val_samples=len(X_valid), callbacks=[checkpoint], verbose=1)

class Utils:
    """
    TODO: docstring
    """
    def __init__(self):
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS = 66, 200, 3
        self.INPUT_SHAPE = (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS)

    def augument(self, data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
        """
        Generate an augumented image and adjust steering angle
        (The steering angle is associated with the center image).
        """
        image, steering_angle = self.choose_image(data_dir, center, left, right, steering_angle)
        image, steering_angle = self.random_flip(image, steering_angle)
        image, steering_angle = self.random_translate(image, steering_angle, range_x, range_y)
        image = self.random_shadow(image)
        image = self.random_brightness(image)
        return image, steering_angle

    def batch_generator(self, data_dir, image_paths, steering_angles, batch_size, is_training):
        """
        Generate training image give image paths and associated steering angles.
        """
        images = numpy.empty([batch_size,self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS])
        steers = numpy.empty(batch_size)
        while True:
            i = 0
            for index in numpy.random.permutation(image_paths.shape[0]):
                center, left, right = image_paths[index]
                steering_angle = steering_angles[index]
                if is_training and numpy.random.rand() < 0.6:
                    image, steering_angle = self.augument(data_dir, center, left, right, steering_angle)
                else:
                    image = self.load_image(data_dir, center) 
                images[i] = self.preprocess(image)
                steers[i] = steering_angle
                i += 1
                if i == batch_size:
                    break
            yield images, steers

    def choose_image(self, data_dir, center, left, right, steering_angle):
        """
        Randomly choose an image from the center, left or right, and adjust the steering angle.
        """
        choice = numpy.random.choice(3)
        if choice == 0:
            return self.load_image(data_dir, left), steering_angle + 0.2
        elif choice == 1:
            return self.load_image(data_dir, right), steering_angle - 0.2
        return self.load_image(data_dir, center), steering_angle

    def crop(self, image):
        """
        Crop the image (removing the sky at the top and the car front at the bottom).
        """
        return image[60:-25, :, :] # remove the sky and the car front

    def load_image(self, data_dir, image_file):
        """
        Load RGB images from a file.
        """
        return mpimg.imread(os.path.join(data_dir, image_file.strip()))

    def preprocess(self, image):
        """
        Combine all preprocess functions into one.
        """
        image = self.crop(image)
        image = self.resize(image)
        image = self.rgb2yuv(image)
        return image

    def random_brightness(self, image):
        """
        Randomly adjust brightness of the image.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (numpy.random.rand() - 0.5)
        hsv[:,:,2] =  hsv[:,:,2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def random_flip(self, image, steering_angle):
        """
        Randomly flipt the image left <-> right, and adjust the steering angle.
        """
        if numpy.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle
        return image, steering_angle

    def random_shadow(self, image):
        """
        Generates and adds random shadow
        """
        # (x1, y1) and (x2, y2) forms a line
        # xm, ym gives all the locations of the image
        x1, y1 = self.IMAGE_WIDTH * numpy.random.rand(), 0
        x2, y2 = self.IMAGE_WIDTH * numpy.random.rand(), self.IMAGE_HEIGHT
        xm, ym = numpy.mgrid[0:self.IMAGE_HEIGHT, 0:self.IMAGE_WIDTH]
        # mathematically speaking, we want to set 1 below the line and zero otherwise
        # Our coordinate is up side down.  So, the above the line: 
        # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
        # as x2 == x1 causes zero-division problem, we'll write it in the below form:
        # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
        mask = numpy.zeros_like(image[:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
        # choose which side should have shadow and adjust saturation
        cond = mask == numpy.random.randint(2)
        s_ratio = numpy.random.uniform(low=0.2, high=0.5)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    def random_translate(self, image, steering_angle, range_x, range_y):
        """
        Randomly shift the image virtially and horizontally (translation).
        """
        trans_x = range_x * (numpy.random.rand() - 0.5)
        trans_y = range_y * (numpy.random.rand() - 0.5)
        steering_angle += trans_x * 0.002
        trans_m = numpy.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image, steering_angle

    def rgb2yuv(self, image):
        """
        Convert the image from RGB to YUV (This is what the NVIDIA model does).
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    def resize(self, image):
        """
        Resize the image to the input shape used by the network model.
        """
        return cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), cv2.INTER_AREA)
