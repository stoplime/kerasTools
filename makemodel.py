# Keras model Design
# Using the Keras implementation of the Xception Model

import json
import numpy as np
from enum import Enum, auto

# from sklearn.metrics import fbeta_score
# from custom_metric import FScore2

from keras.models import Model
import parameters as params
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D
from keras.layers.core import Dropout, Lambda
from keras import metrics, losses, optimizers

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras import backend as K


import sys
import os
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PATH, "resnet", "keras-resnet"))

class MakeModel(object):
    # Getter and Setters
    @property
    def model(self):
        return self.model

    @model.setter
    def model(self, value):
        if type(value) is Model:
            self.model = value
    
    class basemodel_enum(Enum):
        xception = auto()
        inceptionv3 = auto()
        resnet50 = auto()
        vgg19 = auto()
        vgg16 = auto()

    # constuctor
    def __init__(self, input_shape=(256, 256, 3), output_classes=17, output_shape=None):
        self.input_tensor = Input(input_shape)
        self.input_shape = input_shape
        self.output_size = output_classes
        self.basemodel = self.basemodel_enum.xception
        self.output_shape = output_shape
    
    def init_base(self, model_type=None):
        base = None
        if (model_type == self.basemodel_enum.xception):
            base = Xception(include_top=False, weights='imagenet', input_tensor=self.input_tensor, classes=self.output_size, pooling='avg')
            model_name = self.basemodel_enum.xception
            pred = base.output
        elif(model_type == self.basemodel_enum.inceptionv3):
            base = InceptionV3(include_top=False, weights='imagenet', input_tensor=self.input_tensor, classes=self.output_size, pooling='avg')
            model_name = self.basemodel_enum.inceptionv3
            pred = base.output
        elif(model_type == self.basemodel_enum.resnet50):
            base = ResNet50(include_top=False, weights='imagenet', input_tensor=self.input_tensor, classes=self.output_size, pooling='avg')
            model_name = self.basemodel_enum.resnet50
            pred = base.output
        elif(model_type == self.basemodel_enum.vgg19):
            base = VGG19(include_top=False, weights='imagenet', input_tensor=self.input_tensor, classes=self.output_size, pooling='avg')
            model_name = self.basemodel_enum.vgg19
            pred = base.output
        elif(model_type == self.basemodel_enum.vgg16):
            base = VGG16(include_top=False, weights='imagenet', input_tensor=self.input_tensor, classes=self.output_size, pooling='avg')
            model_name = self.basemodel_enum.vgg16
            pred = base.output

        return base, pred

    # create self.model from a base
    def create_model(self, model_type=None, load_weights=None, overrideModel=True, poped_layers=3):
        base, pred = self.init_base(model_type)
        if self.output_shape != None:
            pred = base.layers[-poped_layers].output

        pred = Dense(self.output_size, activation='sigmoid', name='predictions')(pred)
        if self.model == None or overrideModel:
            self.model = Model(base.input, pred, name=model_name)
        elif not overrideModel:
            self.model = Model(base.input, pred, name=model_name)(self.model)

        if load_weights != None:
            self.model.load_weights(load_weights)

        if base != None:
            for layer in base.layers:
                layer.trainable = True

        self.compile_model()

        return self.model

    def compile_model(self, loss = None, optimizer = None, metric = None):
        """
        compile_model compiles the self.model instance and specifies the
        loss, optimizer, and metric of the model

        Does not Return anything

        loss has to be from the parameters.losses_enum object
        it is default to binary_crossentropy

        optimizer has to be from the parameters.optimizers_enum object
        it is default to Adam

        metric has to be from the parameters.metrics_enum object
        it is default to binary_accuracy
        """
        if not isinstance(loss, params.losses_enum):
            loss = params.losses_enum.binary_crossentropy
        
        if not isinstance(optimizer, params.optimizers_enum):
            optimizer = params.optimizers_enum.Adam

        if not isinstance(metric, params.metrics_enum):
            metric = params.metrics_enum.binary_accuracy

        self.compile()

    def compile(self):
        self.model.compile(loss="mse", optimizer=optimizers.RMSprop(lr=0.000001, rho=0.9, epsilon=1e-08, decay=0.0), metrics=["mae"])

    def summary(self):
        self.model.summary()

    # def train_model(self, input_train, labels, monitor='val_loss', validation=None, save_path=None):
    #     def load_weights(self, weights_path):
    #     self.model.load_weights(weights_path)

    def add_normalize(self):
        self.model.layers.pop()
        self.model.layers.pop()
        x = self.model.layers[-1].output
        x = Conv2D(2, (1, 1), padding='valid', name='conv2')(x)
        x = GlobalAveragePooling2D()(x)
        x = Lambda(lambda x: K.l2_normalize(x, axis=1), output_shape=(2,))(x)
        self.model = Model(self.model.input, outputs=[x])
        self.compile()
    
    def train_model(self, input_train, labels, validation=None, save_path=None):
        num_epochs = 15
        batch_size = 8

        logging = TensorBoard()
        if save_path != None:
            checkpoint = ModelCheckpoint(str(save_path)+".h5", monitor=monitor, save_weights_only=False, save_best_only=True)
        early_stopping = EarlyStopping(monitor=monitor, min_delta=0.01, patience=5, verbose=1, mode='max')

        if validation==None:
            history = self.model.fit(input_train, labels, validation_split=0.2, batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=[logging, checkpoint, early_stopping])
        else:
            history = self.model.fit(input_train, labels, validation_data=validation, batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=[logging, checkpoint, early_stopping])
        return history.history
        
    def kaggle_metric(self, input_val, labels_val):
        p_val = self.model.predict(input_val, batch_size=128)
        return fbeta_score(labels_val, np.array(p_val) > 0.2, beta=2, average='samples')

    def predict(self, input_val):
        pred = self.model.predict(input_val)
        return pred

    def evaluate(self, input_val, labels_val, batch_size=16):
        self.model.evaluate(input_val, labels_val, batch_size)

