# Keras model Design
# Using the Keras implementation of the Xception Model

import json
import numpy as np

from sklearn.metrics import fbeta_score

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.core import Dropout
import parameters as params

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16

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

    # constuctor
    def __init__(self, input_shape=(256, 256, 3), output_classes=17):
        self.input_tensor = Input(input_shape)
        self.input_shape = input_shape
        self.output_size = output_classes

    # create self.model
    def create_model(self, model_type='xception', load_weights=None):
        base = None
        if(model_type == 'inceptionv3' or model_type == 1):
            base = InceptionV3(include_top=False, weights='imagenet', input_tensor=self.input_tensor, classes=self.output_size, pooling='avg')
            model_name = 'inceptionv3'
            pred = base.output
        elif(model_type == 'resnet50' or model_type == 2):
            base = ResNet50(include_top=False, weights='imagenet', input_tensor=self.input_tensor, classes=self.output_size, pooling='avg')
            model_name = 'resnet50'
            pred = base.output
        elif(model_type == 'vgg19' or model_type == 3):
            base = VGG19(include_top=False, weights='imagenet', input_tensor=self.input_tensor, classes=self.output_size, pooling='avg')
            model_name = 'vgg19'
            pred = base.output
        elif(model_type == 'vgg16' or model_type == 4):
            base = VGG16(include_top=False, weights='imagenet', input_tensor=self.input_tensor, classes=self.output_size, pooling='avg')
            model_name = 'vgg16'
            pred = base.output
        else:
            base = Xception(include_top=False, weights='imagenet', input_tensor=self.input_tensor, classes=self.output_size, pooling='avg')
            model_name = 'xception'
            pred = base.output
        pred = Dense(self.output_size, activation='sigmoid', name='predictions')(pred)
        self.model = Model(base.input, pred, name=model_name)

        if load_weights != None:
            self.model.load_weights(load_weights)

        if base != None:
            for layer in base.layers:
                layer.trainable = True

        return self.model

    def compile_model(self, loss = None, optimizer = None, metric = None):
        """
        compile_model compiles the self.model instance and specifies the
        loss, optimizer, and metric of the model

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

        self.model.compile(loss=params.loss[loss],
                           optimizer=params.optimizer[optimizer],
                           metrics=[params.metric[metric]])

    def train_model(self, input_train, labels, validation=None, save_path=None):
        num_epochs = 15
        batch_size = 8

        logging = TensorBoard()
        if save_path != None:
            checkpoint = ModelCheckpoint(str(save_path)+".h5", monitor='val_FScore2', save_weights_only=True, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_FScore2', min_delta=0.01, patience=5, verbose=1, mode='max')

        if validation==None:
            history = self.model.fit(input_train, labels, validation_split=0.2, batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=[logging, checkpoint, early_stopping])
        else:
            history = self.model.fit(input_train, labels, validation_data=validation, batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=[logging, checkpoint, early_stopping])
        return history.history
        
    def kaggle_metric(self, input_val, labels_val):
        p_val = self.model.predict(input_val, batch_size=128)
        return fbeta_score(labels_val, np.array(p_val) > 0.2, beta=2, average='samples')
