# data handler class for data generators

import json
import os
from keras.preprocessing import image
import numpy as np
import random

class dataTool:
    def __init__(self, image_folder, label_parser, batch_size=32):
        self.image_folder = image_folder
        # function to parse the lables into label_name and label_data
        self.label_parser = label_parser
        self.batch_size = batch_size
        
        self.image_names = [fn for fn in os.listdir(image_folder)]

    @staticmethod
    def label_parser_json(parameter_list):
        pass

    def generator(self, parameter_list):
        pass

    def get_image_per_batch(self, parameter_list):
        pass
        
    def get_label_per_batch(self, parameter_list):
        pass