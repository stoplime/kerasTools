# data handler class for data generators

import json
import os
# from keras.preprocessing import image
import numpy as np
import random

# assumes that the labels is a dictionary with "image_name": label

class dataTool:
    def __init__(self, image_folder, label_file, batch_size=32):
        self.image_folder = image_folder
        self.label_file = label_file
        self.batch_size = batch_size
        
        self.image_names = [fn for fn in os.listdir(image_folder)]
        self.image_names.sort()

        with open(label_file) as json_data:
            self.labels_dict = json.load(json_data)
            json_data.close()

    def generator(self, parameter_list):
        pass

    def get_image_per_batch(self, index):
        images = []
        for b in range(self.batch_size):
            labels.append(self.image_names[index + b])
        
    def get_label_per_batch(self, index):
        labels = []
        for b in range(self.batch_size):
            if index + b >= len(self.image_names):
                break
            label_name = self.image_names[index + b]
            label_name = label_name.rsplit('.', 1)
            labels.append(self.labels_dict[label_name[0]])
        return np.asarray(labels)
