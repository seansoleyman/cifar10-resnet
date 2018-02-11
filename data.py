import numpy as np
import pickle
import os
import tensorflow as tf

import params

class Cifar10():
    """Load the CIFAR-10 data into NumPy arrays upon instantiation. """
    def __init__(self):
        self._loadDatasetFromFiles()
        
    def _loadDatasetFromFiles(self):
        file_dir = os.path.join(params.DATA_DIR, 'cifar-10-batches-py')
        train_data_list = []
        train_labels_list = []
        for i in range(1,6):
            train_filename = 'data_batch_{}'.format(i)
            train_dir = os.path.join(file_dir, train_filename)
            train_dict = _unpickle(train_dir)
            train_data_list.append(train_dict[b'data'])
            train_labels_list.append(train_dict[b'labels'])
        self.train_images = np.concatenate(train_data_list, 0)
        self.train_images = self.train_images.reshape([50000,3,32,32])
        self.train_images = self.train_images.transpose([0,2,3,1])
        self.train_labels = np.concatenate(train_labels_list, 0)
        
        test_dict = _unpickle(os.path.join(file_dir, 'test_batch'))
        self.test_images = np.array(test_dict[b'data'])
        self.test_images = self.test_images.reshape([10000,3,32,32])
        self.test_images = self.test_images.transpose([0,2,3,1])
        self.test_labels = np.array(test_dict[b'labels'])

def _unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def randomization_function(data, label):
    """Apply randomization and standardization to the dataset for training. """
    
    # Randomly crop a section of the image with padding of 4 pixels.
    data = tf.pad(data, [[4,4],[4,4],[0,0]])
    data = tf.random_crop(data, [params.IMAGE_SIZE, 
                                 params.IMAGE_SIZE, 
                                 params.CHANNELS])
    
    # Randomly flip the image horizontally.
    data = tf.image.random_flip_left_right(data)
    
    # Scale values to range from 0.0 to 1.0 in float32 format.
    data = tf.cast(data, tf.float32)/255.0
    
    return data, label

def standardization_function(data, label):
    """Scale values to range from 0.0 to 1.0 in float32 format. """
    
    data = tf.cast(data, tf.float32)/255.0
    
    return data, label
