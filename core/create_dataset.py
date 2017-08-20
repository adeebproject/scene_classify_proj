# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 14:35:38 2017

@author: meadeeb
"""

import random
import tensorflow as tf
import sys
from core.dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset

#====================================================DEFINE YOUR ARGUMENTS=======================================================================
flags = tf.app.flags


def main(dataset_dir, validation_size, num_shards, random_seed, tfrecord_filename):

    #==============================================================CHECKS==========================================================================
    #Check if there is a tfrecord_filename entered
    if not tfrecord_filename:
        raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

    #Check if there is a dataset directory entered
    if not dataset_dir:
        raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')

    #If the TFRecord files already exist in the directory, then exit without creating the files again
    if _dataset_exists(dataset_dir = dataset_dir, _NUM_SHARDS = num_shards, output_filename = tfrecord_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return None
    #==============================================================END OF CHECKS===================================================================

    #Get a list of photo_filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
    photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)

    #Refer each of the class name to a specific integer number for predictions later
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    #Find the number of validation examples we need
    num_validation = int(validation_size * len(photo_filenames))

    # Divide the training datasets into train and test:
    random.seed(random_seed)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids, dataset_dir = dataset_dir, tfrecord_filename = tfrecord_filename, _NUM_SHARDS = num_shards)
    _convert_dataset('validation', validation_filenames, class_names_to_ids, dataset_dir = dataset_dir, tfrecord_filename = tfrecord_filename, _NUM_SHARDS = num_shards)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, dataset_dir)

    print ('\nFinished converting the %s dataset!' %tfrecord_filename)

if __name__ == "__main__":
    main(dataset_dir, validation_size, num_shards, random_seed, tfrecord_filename)