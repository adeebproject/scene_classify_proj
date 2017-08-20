# -*- coding: utf-8 -*-
"""
Created on Fri Jul  14 16:04:04 2017

@author: Admin
"""
import time
import os
import sys
import tensorflow as tf
from core import create_dataset
from core import training_proc
from core import visualise

flags = tf.app.flags

#State dataset and record directory
flags.DEFINE_string('dataset_dir', '', 'String: dataset directory')

flags.DEFINE_string('log_dir', '', 'String: Training log directory')

flags.DEFINE_string('evaluation_dir', '', 'String: Your evaluation log directory')

flags.DEFINE_string('action', '', 'String: The operation e.g train(dataset_dir, log_dir, tfrecord_filename), evaluation(dataset_dir, log_dir, evaluation_dir, tfrecord_filename), visualise(dataset_dir, log_dir, tfrecord_filename, convlayer) and create_dataset(dataset_dir, tfrecord_filename)')

# The number of images in the validation set. You would have to know the total number of examples in advance. This is essentially your evaluation dataset.
flags.DEFINE_float('validation_size', 0.2, 'Float: The proportion of examples in the dataset to be used for validation')

# The number of shards to split the dataset into
flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', '', 'String: The output filename to name your TFRecord file')
#The Conv layer for visualisation
flags.DEFINE_string('convlayer', '', 'String: The convolutional layer to visualise When calling Visualise')

FLAGS = flags.FLAGS


#Create an array with the number of operation to be performed

Ops=['create_dataset', 'train', 'visualise', 'evaluate']
print('Welcome! \n The following are the various operations you can perform!')
for op in Ops:
    print('{0}{1}'.format('::',op))
def get_input():
    operation=FLAGS.action
    
    return operation

#we can now execute thye selected operation
#============================================Main Operaytion==================#
def main():
    #retireve the input operation
    operation=get_input()
    if not operation in Ops:
        print('wrong operation!')
    else:
        print('Initializing ' + operation + ' operation...')
        if operation == 'create_dataset':           
            if not FLAGS.dataset_dir:
                print('Dataset directory is required!')
            elif not FLAGS.tfrecord_filename:
                print('TFRecord filename is required!')
            else:
                create_dataset.main(FLAGS.dataset_dir, FLAGS.validation_size, FLAGS.num_shards, FLAGS.random_seed, FLAGS.tfrecord_filename)
        elif operation == 'visualise':
            if not FLAGS.dataset_dir:
                print('Dataset directory is required!')
            elif not FLAGS.log_dir:
                print('Log directory is required!')
            elif not FLAGS.convlayer:
                print('The convolutional layer to visualise is required!')
            else:
                visualise.main(FLAGS.dataset_dir, FLAGS.log_dir, FLAGS.tfrecord_filename, FLAGS.convlayer)
        elif operation == 'train':
            if not FLAGS.dataset_dir:
                print('Dataset directory is required!')
            if not FLAGS.log_dir:
                print('log directory is required!')
            elif not FLAGS.tfrecord_filename:
                print('TFRecord filename is required!')
            else:
                training_proc.main(FLAGS.dataset_dir, FLAGS.log_dir, FLAGS.tfrecord_filename)
        elif operation == 'evaluate':
            from core import evaluation_proc
            evaluation_proc.main(FLAGS.dataset_dir, FLAGS.log_dir, FLAGS.evaluation_dir, FLAGS.tfrecord_filename)


if __name__ == '__main__':
    main()