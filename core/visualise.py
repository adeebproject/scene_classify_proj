# -*- coding: utf-8 -*-
"""
Created on Thur Jun  29 18:13:17 2017

@author: meadeeb
"""
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib import slim
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time
from models.slim.preprocessing import inception_preprocessing
from models.slim.nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from core.training_proc import get_split, load_batch




def main(dataset_dir, log_dir, tfrecord_filename, convlayer):
    
    plt.style.use('ggplot')
    image_size=299
    img_size=image_size*image_size*3

    file_pattern=tfrecord_filename + '_%s_*.tfrecord'
    #State the batch_size to evaluate each time, which can be a lot more than the training batch
    batch_size = 36

    #State the number of epochs to evaluate
    num_epochs = 1

    #Get the latest checkpoint file
    checkpoint_file = tf.train.latest_checkpoint(log_dir)

    #Just construct the graph from scratch again
    
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        #Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
        dataset = get_split('train', dataset_dir, file_pattern, tfrecord_filename)
        images, raw_images, labels = load_batch(dataset, batch_size = batch_size, height=image_size, width=image_size, is_training = False)

        #Create some information about the training steps
        x = tf.placeholder(tf.float32, shape=[None, img_size], name='x')
        #Now create the inference model but set is_training=False
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(images, num_classes = dataset.num_classes, is_training = False)

        # #get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        #Just define the metrics to track without the loss or whatsoever
        conv2dx = end_points[convlayer]
        predictions = tf.argmax(end_points['Predictions'], 1)
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        
        def plot_conv_layer(layer_name, images):
            # Create a feed-dict containing just one image.
            # Calculate and retrieve the output values of the layer
            # when inputting that image.
            for j in range(5):
                image = images[j]
                values = sess.run(layer_name, feed_dict={x:np.reshape([image], [1, img_size], order='F')})
            
                # Number of filters used in the conv. layer.
                num_filters = values.shape[3]
            
                # Number of grids to plot.
                # Rounded-up, square-root of the number of filters.
                grids = math.ceil(math.sqrt(num_filters))
                
                # Create figure with a grid of sub-plots.
                fig, axes = plt.subplots(grids, grids)
                
                # Plot the output images of all the filters.
                for i, ax in enumerate(axes.flat):
                    # Only plot the images for valid filters.
                    if i<num_filters:
                        # Get the output image of using the i'th filter.
                        # See new_conv_layer() for details on the format
                        # of this 4-dim tensor.
                        img = values[0, :, :, i]
            
                        # Plot image.
                        ax.imshow(img, interpolation='nearest')
                    
                    # Remove ticks from the plot.
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                # Ensure the plot is shown correctly with multiple plots
                # in a single Notebook cell.
                plt.show()
        def plot_sample_images(images, labels):
            for j in range(9):
                images = images[j]
                images.append(images)
            
            grids = math.ceil(math.sqrt(batch_size))
            fig, axes = plt.subplots(grids, grids)
            fig.subplots_adjust(hspace=0.50, wspace=0.2, top=0.97, bottom=0.06)
			        
            for i, ax in enumerate(axes.flat):
                label_name = dataset.labels_to_name[labels[i]]
                # Plot image.
                ax.imshow(images[i])
                xlabel = 'GroundTruth: ' + label_name
                # Show the classes as the label on the x-axis.
                ax.set_xlabel(xlabel)
                
                # Remove ticks from the plot.
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Ensure the plot is shown correctly with multiple plots
            # in a single Notebook cell.
            plt.show()

         #Get your supervisor
        sv = tf.train.Supervisor(logdir =  None, summary_op = None, saver = None, init_fn = restore_fn)

        #Now we are ready to run in one session
        with sv.managed_session() as sess:

            #Now we want to visualize the last batch's images just to see what our model has predicted
            raw_images, labels, predictions = sess.run([raw_images, labels, predictions])       
            plot_conv_layer(conv2dx, raw_images)
            
            logging.info('Model Visualisation completed!.')

if __name__ == '__main__':
    main(dataset_dir, log_dir, tfrecord_filename, convlayer)