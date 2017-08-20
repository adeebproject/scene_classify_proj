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
import time
import math
from models.slim.preprocessing import inception_preprocessing
from models.slim.nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from core.training_proc import get_split, load_batch


def main(dataset_dir, log_dir, log_eval, tfrecord_filename):
    #State the batch_size to evaluate each time, which can be a lot more than the training batch
    if not os.path.exists(log_eval):
        print(log_eval+'is required! Creating!'+log_eval+'...')
        os.mkdir(log_eval)
    batch_size = 64

    #State the number of epochs to evaluate
    num_epochs = 1

    #Get the latest checkpoint file
    checkpoint_file = tf.train.latest_checkpoint(log_dir)
    image_size=299
    img_size=image_size*image_size*3
    file_pattern=tfrecord_filename + '_%s_*.tfrecord'

    #Create log_dir for evaluation information
    #if not os.path.exists(log_eval):
     #   os.mkdir(log_eval)
    plt.style.use('ggplot')
    #Just construct the graph from scratch again
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        #Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
        dataset = get_split('validation', dataset_dir, file_pattern, tfrecord_filename)
        images, raw_images, labels = load_batch(dataset, batch_size = batch_size, height=image_size, width=image_size, is_training = False)

        #Create some information about the training steps
        num_batches_per_epoch = dataset.num_samples / batch_size
        num_steps_per_epoch = num_batches_per_epoch

        #Now create the inference model but set is_training=False
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(images, num_classes = dataset.num_classes, is_training = False)

        # #get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        #Just define the metrics to track without the loss or whatsoever
        predictions = tf.argmax(end_points['Predictions'], 1)
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        precision, precision_update=slim.metrics.streaming_precision(predictions, labels)
        metrics_op = tf.group(accuracy_update, precision_update)

        

        #Create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step
        
        def plot_sample_images(images, labels, predictions):
            grids = math.ceil(math.sqrt(batch_size)-2)
            fig, axes = plt.subplots(grids, grids)
            fig.subplots_adjust(hspace=0.50, wspace=0.2, top=0.97, bottom=0.06)
			        
            for i, ax in enumerate(axes.flat):
                prediction_name, label_name = dataset.labels_to_name[predictions[i]], dataset.labels_to_name[labels[i]]
                # Plot image.
                ax.imshow(images[i])
        
                # Show true and predicted classes.
                if predictions is None:
                    xlabel = 'GroundTruth: ' + label_name
                else:
                    xlabel = 'GroundTruth: ' + label_name + '\n' + 'Prediction: ' + prediction_name
        
                # Show the classes as the label on the x-axis.
                ax.set_xlabel(xlabel)
                
                # Remove ticks from the plot.
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Ensure the plot is shown correctly with multiple plots
            # in a single Notebook cell.
            
            plt.show()
        #Create a evaluation step function
        def eval_step(sess, metrics_op, global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, global_step_count, accuracy_value, precision_value = sess.run([metrics_op, global_step_op, accuracy, precision])
            time_elapsed = time.time() - start_time

            #Log some information
            logging.info('Global Step %s:- Streaming Accuracy: %.4f : Precision: %2f (%.2f sec/step)', global_step_count, accuracy_value, precision_value, time_elapsed)

            return accuracy_value, precision_value


        #Define some scalar quantities to monitor
        tf.summary.scalar('Validation_Accuracy', accuracy)
        tf.summary.scalar('precision', precision)
        my_summary_op = tf.summary.merge_all()

        #Get your supervisor
        sv = tf.train.Supervisor(logdir = log_eval, summary_op = None, saver = None, init_fn = restore_fn)

        #Now we are ready to run in one session
        with sv.managed_session() as sess:
            for step in range(int(num_steps_per_epoch * num_epochs)):
                sess.run(sv.global_step)
                #print vital information every start of the epoch as always
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy))
                    
                #Compute summaries every 10 steps and continue evaluating
                if step % 10 == 0:
                    eval_step(sess, metrics_op = metrics_op, global_step = sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                    

                #Otherwise just run as per normal
                else:
                    eval_step(sess, metrics_op = metrics_op, global_step = sv.global_step)

            #At the end of all the evaluation, show the final accuracy
            logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy))
            logging.info('Precision: %.4f', sess.run(precision))
            #Now we want to visualize the last batch's images just to see what our model has predicted
            raw_images, labels, predictions = sess.run([raw_images, labels, predictions])
            plot_sample_images(raw_images, labels, predictions)
            confusion_matrix_= confusion_matrix(y_true=labels, y_pred=predictions)
            print('\n Evaluation Confusion Matirx:')
            print(confusion_matrix_)
            
            plt.matshow(confusion_matrix_)
            
            plt.colorbar()
            ticks = labels
            plt.xticks(ticks)
            plt.yticks(ticks)
            plt.xlabel('Prediction')
            plt.ylabel('Ground_Truth')
            plt.show()
            logging.info('Model evaluation completed.')

if __name__ == '__main__':
    main(dataset_dir, log_dir, log_eval, tfrecord_filename)