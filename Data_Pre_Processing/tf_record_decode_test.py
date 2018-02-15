# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
from tf_records_win import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from augmentation_win import scale_randomly_image_with_annotation_with_fixed_size_output
from pascal_voc_win import pascal_segmentation_lut
from training_win import get_valid_logits_and_labels, get_labels_from_annotation_batch
from skimage import io
import numpy as np

tfrecord_filename = './patient_tfrecord/patient_cross_val-3.tfrecords'

filename_queue = tf.train.string_input_producer(
    [tfrecord_filename], num_epochs=1)

image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)


image_train_size = [200, 200]

resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(
                                        image, annotation, image_train_size,
                                        min_relative_random_scale_change=1,
                                        max_realtive_random_scale_change=1)

image_batch, annotation_batch = tf.train.batch(
                [resized_image, resized_annotation],
                 batch_size = 2,
                 capacity=32,
                 num_threads = 4,
                 name = 'input_quene_operation')
annotation_batch = tf.squeeze(annotation_batch)

weights = tf.to_float(tf.not_equal(annotation_batch, 255))
#pred_np = np.ones((200, 200, 22))
num_classes = 4

preds = tf.fill([2, 200,200, 4], 1/num_classes)
predictions = tf.argmax(preds, axis = 3)

#pascal_voc_lut = pascal_segmentation_lut()
        # A look-up table for pascal segmentation dataset
#class_labels = list(pascal_voc_lut.keys()) 
class_labels = [0, 1, 2, 3]

valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(
                annotation_batch_tensor = annotation_batch,
                logits_batch_tensor = preds,
                class_labels = class_labels)
#labels = get_labels_from_annotation_batch(annotation_batch, class_labels)
#
#cross_entropies = tf.nn.softmax_cross_entropy_with_logits(
#                logits = valid_logits_batch_tensor,
#                labels = valid_labels_batch_tensor)
#
#cross_entropy_sum = tf.reduce_mean(cross_entropies)

accuracy_iou, update_op = tf.metrics.mean_iou(labels = tf.argmax(valid_labels_batch_tensor, axis = 1),
                                           predictions = tf.argmax(valid_labels_batch_tensor, axis =1),
                                           num_classes = 4)

"""Run the Session for debugging
"""
global_vars_init_op = tf.global_variables_initializer()
local_vars_init_op = tf.local_variables_initializer()
combined_op = tf.group(local_vars_init_op, global_vars_init_op)
with tf.Session() as sess:
    sess.run(combined_op)    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    for i in range(396):
        img, ann, pred, prediction, _accuracy_iou, _ = sess.run(
                    [image_batch, annotation_batch, preds, predictions, accuracy_iou, update_op])
#        print(np.unique(ann))
        print(_accuracy_iou)
#    for i in range(1):
#        img, ann = sess.run(
#                [image, annotation])
#        io.imshow(img)
coord.request_stop()
coord.join(threads)