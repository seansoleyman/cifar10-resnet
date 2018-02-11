"""Train the CIFAR-10 network. """

import time
import math

import tensorflow as tf

import params
import net
import data
import losses

import csv
import os

def create_train_op(total_loss, global_step):
    """Create an operation for training the CIFAR-10 model.

    Args:
        total_loss: total loss from loss().
        global_step: Integer Variable counting the number of training steps
             processed.
    Returns:
        train_op: op for training.
    """
    with tf.name_scope('train_op'):
        # Decay the learning rate based on the number of steps.
        lr = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
                                         params.LR_BOUNDARIES,
                                         params.LR_VALUES)
        tf.summary.scalar('learning_rate', lr)
    
        # Compute gradients.
        opt = tf.train.MomentumOptimizer(lr, 
                                         momentum=params.MOMENTUM, 
                                         use_nesterov=True)
        grads = opt.compute_gradients(total_loss)
    
        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
      
        # Apply additional operations such as those needed for batch norm
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
    
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
    
        with tf.control_dependencies(
                [apply_gradient_op] + extra_update_ops):
            train_op = tf.no_op(name='train')
    
    return train_op


def train():
    """Train CIFAR-10 for a number of steps."""
    
    with tf.device('/cpu:0'), tf.name_scope('input'):
        global_step = tf.train.get_or_create_global_step()
    
        print("Loading CIFAR-10 Data")
        cifar10 = data.Cifar10()
      
        images_placeholder = tf.placeholder(
                cifar10.train_images.dtype, 
                (None, params.IMAGE_SIZE, params.IMAGE_SIZE, params.CHANNELS), 
                name='images_placeholder')
        labels_placeholder = tf.placeholder(
                cifar10.train_labels.dtype, 
                (None,), 
                name='labels_placeholder')
    
        train_data_dict = {images_placeholder: cifar10.train_images, 
                           labels_placeholder: cifar10.train_labels}
        test_data_dict = {images_placeholder: cifar10.test_images, 
                          labels_placeholder: cifar10.test_labels}
      
        training_dataset = tf.data.Dataset.from_tensor_slices(
                (images_placeholder, labels_placeholder))
        training_dataset = training_dataset.prefetch(params.SHUFFLE_BUFFER)
        training_dataset = training_dataset.map(
                data.randomization_function, 
                num_parallel_calls=params.NUM_THREADS)
        training_dataset = training_dataset.shuffle(
                buffer_size=params.SHUFFLE_BUFFER)
        training_dataset = training_dataset.batch(params.BATCH_SIZE)
        training_dataset = training_dataset.repeat()
        training_dataset = training_dataset.prefetch(params.TRAIN_OUTPUT_BUFFER)
      
        validation_dataset = tf.data.Dataset.from_tensor_slices(
                (images_placeholder, labels_placeholder))
        validation_dataset = validation_dataset.map(
                data.standardization_function, 
                num_parallel_calls=params.NUM_THREADS)
        validation_dataset = validation_dataset.batch(
                params.BATCH_SIZE)
        validation_dataset = validation_dataset.prefetch(
            params.VALIDATION_OUTPUT_BUFFER)
      
        iterator = tf.contrib.data.Iterator.from_structure(
                training_dataset.output_types, training_dataset.output_shapes)
        next_element = iterator.get_next()
        training_init_op = iterator.make_initializer(training_dataset)
        validation_init_op = iterator.make_initializer(validation_dataset)
      
        training_placeholder = tf.placeholder_with_default(
              False, (), name='training_placeholder')

    print("Building TensorFlow Graph")
    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = net.inference(next_element[0], training=training_placeholder)

    # Calculate loss.
    total_loss = losses.total_loss(logits, next_element[1])
    
    with(tf.name_scope('accuracy')):
        correct = tf.nn.in_top_k(logits, next_element[1], 1)
        number_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = create_train_op(total_loss, global_step)
    
    init = tf.global_variables_initializer()
    
    print("Starting TensorFlow Session")
    
    saver = tf.train.Saver()
    tf_file_writer = tf.summary.FileWriter(params.TRAIN_DIR, 
                                           tf.get_default_graph())
    merged_summary = tf.summary.merge_all()
    csv_file_dir = os.path.join(params.TRAIN_DIR, 'log.csv')
    with tf.Session() as sess, open(csv_file_dir, 'w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Step", "Train Error", "Test Error", "Step Time"])
        
        print("Initializing Global Variables")
        init.run()
        
        print("Training in Progress")
        while global_step.eval() < params.TRAIN_STEPS:
            # Run a number of training steps set by params.LOG_FREQUENCY
            training_init_op.run(feed_dict=(train_data_dict))
            start_time = time.perf_counter()
            for _ in range(0,params.LOG_FREQUENCY):
                batch_loss, summary_str = sess.run(
                        [train_op, total_loss, merged_summary], 
                        feed_dict={training_placeholder: True})[1:]
            end_time = time.perf_counter()
            average_time_per_step = (end_time-start_time)/params.LOG_FREQUENCY
            
            # Write a summary of the last training batch for TensorBoard
            tf_file_writer.add_summary(summary_str, global_step.eval())
            
            # Calculate error rate based on the full train set.  
            validation_init_op.run(feed_dict=train_data_dict)
            total_correct = 0
            n_train_validation_steps = math.ceil(
                    params.NUM_TRAIN_EXAMPLES/params.BATCH_SIZE)
            for _ in range(0, n_train_validation_steps):
                total_correct += number_correct.eval()
            train_error_rate = 1.0-total_correct/params.NUM_TRAIN_EXAMPLES
            
            # Calculate error rate based on the full test set.  
            validation_init_op.run(feed_dict=test_data_dict)
            total_correct = 0
            n_test_validation_steps = math.ceil(
                    params.NUM_TEST_EXAMPLES/params.BATCH_SIZE)
            for _ in range(0, n_test_validation_steps):
                total_correct += number_correct.eval()
            test_error_rate = 1.0-total_correct/params.NUM_TEST_EXAMPLES
            
            print("Step:", global_step.eval())
            print("  Train Set Error Rate:", train_error_rate)
            print("  Test Set Error Rate:", test_error_rate)
            print("  Average Training Time per Step:", average_time_per_step)
            log_writer.writerow([global_step.eval(), 
                                 train_error_rate, 
                                 test_error_rate, 
                                 average_time_per_step])
        saver.save(sess, os.path.join(params.TRAIN_DIR, "model.ckpt"))
    tf_file_writer.close()

# Clear the training directory
if tf.gfile.Exists(params.TRAIN_DIR):
    tf.gfile.DeleteRecursively(params.TRAIN_DIR)
tf.gfile.MakeDirs(params.TRAIN_DIR)

# Train the network
train()
