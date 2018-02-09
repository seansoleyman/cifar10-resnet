# Decay rate for the batch normalization layers. 
BN_MOMENTUM = 0.9

# L2 regularization scale for the shortcut connection with non-unity stride. 
SHORTCUT_L2_SCALE = 0.0001

import tensorflow as tf

"""Adds two sets of BN->relu->conv layers, as well as a shortcut. """
def _residual_block(inputs, filters, training, stride=1):
    bn1 = tf.layers.batch_normalization(inputs, 
                                        momentum=BN_MOMENTUM, 
                                        training=training, 
                                        name='bn1')
    relu1 = tf.nn.relu(bn1, name='relu1')
    conv1 = tf.layers.conv2d(relu1, filters=filters, kernel_size=3, 
                             strides=stride, 
                             padding='same', 
                             name='conv1')
    
    bn2 = tf.layers.batch_normalization(conv1, 
                                        momentum=BN_MOMENTUM, 
                                        training=training, 
                                        name='bn2')
    relu2 = tf.nn.relu(bn2, name='relu2')
    conv2 = tf.layers.conv2d(relu2, filters=filters, kernel_size=3, 
                             padding='same', name='conv2')
    
    shortcut = inputs
    if stride>1:
        shortcut = tf.layers.conv2d(inputs, 
                                    filters=filters, 
                                    kernel_size=1, 
                                    strides=stride, 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(SHORTCUT_L2_SCALE), 
                                    name='conv_shortcut')
    
    return tf.add(shortcut, conv2, name='add')

def inference(images, training, n=20):
    """Builds a ResNet for CIFAR-10.

    Args:
        images: Input for the network. 
            Tensor with shape (batches, height=32, width=32, channels=3). 
        training: Boolean tensor that specifies if the network is
            being trained. This is necessary for batch normalization. 
        n: number of residual blocks in each convolutional section. 

    Returns:
        A tensor that evaluates scores for the CIFAR-10 classes. 
    """
    
    assert n>0

    with tf.variable_scope('inference_model'):
        # Normalize inputs. 
        bn1 = tf.layers.batch_normalization(images, 
                                            momentum=BN_MOMENTUM, 
                                            training=training, 
                                            name='bn1')
    
        # Construct the first convolutional layer with no activation. 
        conv1 = tf.layers.conv2d(bn1, filters=16, kernel_size=3, 
                                 padding='same', name='conv')
        
        # Stack n residual blocks with 16 feature maps sized 32x32. 
        with tf.variable_scope('stack1'):
            block = conv1
            filters = 16
            for i in range(1,n+1):
                with tf.variable_scope('block{}'.format(i)):
                    block = _residual_block(block, filters, training)
            
        # Stack n residual blocks with 32 feature maps sized 16x16. 
        with tf.variable_scope('stack2'):
            filters *= 2
            with tf.variable_scope('block1'):
                block = _residual_block(block, filters, training, stride=2)
            for i in range(2,n+1):
                with tf.variable_scope('block{}'.format(i)):
                    block = _residual_block(block, filters, training)
            
        # Stack n residual blocks with 64 feature maps sized 8x8. 
        with tf.variable_scope('stack3'):
            filters *= 2
            with tf.variable_scope('block1'):
                block = _residual_block(block, filters, training, stride=2)
            for i in range(2,n+1):
                with tf.variable_scope('block{}'.format(i)):
                    block = _residual_block(block, filters, training)
        
        # Stack n residual blocks with 128 feature maps sized 4x4. 
        with tf.variable_scope('stack4'):
            filters *= 2
            with tf.variable_scope('block1'):
                block = _residual_block(block, filters, training, stride=2)
            for i in range(2,n+1):
                with tf.variable_scope('block{}'.format(i)):
                    block = _residual_block(block, filters, training)
                    
        # Batch Normalization and Rectified Linear Unit layers. 
        bn_final = tf.layers.batch_normalization(block, 
                                                 momentum=BN_MOMENTUM, 
                                                 training=training, 
                                                 name='bn_final')
        relu_final = tf.nn.relu(bn_final, name='relu_final')
    
        # Global Average Pooling and Classifier. 
        global_average_pool = tf.layers.average_pooling2d(relu_final, 
                                                          pool_size=(4, 4),
                                                          strides=1, 
                                                          name='global_average_pooling')
        global_average_flat = tf.reshape(global_average_pool, 
                                         [-1, filters], 
                                         name='flatten')
        logits = tf.layers.dense(global_average_flat, 
                                 units=10, 
                                 name='fully_connected')

    return logits
