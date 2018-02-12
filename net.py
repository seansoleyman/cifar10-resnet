"""Build the CIFAR-10 ResNet."""

import tensorflow as tf

import params

"""Create two sets of BN->relu->conv layers, as well as a shortcut. """
def _residual_block(inputs, filters, initializer, training, stride=1):
    bn1 = tf.layers.batch_normalization(
        inputs, 
        momentum=params.BN_MOMENTUM, 
        training=training, 
        fused=True, 
        name='bn1')
    relu1 = tf.nn.relu(bn1, name='relu1')
    conv1 = tf.layers.conv2d(
        relu1, 
        filters=filters, 
        kernel_size=3, 
        strides=stride, 
        padding='same', 
        kernel_initializer=initializer, 
        name='conv1')
    
    bn2 = tf.layers.batch_normalization(
        conv1, 
        momentum=params.BN_MOMENTUM, 
        training=training, 
        fused=True, 
        name='bn2')
    relu2 = tf.nn.relu(bn2, name='relu2')
    conv2 = tf.layers.conv2d(
        relu2, 
        filters=filters, 
        kernel_size=3, 
        padding='same', 
        kernel_initializer=initializer, 
        name='conv2')
    
    shortcut = inputs
    if stride>1:
        shortcut = tf.layers.conv2d(
            inputs, 
            filters=filters, 
            kernel_size=1, 
            strides=stride, 
            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                params.SHORTCUT_L2_SCALE), 
            kernel_initializer=initializer, 
            name='conv_shortcut')
    
    return tf.add(shortcut, conv2, name='add')

def inference(images, training):
    """Build a ResNet for CIFAR-10.

    Args:
        images: Input for the network. 
            Tensor with shape (batches, height=32, width=32, channels=3). 
        training: Boolean tensor that specifies if the network is
            being trained. This is necessary for batch normalization. 

    Returns:
        A tensor that evaluates scores for the CIFAR-10 classes. 
    """
    initializer = tf.contrib.layers.variance_scaling_initializer()

    with tf.variable_scope('inference_model'):
    
        # Construct the first convolutional layer with no activation. 
        conv1 = tf.layers.conv2d(
            images, 
            filters=16 * params.WIDEN_FACTOR, 
            kernel_size=3, 
            padding='same', 
            kernel_initializer = initializer,
            name='conv')
        
        # Stack n residual blocks with 16*WIDEN_FACTOR feature maps sized 32x32. 
        with tf.variable_scope('stack1'):
            block = conv1
            filters = 16 * params.WIDEN_FACTOR
            for i in range(1, params.DEPTH+1):
                with tf.variable_scope('block{}'.format(i)):
                    block = _residual_block(block, filters, initializer, training)
            
        # Stack n residual blocks with 32*WIDEN_FACTOR feature maps sized 16x16. 
        with tf.variable_scope('stack2'):
            filters *= 2
            with tf.variable_scope('block1'):
                block = _residual_block(block, filters, initializer, training, stride=2)
            for i in range(2, params.DEPTH+1):
                with tf.variable_scope('block{}'.format(i)):
                    block = _residual_block(block, filters, initializer, training)
            
        # Stack n residual blocks with 64*WIDEN_FACTOR feature maps sized 8x8. 
        with tf.variable_scope('stack3'):
            filters *= 2
            with tf.variable_scope('block1'):
                block = _residual_block(block, filters, initializer, training, stride=2)
            for i in range(2, params.DEPTH+1):
                with tf.variable_scope('block{}'.format(i)):
                    block = _residual_block(block, filters, initializer, training)
                    
        # Batch Normalization and Rectified Linear Unit layers. 
        bn_final = tf.layers.batch_normalization(
            block, 
            momentum=params.BN_MOMENTUM, 
            training=training, 
            fused=True, 
            name='bn_final')
        relu_final = tf.nn.relu(bn_final, name='relu_final')
    
        # Global Average Pooling and Classifier. 
        global_average_pool = tf.layers.average_pooling2d(
            relu_final, 
            pool_size=(8, 8),
            strides=1, 
            name='global_average_pooling')
        global_average_flat = tf.reshape(
            global_average_pool, 
            [-1, filters], 
            name='flatten')
        logits = tf.layers.dense(
            global_average_flat, 
            units=10, 
            kernel_initializer = initializer,
            name='fully_connected')

    return logits
