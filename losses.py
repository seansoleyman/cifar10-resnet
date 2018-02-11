import tensorflow as tf

def total_loss(logits, labels):
    """Calculate the total loss including regularization losses. """
    with tf.name_scope('loss'):
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
      
        # The total loss is defined as the cross entropy loss plus all of the 
        # weight decay terms (L2 loss).
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(
            [cross_entropy_mean] + reg_losses, name='total_loss')
        
        # Add summaries: regularization losses, cross entropy mean, total loss. 
        for l in reg_losses + [cross_entropy_mean] + [total_loss]:
            tf.summary.scalar(l.op.name, l)
      
        return total_loss
