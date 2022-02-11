import tensorflow as tf

def weighted_cce(y_train, output_layer, debias_coefficient=1.):
    old_max = 1
    old_min = -1
    new_max = 1
    new_min = 0
    scaled_coeff = (((debias_coefficient - old_min) * (new_max-new_min)) / (old_max-old_min)) + new_min
    cross_entropy_loss = tf.reduce_mean(-tf.math.reduce_sum(y_train * tf.math.log(output_layer), axis=[1]))
    return scaled_coeff*cross_entropy_loss
