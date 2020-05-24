import tensorflow as tf

graph3 =  tf.Graph()

with graph3.as_default():
    Matrix_one = tf.constant([[1, 2, 3], [2, 3, 4], [3, 4, 5]])