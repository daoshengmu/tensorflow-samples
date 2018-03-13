import numpy as np
import tensorflow as tf

tensor_1d = np.array([1.3, 1, 4.0, 23.99])
tf_tensor = tf.convert_to_tensor(tensor_1d, dtype=tf.float64)
print tensor_1d
print tensor_1d[2]
print tensor_1d.ndim
print tensor_1d.shape
print tensor_1d.dtype

with tf.Session() as sess:
    print sess.run(tf_tensor)
    print sess.run(tf_tensor[0])
    print sess.run(tf_tensor[2])

