import tensorflow as tf
hello = tf.constant("hello my tensorflow")
sess = tf.Session()
print(sess.run(hello))
