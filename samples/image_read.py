import matplotlib.image as mp_image
import matplotlib.pyplot as plt
import tensorflow as tf

filename = "myimg.jpg"
input_image = mp_image.imread(filename)

print("input dim = ", input_image.ndim)
print("input shape = ", input_image.shape)

my_image = tf.placeholder("uint8", [None, None, 3])
slice = tf.slice(my_image, [10, 0, 0], [60, -1, -1])

with tf.Session() as session:
  result = session.run(slice, feed_dict={my_image: input_image})
  print(result.shape)

# plt.imshow(input_image) # show the original result.
plt.imshow(result) # show the slice result.
plt.show()