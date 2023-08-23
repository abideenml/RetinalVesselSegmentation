import tensorflow as tf
print(tf.test.gpu_device_name())
print(tf.config.list_physical_devices('GPU'))