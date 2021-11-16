
import tensorflowjs as tfjs
import tensorflow as tf

new_model = tf.keras.models.load_model('Model/model.h5')
new_model.summary()
tfjs.converters.save_keras_model(new_model, "docs/ModelJS")