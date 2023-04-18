import os
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image

pb_model_dir = "/home/gaofei/PycharmProjects/ElectroMagnetArea/saved_model"
h5_model = "/home/gaofei/PycharmProjects/ElectroMagnetArea/modelfile/Matrix.h5"
#
# Loading the Tensorflow Saved Model (PB)
model = tf.keras.models.load_model(pb_model_dir)
print(model)

# Saving the Model in H5 Format
keras.models.save_model(model, h5_model)

# # Loading the H5 Saved Model
# loaded_model_from_h5 = tf.keras.models.load_model(h5_model)
# print(loaded_model_from_h5.summary())


# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
#
# New_Model = tf.saved_model.load(pb_model_dir) # Loading the Tensorflow Saved Model (PB)
# New_Model =  tf.compat.v2.saved_model.load(pb_model_dir, None)
# print(New_Model.summary())
# # Saving the Model in H5 Format and Loading it (to check if it is same as PB Format)
# tf.keras.models.save_model(New_Model, 'New_Model.h5') # Saving the Model in H5 Format

#
# loaded_model_from_h5 = tf.keras.models.load_model('New_Model.h5') # Loading the H5 Saved Model
# print(loaded_model_from_h5.summary())
