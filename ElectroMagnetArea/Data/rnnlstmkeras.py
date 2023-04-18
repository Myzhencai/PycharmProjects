# https://cloud.tencent.com/developer/news/706635
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

inputs = keras.Input(shape=(784, ), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
model.summary()

x_train, y_train = (
    np.random.random((60000, 784)),
    np.random.randint(10, size=(60000, 1)),
)
x_test, y_test = (
    np.random.random((10000, 784)),
    np.random.randint(10, size=(10000, 1)),
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train, y_train, batch_size=64, epochs=1)

# Save predictions for future checks
predictions = model.predict(x_test)


# Save the model
model.save('path_to_my_model.h5')

# Recreate the exact same model purely from the file
new_model = keras.models.load_model('path_to_my_model.h5')

# Check that the state is preserved
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

# Note that the optimizer state is preserved as well:
# you can resume training where you left off.
new_model.fit(x_train, y_train, batch_size=64, epochs=1)
