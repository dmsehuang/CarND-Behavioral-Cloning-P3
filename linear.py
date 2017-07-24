from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

def linear():
    # Build a model
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model
