import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from helper import *

samples = read_lines_from_file('./data/driving_log.csv')
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(len(train_samples))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# run models
from linear import linear
from lenet import lenet
from nvidia import nvidia

#models = {"linear-model" : linear(),
#          "lenet-model" : lenet(),
#          "nvidia-model" : nvidia()}

models = {"nvidia-model" : nvidia()}

for model_name in models.keys():
    print("Traning model", model_name)
    model = models[model_name]

    model.compile(loss='mse', optimizer='adam')

    history_object = model.fit_generator(train_generator,
                                        samples_per_epoch= 6*len(train_samples),
                                        validation_data=validation_generator,
                                        nb_val_samples=6*len(validation_samples),
                                        nb_epoch=8, verbose=1)

    # Visualize loss
    
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model.save(model_name + '.h5')
