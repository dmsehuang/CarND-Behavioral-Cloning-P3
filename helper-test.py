import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from helper import *

# Test read line function
lines = read_lines_from_file('./data/driving_log.csv')
for i in range(5):
    print(lines[i])

# Test line process function
line = lines[155]
images, angles = process_line(line)
for i in range(len(images)):
    image = images[i]
    angle = angles[i]
    plt.imshow(image)
    plt.show()
    print(angle)

# Test generator
train_generator = generator(lines, batch_size = 2)
for i in range(3):
    print("Generate data step:", i)
    X_train, y_train = next(train_generator)
    for j in range(len(X_train)):
        plt.imshow(X_train[j])
        plt.show()
        print(y_train[j])
