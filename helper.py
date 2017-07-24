import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle

'''Read from CSV file and return lines'''
def read_lines_from_file(file_path):
    lines = []
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None) # skip the header
        for line in reader:
            lines.append(line)
    return lines

'''Generator produces images and steering angles'''
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # process batch sample
                images_data, angles_data = process_line(batch_sample)
                images.extend(images_data)
                angles.extend(angles_data)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

'''Process line, return images, angles, flipped images and flipped angles'''
def process_line(line):
    images = []
    angles = []

    image_dir = './data/IMG/'
    correction = [0, 0.2, -0.2]
    for i in range(3):
        # read the image
        image_path = image_dir + line[i].split('/')[-1]
        image = mpimg.imread(image_path)
        images.append(image)
        angle = float(line[3]) + correction[i]
        angles.append(angle)

        # flip the image
        image_flipped = np.fliplr(image)
        images.append(image_flipped)
        angle_flipped = -angle
        angles.append(angle_flipped)
    return images, angles
