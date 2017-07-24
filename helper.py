import csv
import cv2
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
                image_list, angle_list = gen_images_and_angles(batch_sample)
                images.extend(image_list)
                angles.extend(angle_list)
                flip_image_list, flip_angle_list = flip(image_list, angle_list)
                images.extend(flip_image_list)
                angles.extend(flip_angle_list)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

'''Process batch sample and generate images and angles'''
def gen_images_and_angles(batch_sample):
    images = []
    angles = []

    # Read images
    center_image_path = './data/IMG/'+batch_sample[0].split('/')[-1]
    center_image = cv2.imread(center_image_path)
    left_image_path = './data/IMG/'+batch_sample[1].split('/')[-1]
    left_image = cv2.imread(left_image_path)
    right_image_path = './data/IMG/'+batch_sample[2].split('/')[-1]
    right_image = cv2.imread(right_image_path)
    images.append(center_image)
    images.append(left_image)
    images.append(right_image)

    # Read angles
    correction = 0.2
    center_angle = float(batch_sample[3])
    left_angle = center_angle + correction
    right_angle = center_angle - correction
    angles.append(center_angle)
    angles.append(left_angle)
    angles.append(right_angle)
    return images, angles

'''Flip images and angles'''
def flip(images, angles):
    flip_images = []
    flip_angles = []
    for i in range(len(images)):
        image = images[i]
        angle = angles[i]
        image_flipped = np.fliplr(image)
        angle_flipped = -angle
        flip_images.append(image_flipped)
        flip_angles.append(angle_flipped)
    return flip_images, flip_angles
