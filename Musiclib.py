#!/usr/bin/env python
# coding: utf-8

# # Musical Note classification
#
# !pip install kapre
#
# @inproceedings{choi2017kapre,
#   title={Kapre: On-GPU Audio Preprocessing Layers for a Quick Implementation of Deep Neural Network Models with Keras},
#   author={Choi, Keunwoo and Joo, Deokjin and Kim, Juho},
#   booktitle={Machine Learning for Music Discovery Workshop at 34th International Conference on Machine Learning},
#   year={2017},
#   organization={ICML}
# }

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from icecream import ic
from scipy.io import wavfile as wav
from kapre import STFT, Magnitude, MagnitudeToDecibel
from kapre.composed import get_melspectrogram_layer, get_log_frequency_spectrogram_layer

# Get files
data_dir = r'/home/hhj/Documents/UniversityDocs/AIE605-SpecialTopicInAI/Project/Datasetv6'
MusicalNotesClasses = np.array(tf.io.gfile.listdir(str(data_dir)))

print('Muisal Notes:', MusicalNotesClasses)
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.wav')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
for note in MusicalNotesClasses:
    print(f'Number of examples for the label {note}:',
          len(tf.io.gfile.listdir(str(data_dir + '//' + note))))
print('Example file tensor:', filenames[0])
filenames = [str(file)[2:-1] for file in filenames.numpy()]


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    str_label = parts[-2]
    int_label = np.where(MusicalNotesClasses == str_label)[0][0]
    return int_label


def load_data():
    for filename in filenames:
        sr, audio_data = wav.read(filename)
        sample_for_100ms = sr // 10
        audio_data = audio_data[sr // 2:sr, :]
        audio_data = np.split(audio_data, sample_for_100ms)
        audio_data = np.array(audio_data).reshape((-1, sample_for_100ms, 2))
        np.delete(audio_data, -1)
        label = get_label(filename)
        for audio_segment in audio_data:
            yield audio_segment, label


audio_ds = tf.data.Dataset.from_generator(load_data, (tf.float32, tf.int32))
total_ds_size = 5 * len(filenames)
train_size = int(total_ds_size * 0.95)
test_size = int(total_ds_size * 0.5)
audio_ds = audio_ds.shuffle(1024).batch(1).prefetch(tf.data.experimental.AUTOTUNE)
train_ds = audio_ds.take(train_size)
test_ds = audio_ds.skip(train_size).take(test_size)

sample_rate = 44100
input_shape = (sample_rate // 10), 2

resnet50 = tf.keras.applications.ResNet50V2(include_top=False,
                                            input_shape=(254, 244, 3),
                                            )

resnet50.trainable = False
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
    STFT(n_fft=2048,
         win_length=2018,
         hop_length=1024,
         window_name=None,
         pad_end=False,
         input_data_format='channels_last',
         output_data_format='channels_last',
         input_shape=input_shape),
    Magnitude(),
    MagnitudeToDecibel(),
    tf.keras.layers.Conv2D(128, (4, 4), padding='same'),
    tf.keras.layers.Conv2D(128, (4, 4), padding='same'),
    tf.keras.layers.Conv1D(128, 4, strides=2),
    tf.keras.layers.Conv1D(244, 4, strides=2),
    tf.keras.layers.Reshape((254, 244, 3)),
    resnet50,
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(num_samples * 8, activation='relu'),
    tf.keras.layers.Dense(num_samples * 4, activation='relu'),
    tf.keras.layers.Dense(num_samples * 2, activation='relu'),
    tf.keras.layers.Dense(num_samples, activation='softmax'),
])

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              )

model.fit(train_ds,
          verbose=1,
          validation_data=test_ds,
          epochs=128,
          callbacks=[
              tf.keras.callbacks.EarlyStopping(
                  verbose=1,
                  patience=20, ),
              tf.keras.callbacks.ReduceLROnPlateau(
                  verbose=1,
                  patience=5,
                  factor=0.95,
              )
          ])
