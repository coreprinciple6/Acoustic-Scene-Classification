import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display
from scipy.stats.mstats import gmean
import soundfile
from audiomentations import Compose, Shift


input_folder = 'AudioFiles/Input'
output_folder = 'AudioFiles/Output'   # (this is declared inside function)


def display_audio_files(folder_name):

    for file_name in os.listdir(folder_name):

        # Loading the audio files here.
        target = folder_name + '/' + file_name
        data, sampling_rate = librosa.load(target)

        x = 10

        plt.figure(figsize=(x, 4))
        librosa.display.waveplot(data, sampling_rate)
        plt.show()


def add_noise(audiofile):

    data, sampling_rate = librosa.load(audiofile)

    x = 10
    plt.figure(figsize=(x, 4))
    librosa.display.waveplot(data, sampling_rate)
    plt.title(audiofile)
    plt.ylim(-0.02, 0.02)
    plt.show()

    geom_mean = gmean(abs(data))
    factor = 0.1 * np.random.normal(0, 5, len(data))

    aug_data = data

    for i in range(0, len(data)):
        if aug_data[i] < 0:
            if aug_data[i] > -geom_mean:
                aug_data[i] = aug_data[i] / 2
            else:
                aug_data[i] = aug_data[i] * 2
        if aug_data[i] >= 0:
            if aug_data[i] > geom_mean:
                aug_data[i] = aug_data[i] * 2
            else:
                aug_data[i] = aug_data[i] / 2

    noise = 0.25 * np.random.randn(len(data))
    aug_data = (aug_data * factor * noise) + 0.002

    # plt.figure(figsize=(x, 4))
    # librosa.display.waveplot(aug_data, sampling_rate)
    # plt.title("Augmented - " + audiofile)
    # plt.ylim(-0.02, 0.02)
    # plt.show()

    return aug_data, sampling_rate


"""Run this AFTER add noise and pass output parameters of Add Noise to this"""

def randomize_time(data, sampling_rate, audiofile, op_name):


    # x = 10
    # plt.figure(figsize=(x, 4))
    # librosa.display.waveplot(data, sampling_rate)
    # plt.title(audiofile)
    # plt.ylim(-0.02, 0.02)
    # plt.show()

    augment = Compose([
        Shift(min_fraction=-0.4, max_fraction=0.4, p=1.0)
    ])

    augmented_data = augment(samples=data, sample_rate=sampling_rate)

    x = 10
    plt.figure(figsize=(x, 4))
    librosa.display.waveplot(augmented_data, sampling_rate)
    plt.title("Augmented - " + audiofile)
    plt.ylim(-0.02, 0.02)
    plt.show()

    output_folder = 'AudioFiles/Output'

    out_target = output_folder + '/aug' + op_name
    soundfile.write(file=out_target, data=augmented_data, samplerate=sampling_rate)


# display_audio_files(input_folder)
# display_audio_files(output_folder)

count = 0

for file_name in os.listdir(input_folder):

    # Loading the audio files here.
    target = input_folder + '/' + file_name

    # running both noise and shifting
    randomize_time((add_noise(target))[0], (add_noise(target))[1], target, file_name)

    count += 1

print(count, 'files augmented.')


