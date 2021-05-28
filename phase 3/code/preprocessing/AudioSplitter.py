import os
import matplotlib.pyplot as plt
import librosa
from librosa import display
from pydub import AudioSegment


"""
Use the same structure as I used here.
Check messenger for screenshot of folder structure
Using a different folder structure might lead to issues

Simply copy and paste all the files into the Input folder.
Once they are converted, they'll be placed in the Output folder.
Create the Output folder first. Make sure it's empty.

Folder Structure:

AudioSplitter.py
AudioFiles (Folder)
--Input (Folder)
----audiofile1.wav
----audiofile2.wav
--Output
----audiofile1_1.wav
----audiofile1_2.wav
----audiofile2_1.wav
----audiofile2_2.wav

"""

input_folder = 'AudioFiles/Input'
output_folder = 'AudioFiles/Output'


def display_audio_files(folder_name):

    for file_name in os.listdir(folder_name):

        # Loading the audio files here.
        target = folder_name + '/' + file_name
        data, sampling_rate = librosa.load(target)

        # This is just for comparison's sake.
        # If it's a full length 10s audio, the output graph will be 12 units long
        # If it's a half length 5s audio, the output graph will be 6 units long
        if folder_name[-5:] == 'Input': x = 12
        elif folder_name[-6:] == 'Output': x = 6
        else: x = 12

        plt.figure(figsize=(x, 4))
        librosa.display.waveplot(data, sampling_rate)
        plt.show()


def split_all_audio_files(inp_folder, op_folder):

    for file_name in os.listdir(inp_folder):

        # Loading Audio files here,
        # Doing some string matching to remove the .wav extension, then adding _1 or _2 labels
        # Then re-adding the .wav extension.
        # The "op_target_1" and "op_target_2" files will be put into output
        inp_target = inp_folder + '/' + file_name
        op_tgt = file_name[:len(file_name) - 4]
        op_target_1 = op_folder + "/" + op_tgt + "_1" + ".wav"
        op_target_2 = op_folder + "/" + op_tgt + "_2" + ".wav"

        # Using pydub cuz librosa was being weird.
        # Just slicing the first half, then second half.
        audio = AudioSegment.from_wav(inp_target)
        first_half = audio[:5000]
        second_half = audio[5000:]

        # Exporting each half as an audio file
        first_half.export(op_target_1, format="wav")
        print(op_target_1, "exported.")
        second_half.export(op_target_2, format="wav")
        print(op_target_2, "exported.")


if __name__ == '__main__':

    split_all_audio_files(input_folder, output_folder)

    display_audio_files(input_folder)
    
    # display_audio_files(output_folder)
