#import libraries
import os
from numpy import save
from sklearn import preprocessing
import python_speech_features as mfcc
from scipy.io import wavfile

#set path and initilaise variables

#there are 2 categories in my given example(indoor and outdoor)
indoor_path = 'E:/yoda/Reduced Dset/Test/Indoor'
outdoor_path = 'E:/yoda/Reduced Dset/Test/Outdoor'

#arrays to hold mfcc and their categories
x_test = []
y_test = []

#counter for categories. 1 is indoor, 2 is outdoor
i=0

# contains the filenames from both categories
in_files = [f for f in os.listdir(indoor_path) if os.path.isfile(os.path.join(indoor_path, f))]
out_files = [f for f in os.listdir(outdoor_path) if os.path.isfile(os.path.join(outdoor_path, f))]


# code to extract mfcc then save as a numpy array along with categories in x test, y test
print('extracting mfcc')

for _file in in_files:
    sampling_rate, audio_signal =wavfile.read((os.path.join(indoor_path, _file)))
    mfcc_feature = mfcc.mfcc(audio_signal,sampling_rate, 0.01, 0.001,numcep=20,nfilt=30,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    #mfcc_feature is a numpy array of mfcc coefficients
    x_test.append(mfcc_feature)
    y_test.append(1)

print('indoor done')
    
for _file in out_files:
    sampling_rate, audio_signal =wavfile.read((os.path.join(outdoor_path, _file)))
    mfcc_feature = mfcc.mfcc(audio_signal,sampling_rate, 0.01, 0.001,numcep=20,nfilt=30,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    x_test.append(mfcc_feature)
    y_test.append(2)

print('outdoor done')

#save data to x test and y test

save('x_test.npy', x_test)
print('saved x file!')

save('y_test.npy', y_test)
print('saved y file!')