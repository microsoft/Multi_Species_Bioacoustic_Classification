#
# step1_extract_melspectrograms.py
#
# Load detection labels, extract audio for detection and non-detection regions,
# compute and save spectrograms.
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

#%% Imports

import pandas as pd
import numpy as np
from math import floor, ceil
#import glob
import os
import librosa                ## version 0.6.3
import librosa.display
from matplotlib import pyplot
from datetime import datetime, timedelta
from joblib import Parallel, delayed   ## version 0.12.0
import multiprocessing
import gc


#%% Path configuration

current_dir = "./multispecies_bioacoustics/"

data_dir = current_dir + "Data/"
labeled_data_dir = data_dir + 'Labeled_Data/'
audio_dir = data_dir + "Raw_Audio/" + species_version
output_mel_spectrogram_dir = data_dir + "Extracted_Mel_Spectrogram/"
output_mel_spectrogram_tp_dir = output_mel_spectrogram_dir + 'tp/'
output_mel_spectrogram_fp_dir = output_mel_spectrogram_dir + 'fp/'

if not os.path.exists(output_mel_spectrogram_dir):
    os.makedirs(output_mel_spectrogram_dir)

if not os.path.exists(output_mel_spectrogram_tp_dir):
    os.makedirs(output_mel_spectrogram_tp_dir)

if not os.path.exists(output_mel_spectrogram_fp_dir):
    os.makedirs(output_mel_spectrogram_fp_dir)


#%% Step 1: import the labels

## true-positive labeled data
truepositive_labeled_data = pd.read_csv(labeled_data_dir + 'true_positive_metadata.csv')
print("true positive labeled data:", truepositive_labeled_data.shape)

## false-positive labeled data
falsepositive_labeled_data = pd.read_csv(labeled_data_dir + 'false_positive_metadata.csv')
print("false positive labeled data:", falsepositive_labeled_data.shape)


#%% Step 2: match each labeled data segment to the corresponding audio file

all_audio_filenames = os.listdir(audio_dir)
truepositive_labeled_data['matched_audio_filename'] = ''
truepositive_labeled_data['sound_start_second'] = 0
truepositive_labeled_data['sound_end_second'] = 0
for index, row in truepositive_labeled_data.iterrows():
    if (index % 10000 == 0):
        print(index)
    truepositive_labeled_data.loc[index, 'sound_start_second'] = floor(truepositive_labeled_data.loc[index, 't_min'])
    truepositive_labeled_data.loc[index, 'sound_end_second'] = ceil(truepositive_labeled_data.loc[index, 't_max'])
    project, site, year, month, filename = row['uri'].rsplit('/')[-5:]
    audio_filename = project + '_' + site + '_' + year + '_' + month + '_' + filename
    if audio_filename in all_audio_filenames:
        truepositive_labeled_data.loc[index, 'matched_audio_filename'] = audio_filename
    else:
        truepositive_labeled_data.loc[index, 'matched_audio_filename'] = 'No Matched Audio'

matched_truepositive_labeled_data = truepositive_labeled_data.loc[(~truepositive_labeled_data.matched_audio_filename.str.contains('No Matched Audio'))][['species_id','tod', 'matched_audio_filename','sound_start_second', 'sound_end_second']]
print("matched true positive data:", matched_truepositive_labeled_data.shape)

falsepositive_labeled_data['matched_audio_filename'] = ''
falsepositive_labeled_data['sound_start_second'] = 0
falsepositive_labeled_data['sound_end_second'] = 0
for index, row in falsepositive_labeled_data.iterrows():
    if (index % 10000 == 0):
        print(index)
    falsepositive_labeled_data.loc[index, 'sound_start_second'] = floor(falsepositive_labeled_data.loc[index, 't_min'])
    falsepositive_labeled_data.loc[index, 'sound_end_second'] = ceil(falsepositive_labeled_data.loc[index, 't_max'])
    project, site, year, month, filename = row['uri'].rsplit('/')[-5:]
    audio_filename = project + '_' + site + '_' + year + '_' + month + '_' + filename
    if audio_filename in all_audio_filenames:
        falsepositive_labeled_data.loc[index, 'matched_audio_filename'] = audio_filename
    else:
        falsepositive_labeled_data.loc[index, 'matched_audio_filename'] = 'No Matched Audio'

matched_falsepositive_labeled_data = falsepositive_labeled_data.loc[(~falsepositive_labeled_data.matched_audio_filename.str.contains('No Matched Audio'))]
print("matched false positive data:", matched_falsepositive_labeled_data.shape)


#%% Step 3: extract mel-spectrograms from detections

def graph_spectrogram(spectrogram_second_length, matched_labeled_data, audio_filename, label):
    sound_info, sampling_rate = librosa.load(audio_dir + audio_filename, sr = 48000)
    audio_length_second = int(len(sound_info) / sampling_rate)
    matched_labeled_data_same_audio = matched_labeled_data.loc[matched_labeled_data.matched_audio_filename == audio_filename].reset_index(drop=True)
    dpi = 100 ## resolution
    for j in range(len(matched_labeled_data_same_audio)):
        species_id = matched_labeled_data_same_audio.loc[j, 'species_id']
        sound_start_second = matched_labeled_data_same_audio.loc[j, 'sound_start_second']
        sound_end_second = matched_labeled_data_same_audio.loc[j, 'sound_end_second']
        tod = str(matched_labeled_data_same_audio.loc[j, 'tod'])[:4]  
        if (sound_start_second + spectrogram_second_length <= audio_length_second):
            pyplot.figure(num=None, figsize=(300 / dpi, 300 / dpi), dpi = dpi)
            pyplot.subplot(222)
            ax = pyplot.axes()
            ax.set_axis_off()
            S = librosa.feature.melspectrogram(y = sound_info[sampling_rate * sound_start_second: sampling_rate * (sound_start_second + spectrogram_second_length)], sr = sampling_rate)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
            pyplot.savefig(output_mel_spectrogram_dir + label + '/' + audio_filename.split('.')[0] + '_' + str(sound_start_second) + '_' + str(sound_start_second + spectrogram_second_length) + '_' + str(species_id) + '_' + tod + '_' + label +  '.png', bbox_inches='tight', transparent=True, pad_inches=0.0)
            pyplot.close()
        elif (sound_start_second + spectrogram_second_length > audio_length_second):
            pyplot.figure(num=None, figsize=(300 / dpi, 300 / dpi), dpi = dpi)
            pyplot.subplot(222)
            ax = pyplot.axes()
            ax.set_axis_off()
            S = librosa.feature.melspectrogram(y = sound_info[sampling_rate * (sound_end_second - spectrogram_second_length): sampling_rate * sound_end_second], sr = sampling_rate)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
            pyplot.savefig(output_mel_spectrogram_dir + label + '/' + audio_filename.split('.')[0] + '_' + str(sound_end_second - spectrogram_second_length) + '_' + str(sound_end_second) + '_' + str(species_id) + '_' + tod + '_' + label +  '.png', bbox_inches='tight', transparent=True, pad_inches=0.0)
            pyplot.close()   

    gc.collect()   ## clear-up memory


### extract 2-second true_positive mel-spectrograms
def generate_spectrogram_tp(i):
    audio_filename = matched_audio_filename_tp[i]
    try:
        return graph_spectrogram(2, matched_truepositive_labeled_data_unique, audio_filename, 'tp')
    except:
        pass

num_cores = multiprocessing.cpu_count()
matched_audio_filename_tp = list(set(matched_truepositive_labeled_data_unique.matched_audio_filename))
spectrogram_tp = Parallel(n_jobs=num_cores)(delayed(generate_spectrogram_tp)(i) for i in range(len(matched_audio_filename_tp)))

### extract 2-second false_positive mel-spectrograms
def generate_spectrogram_fp(i):
    audio_filename = matched_audio_filename_fp[i]
    try:
        return graph_spectrogram(2, matched_falsepositive_labeled_data_unique, audio_filename, 'fp')
    except:
        pass

num_cores = multiprocessing.cpu_count()
matched_audio_filename_fp = list(set(matched_falsepositive_labeled_data_unique.matched_audio_filename))
spectrogram_fp = Parallel(n_jobs=num_cores)(delayed(generate_spectrogram_fp)(i) for i in range(len(matched_audio_filename_fp)))
