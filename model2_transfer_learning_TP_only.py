#
# step2_model_transfer_learning_TP_only.py
#
# Fine tune a pre-trained ResNet50. Only True Positive labeled data is used.
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

#%% Imports

import os
import glob
import cv2
import numpy as np
from collections import Counter

import csv
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras_preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from keras.models import  Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D, BatchNormalization, Concatenate
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score


#%% Path configuration

current_dir = "./multispecies_bioacoustics/"

data_dir = current_dir + "Data/"
labeled_data_dir = data_dir + 'Labeled_Data/'
output_dir = current_dir + "Output/"
mel_spectrogram_dir = data_dir + "Extracted_Mel_Spectrogram/"

model_dir = current_dir + "Model/"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#%% Step 1: import data and train/validation/test split

directory_filenames_tp = glob.glob(mel_spectrogram_dir + '/*/*/*_tp.png')
directory_filenames_tp.sort()
print("Total number of True Positive mel-spectrograms (with duplicates):", len(directory_filenames_tp))

insects_species_id = ['16769', '16770', '16772'] ## three insects species_id to remove
filenames_label_tp = list(set([(directory_filenames_tp[i].split('\\')[-1], directory_filenames_tp[i].split('_')[-3]) for i in range(len(directory_filenames_tp)) if directory_filenames_tp[i].split('_')[-3] not in insects_species_id]))
filenames_label_tp.sort()
print("Total number of True Positive mel-spectrograms (unique):", len(filenames_label_tp))

filenames_tp = [x[0] for x in filenames_label_tp]
labels_tp = [x[1] for x in filenames_label_tp]  ## extract species_id as label

directory_filenames_tp_unique = []
for filename in filenames_tp:
    directory_filenames_tp_unique.append([x for x in directory_filenames_tp if filename in x][0])
print("Total number of True Positive mel-spectrograms (unique):", len(directory_filenames_tp_unique))

tp_cnt_by_species = Counter(labels_tp)
tp_cnt_by_species_dataframe = pd.DataFrame.from_dict(tp_cnt_by_species, orient='index').reset_index()
tp_cnt_by_species_dataframe = tp_cnt_by_species_dataframe.rename(columns={'index':'species_id', 0:'TP_count'})
tp_cnt_by_species_dataframe.sort_values(by=['TP_count'],ascending=False)

lb = LabelBinarizer()
labels_tp = lb.fit_transform(labels_tp)
number_of_classes = len(lb.classes_)
print("Number of Species: ", number_of_classes)

species_id_class_dict = dict()
for (class_label, species_id) in enumerate(lb.classes_):
    species_id_class_dict[species_id] = class_label
    print("species_id ", species_id, ": ", class_label)

mel_spectrograms_tp = []
ncol, nrow = 224, 224

## tp
for i in range(len(directory_filenames_tp_unique)):
    if(i % 10000 == 0):
        print(i)
    img = cv2.imread(directory_filenames_tp_unique[i])  
    img = cv2.resize(img, (ncol, nrow)) / 255.0
    mel_spectrograms_tp.append(img)
mel_spectrograms_tp = np.asarray(mel_spectrograms_tp)

X_train_validation, X_test, y_train_validation, y_test, directory_filenames_tp_train_validation, directory_filenames_tp_test = train_test_split(mel_spectrograms_tp, labels_tp, directory_filenames_tp_unique, test_size = 0.3, random_state = 42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size = 0.3, random_state = 42)
print(X_train.shape)   
print(y_train.shape)
print(X_validation.shape)   
print(y_validation.shape)
print(X_test.shape)
print(y_test.shape)


#%% Step 2: build model

from keras.applications import ResNet50
#Load the ResNet50 model
ResNet50_conv = ResNet50(weights='imagenet', include_top=False, input_shape=(nrow, ncol, 3))

for layer in ResNet50_conv.layers:
    layer.trainable = True

# Check the trainable status of the individual layers
for layer in ResNet50_conv.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()
# Add the vgg convolutional base model
model.add(ResNet50_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(number_of_classes, activation='sigmoid'))
 # Compile the model
optimizer = optimizers.adam(lr=0.0001, decay=1e-7)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
# Show a summary of the model. Check the number of trainable parameters
model.summary()

model_history = model.fit(X_train, y_train, batch_size=16, epochs=5, verbose=1, validation_data=(X_validation, y_validation))
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

# Save the weights
model.save_weights(model_dir + 'resnet_weights.h5')

# Save the model architecture
with open(model_dir + 'resnet_architecture.json', 'w') as f:
    f.write(model.to_json())


#%% Step 3: predict on the test set

yhat_probs_tp = model.predict(X_test)
output_tp = pd.DataFrame()
output_tp['filename'] = directory_filenames_tp_test
output_tp['tp_fp'] = 'tp'
output_tp['species_id'] = ''
output_tp['predicted_probability'] = 0.0
for i in range(len(output_tp)):
    species_id = output_tp.loc[i, 'filename'].split('_')[-3]
    predicted_probability = yhat_probs_tp[i][species_id_class_dict[species_id]]
    output_tp.at[i, 'species_id'] = species_id
    output_tp.at[i, 'predicted_probability'] = predicted_probability

## free-up memory
del mel_spectrograms_tp
del labels_tp

directory_filenames_fp = glob.glob(mel_spectrogram_dir + '/*/*/*_fp.png')
directory_filenames_fp.sort()
print("Total number of False Positive mel-spectrograms (with duplicates):", len(directory_filenames_fp))

filenames_label_fp = list(set([(directory_filenames_fp[i].split('\\')[-1], directory_filenames_fp[i].split('_')[-3]) for i in range(len(directory_filenames_fp)) if directory_filenames_fp[i].split('_')[-3] not in insects_species_id]))
print("Total number of False Positive mel-spectrograms (unique):", len(filenames_label_fp))

filenames_fp = [x[0] for x in filenames_label_fp]
labels_fp = [x[1] for x in filenames_label_fp]  ## extract species_id as label

directory_filenames_fp_unique = []
for filename in filenames_fp:
    directory_filenames_fp_unique.append([x for x in directory_filenames_fp if filename in x][0])

print("Total number of False Positive mel-spectrograms(unique):", len(directory_filenames_fp_unique))

ncol, nrow = 224, 224
output_fp = pd.DataFrame()
output_fp['filename'] = directory_filenames_fp_unique
output_fp['tp_fp'] = 'fp'
output_fp['species_id'] = ''
output_fp['predicted_probability'] = 0.0

for i in range(len(directory_filenames_fp_unique)):
    if(i % 10000 == 0):
        print(i)
    temp = []
    img = cv2.imread(directory_filenames_fp_unique[i]) 
    img = cv2.resize(img, (ncol, nrow)) / 255.0
    temp.append(img)   
    yhat_prob = model.predict(np.asarray(temp))
    
    species_id = output_fp.loc[i, 'filename'].split('_')[-3]
    predicted_probability = yhat_prob[0][species_id_class_dict[species_id]]
    output_fp.at[i, 'species_id'] = species_id
    output_fp.at[i, 'predicted_probability'] = predicted_probability  


plt.hist(output_tp['predicted_probability'])
plt.hist(output_tp['predicted_probability'])

classification_threshold = 0.5
prediction_tp = len(output_tp.loc[output_tp.predicted_probability >= classification_threshold])
prediction_fn = len(output_tp.loc[output_tp.predicted_probability < classification_threshold])
prediction_fp = len(output_fp.loc[output_fp.predicted_probability >= classification_threshold])
precision = prediction_tp / (prediction_tp + prediction_fp) 
recall = prediction_tp / (prediction_tp + prediction_fn) 
beta = 0.5
f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
print('precision: ', precision)
print('recall: ', recall)
print('F_0.5 Score: ', f_score)

evaluation_metrics_by_species = pd.DataFrame()
evaluation_metrics_by_species['species_id'] = list(species_id_class_dict)
evaluation_metrics_by_species['species_name'] = ''
evaluation_metrics_by_species['positive_cnt'] = 0
evaluation_metrics_by_species['negative_cnt'] = 0
evaluation_metrics_by_species['TP_cnt'] = 0
evaluation_metrics_by_species['FP_cnt'] = 0
evaluation_metrics_by_species['TN_cnt'] = 0
evaluation_metrics_by_species['FN_cnt'] = 0
evaluation_metrics_by_species['Precision'] = 0.0
evaluation_metrics_by_species['Recall'] = 0.0
evaluation_metrics_by_species['Accuracy'] = 0.0
evaluation_metrics_by_species['F_score'] = 0.0   ## F_0.5_score
evaluation_metrics_by_species['TPR'] = 0.0   
evaluation_metrics_by_species['TNR'] = 0.0 
evaluation_metrics_by_species['AUC'] = 0.0 

for i in range(len(evaluation_metrics_by_species)):
    species_id = evaluation_metrics_by_species.loc[i, 'species_id']
    species_name = species_id_name_dict[int(species_id)]
    positive_cnt = len(output_tp.loc[output_tp.species_id == species_id])
    negative_cnt = len(output_fp.loc[output_fp.species_id == species_id])

    tp = len(output_tp.loc[(output_tp.species_id == species_id) & (output_tp.predicted_probability >= classification_threshold)])
    fn = len(output_tp.loc[(output_tp.species_id == species_id) & (output_tp.predicted_probability < classification_threshold)])
    fp = len(output_fp.loc[(output_fp.species_id == species_id) & (output_fp.predicted_probability >= classification_threshold)])
    tn = len(output_fp.loc[(output_fp.species_id == species_id) & (output_fp.predicted_probability < classification_threshold)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    beta = 0.5
    f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    tpr = tp / positive_cnt
    tnr = tn / negative_cnt

    species_output_tp = output_tp.loc[output_tp.species_id == species_id]
    species_output_fp = output_fp.loc[output_fp.species_id == species_id]
    y_true = [1] * len(species_output_tp) + [0] * len(species_output_fp)
    y_scores = species_output_tp['predicted_probability'].tolist() + species_output_fp['predicted_probability'].tolist()
    
    evaluation_metrics_by_species.at[i, 'species_name'] = species_name
    evaluation_metrics_by_species.at[i, 'positive_cnt'] = positive_cnt
    evaluation_metrics_by_species.at[i, 'negative_cnt'] = negative_cnt
    evaluation_metrics_by_species.at[i, 'TP_cnt'] = tp
    evaluation_metrics_by_species.at[i, 'FP_cnt'] = fp
    evaluation_metrics_by_species.at[i, 'TN_cnt'] = tn
    evaluation_metrics_by_species.at[i, 'FN_cnt'] = fn
    evaluation_metrics_by_species.at[i, 'Precision'] = precision
    evaluation_metrics_by_species.at[i, 'Recall'] = recall  
    evaluation_metrics_by_species.at[i, 'Accuracy'] = accuracy 
    evaluation_metrics_by_species.at[i, 'F_score'] = f_score
    evaluation_metrics_by_species.at[i, 'TPR'] = tpr    
    evaluation_metrics_by_species.at[i, 'TNR'] = tnr
    evaluation_metrics_by_species.at[i, 'AUC'] = roc_auc_score(y_true, y_scores)
    
print(evaluation_metrics_by_species)
