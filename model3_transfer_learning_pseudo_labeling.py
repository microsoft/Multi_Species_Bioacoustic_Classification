#
# step2_model_transfer_learning_pseudo_labeling.py
#
# Fine tune a pre-trained ResNet50 with pseudo labeling. Both True Positive and False Positive labeled data are used.
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
import keras.backend as K 
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

directory_filenames_fp = glob.glob(mel_spectrogram_dir + '/*/*/*_fp.png')
directory_filenames_fp.sort()
print("Total number of False Positive mel-spectrograms (with duplicates):", len(directory_filenames_fp))

insects_species_id = ['16769', '16770', '16772'] ## three insects species_id to remove

filenames_label_tp = list(set([(directory_filenames_tp[i].split('\\')[-1], directory_filenames_tp[i].split('_')[-3]) for i in range(len(directory_filenames_tp)) if directory_filenames_tp[i].split('_')[-3] not in insects_species_id]))
filenames_label_tp.sort()
print("Total number of True Positive mel-spectrograms (unique):", len(filenames_label_tp))

filenames_label_fp = list(set([(directory_filenames_fp[i].split('\\')[-1], directory_filenames_fp[i].split('_')[-3]) for i in range(len(directory_filenames_fp)) if directory_filenames_fp[i].split('_')[-3] not in insects_species_id]))
filenames_label_fp.sort()
print("Total number of False Positive mel-spectrograms (unique):", len(filenames_label_fp))

filenames_label_tp_sample, filenames_label_tp_test = train_test_split(filenames_label_tp, train_size = 0.7, random_state = 42)
filenames_label_fp_sample, filenames_label_fp_test = train_test_split(filenames_label_fp, train_size = 0.2, random_state = 42)

filenames_tp_sample = [x[0] for x in filenames_label_tp_sample]
labels_tp_sample = [x[1] for x in filenames_label_tp_sample]  ## extract species_id as label

directory_filenames_tp_unique_sample = []
for filename in filenames_tp_sample:
    directory_filenames_tp_unique_sample.append([x for x in directory_filenames_tp if filename in x][0])
print("Total number of True Positive mel-spectrograms (unique) for training/validation:", len(directory_filenames_tp_unique_sample))

filenames_fp_sample = [x[0] for x in filenames_label_fp_sample]
labels_fp_sample = [x[1] for x in filenames_label_fp_sample]  ## extract species_id as label

directory_filenames_fp_unique_sample = []
for filename in filenames_fp_sample:
    directory_filenames_fp_unique_sample.append([x for x in directory_filenames_fp if filename in x][0])
print("Total number of False Positive mel-spectrograms (unique) for training/validation:", len(directory_filenames_fp_unique_sample))

tp_cnt_by_species_sample = Counter(labels_tp_sample)
tp_cnt_by_species_sample_dataframe = pd.DataFrame.from_dict(tp_cnt_by_species_sample, orient='index').reset_index()
tp_cnt_by_species_sample_dataframe = tp_cnt_by_species_sample_dataframe.rename(columns={'index':'species_id', 0:'TP_count'})
tp_cnt_by_species_sample_dataframe.sort_values(by=['TP_count'],ascending=False)

fp_cnt_by_species_sample = Counter(labels_fp_sample)
fp_cnt_by_species_sample_dataframe = pd.DataFrame.from_dict(fp_cnt_by_species_sample, orient='index').reset_index()
fp_cnt_by_species_sample_dataframe = fp_cnt_by_species_sample_dataframe.rename(columns={'index':'species_id', 0:'FP_count'})
fp_cnt_by_species_sample_dataframe.sort_values(by=['FP_count'],ascending=False)

lb_tp = LabelBinarizer()
labels_tp_sample = lb_tp.fit_transform(labels_tp_sample)
labels_tp_sample = labels_tp_sample.astype('float')
labels_tp_sample[labels_tp_sample == 0] = np.nan  

number_of_tp_sample_classes = len(lb_tp.classes_)
print("Number of TP Species: ", number_of_tp_sample_classes)

species_id_class_dict_tp = dict()
for (class_label, species_id) in enumerate(lb_tp.classes_):
    species_id_class_dict_tp[species_id] = class_label
    print("species_id ", species_id, ": ", class_label)

lb_fp = LabelBinarizer()
labels_fp_sample = lb_fp.fit_transform(labels_fp_sample)
labels_fp_sample = labels_fp_sample.astype('float')
labels_fp_sample[labels_fp_sample == 0] = np.nan  
labels_fp_sample[labels_fp_sample == 1] = 0  

number_of_fp_sample_classes = len(lb_fp.classes_)
print("Number of FP Species: ", number_of_fp_sample_classes)

species_id_class_dict_fp = dict()
for (class_label, species_id) in enumerate(lb_fp.classes_):
    species_id_class_dict_fp[species_id] = class_label
    print("species_id ", species_id, ": ", class_label)

species_id_class_dict = species_id_class_dict_tp

ncol, nrow = 224, 224
## tp
mel_spectrograms_tp_sample = []
for i in range(len(directory_filenames_tp_unique_sample)):
    if(i % 10000 == 0):
        print(i)
    img = cv2.imread(directory_filenames_tp_unique_sample[i])  
    img = cv2.resize(img, (ncol, nrow)) / 255.0
    mel_spectrograms_tp_sample.append(img)
mel_spectrograms_tp_sample = np.asarray(mel_spectrograms_tp_sample)

## fp
mel_spectrograms_fp_sample = []
for i in range(len(directory_filenames_fp_unique_sample)):
    if(i % 10000 == 0):
        print(i)
    img = cv2.imread(directory_filenames_fp_unique_sample[i])  
    img = cv2.resize(img, (ncol, nrow)) / 255.0
    mel_spectrograms_fp_sample.append(img)
mel_spectrograms_fp_sample = np.asarray(mel_spectrograms_fp_sample)

X_train_tp, X_validation_tp, y_train_tp, y_validation_tp, directory_filenames_tp_train, directory_filenames_tp_validation = train_test_split(mel_spectrograms_tp_sample, labels_tp_sample, directory_filenames_tp_unique_sample, test_size = 0.3, random_state = 42)
X_train_fp, X_validation_fp, y_train_fp, y_validation_fp, directory_filenames_fp_train, directory_filenames_fp_validation = train_test_split(mel_spectrograms_fp_sample, labels_fp_sample, directory_filenames_fp_unique_sample, test_size = 0.3, random_state = 42)

X_train = np.concatenate([X_train_tp, X_train_fp])
del X_train_tp
del X_train_fp

y_train = np.concatenate([y_train_tp, y_train_fp])
del y_train_tp
del y_train_fp

X_validation = np.concatenate([X_validation_tp, X_validation_fp])
del X_validation_tp
del X_validation_fp

y_validation = np.concatenate([y_validation_tp, y_validation_fp])
del y_validation_tp
del y_validation_fp

print(X_train.shape)   
print(y_train.shape)
print(X_validation.shape)   
print(y_validation.shape)


#%% Step 2: build model

def missing_values_cross_entropy_loss(y_true, y_pred):
    #add a small epsilon value to prevent computing logarithm of 0 (consider y_hat == 0.0 or y_hat == 1.0).
    epsilon = 1.0e-30

    # Temporarily replace missing values with zeroes, storing the missing values mask for later.
    y_true_not_nan_mask = tf.logical_not(tf.math.is_nan(y_true))
    y_true_nan_replaced = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)

    # Cross entropy, but split into multiple lines for readability:
    # y * log(y_hat)
    positive_predictions_cross_entropy = y_true_nan_replaced * tf.math.log(y_pred + epsilon)
    # (1 - y) * log(1 - y_hat)
    negative_predictions_cross_entropy = (1.0 - y_true_nan_replaced) * tf.math.log(1.0 - y_pred + epsilon)
    # c(y, y_hat) = -(y * log(y_hat) + (1 - y) * log(1 - y_hat))
    cross_entropy_loss = -(positive_predictions_cross_entropy + negative_predictions_cross_entropy)

    # Use the missing values mask for replacing loss values in places in which the label was missing with zeroes.
    # (y_true_not_nan_mask is a boolean which when casted to float will take values of 0.0 or 1.0)
    cross_entropy_loss_discarded_nan_labels = cross_entropy_loss * tf.cast(y_true_not_nan_mask, tf.float32)

    mean_loss_per_row = K.mean(cross_entropy_loss_discarded_nan_labels, axis=1)
    mean_loss = K.mean(mean_loss_per_row)

    return mean_loss

number_of_classes = y_train.shape[1]

#Load the ResNet50 model
from keras.applications import ResNet50
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
model.compile(loss = missing_values_cross_entropy_loss, optimizer = optimizer, metrics = ['accuracy'])
# Show a summary of the model. Check the number of trainable parameters
model.summary()

# semi-supervised learning: self-learning to update with pseudo labels
threshold_score_high, threshold_score_low = 0.9, 0.1
for i in range(2):
    model_history = model.fit(X_train, y_train, batch_size=16, epochs=5, verbose=1, validation_data=(X_validation, y_validation))
    y_train_pred_prob = model.predict(X_train)
    y_train_pseudo_label = np.where(y_train_pred_prob >= threshold_score_high, 1, y_train_pred_prob)
    y_train_pseudo_label = np.where(y_train_pseudo_label  <= threshold_score_low, 0, y_train_pseudo_label)
    y_train_pseudo_label = np.where((y_train_pseudo_label > threshold_score_low) & (y_train_pseudo_label < threshold_score_high), np.nan, y_train_pseudo_label)
    y_train = np.maximum(y_train, y_train_pseudo_label)
    
    plt.plot(model_history.history['acc'])
    plt.plot(model_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()

# Save the weights
model.save_weights(model_dir + 'resnet_weights_pseudo_label.h5')

# Save the model architecture
with open(model_dir + 'resnet_architecture_pseudo_label_TP_FP_0.01_0.99.json', 'w') as f:
    f.write(model.to_json())


#%% Step 3: predict on the test set

filenames_tp_test = [x[0] for x in filenames_label_tp_test]
labels_tp_test = [x[1] for x in filenames_label_tp_test]  ## extract species_id as label

## TP
directory_filenames_tp_unique_test = []
for filename in filenames_tp_test:
    directory_filenames_tp_unique_test.append([x for x in directory_filenames_tp if filename in x][0])
print("Total number of True Positive mel-spectrograms (unique) for testing:", len(directory_filenames_tp_unique_test))

output_tp_test = pd.DataFrame()
output_tp_test['filename'] = directory_filenames_tp_unique_test
output_tp_test['tp_fp'] = 'tp'
output_tp_test['species_id'] = ''
output_tp_test['predicted_probability'] = 0.0

for i in range(len(directory_filenames_tp_unique_test)):
    if(i % 10000 == 0):
        print(i)
    temp = []
    img = cv2.imread(directory_filenames_tp_unique_test[i]) 
    img = cv2.resize(img, (ncol, nrow)) / 255.0
    temp.append(img)   
    yhat_prob = model.predict(np.asarray(temp))
    
    species_id = output_tp_test.loc[i, 'filename'].split('_')[-3]
    predicted_probability = yhat_prob[0][species_id_class_dict[species_id]]
    output_tp_test.at[i, 'species_id'] = species_id
    output_tp_test.at[i, 'predicted_probability'] = predicted_probability 

filenames_fp_test = [x[0] for x in filenames_label_fp_test]
labels_fp_test = [x[1] for x in filenames_label_fp_test]  ## extract species_id as label

## FP
directory_filenames_fp_unique_test = []
for filename in filenames_fp_test:
    directory_filenames_fp_unique_test.append([x for x in directory_filenames_fp if filename in x][0])
print("Total number of False Positive mel-spectrograms (unique) for testing:", len(directory_filenames_fp_unique_test))

output_fp_test = pd.DataFrame()
output_fp_test['filename'] = directory_filenames_fp_unique_test
output_fp_test['tp_fp'] = 'fp'
output_fp_test['species_id'] = ''
output_fp_test['predicted_probability'] = 0.0

for i in range(len(directory_filenames_fp_unique_test)):
    if(i % 10000 == 0):
        print(i)
    temp = []
    img = cv2.imread(directory_filenames_fp_unique_test[i]) 
    img = cv2.resize(img, (ncol, nrow)) / 255.0
    temp.append(img)   
    yhat_prob = model.predict(np.asarray(temp))
    
    species_id = output_fp_test.loc[i, 'filename'].split('_')[-3]
    predicted_probability = yhat_prob[0][species_id_class_dict[species_id]]
    output_fp_test.at[i, 'species_id'] = species_id
    output_fp_test.at[i, 'predicted_probability'] = predicted_probability 

plt.hist(output_tp_test['predicted_probability'])
plt.hist(output_fp_test['predicted_probability'])

classification_threshold = 0.5
prediction_tp = len(output_tp_test.loc[output_tp_test.predicted_probability >= classification_threshold])
prediction_fn = len(output_tp_test.loc[output_tp_test.predicted_probability < classification_threshold])
prediction_fp = len(output_fp_test.loc[output_fp_test.predicted_probability >= classification_threshold])
precision = prediction_tp / (prediction_tp + prediction_fp) 
recall = prediction_tp / (prediction_tp + prediction_fn) 
beta = 0.5
f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
print('precision: ', precision)
print('recall: ', recall)

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
    positive_cnt = len(output_tp_test.loc[output_tp_test.species_id == species_id])
    negative_cnt = len(output_fp_test.loc[output_fp_test.species_id == species_id])

    tp = len(output_tp_test.loc[(output_tp_test.species_id == species_id) & (output_tp_test.predicted_probability >= classification_threshold)])
    fn = len(output_tp_test.loc[(output_tp_test.species_id == species_id) & (output_tp_test.predicted_probability < classification_threshold)])
    fp = len(output_fp_test.loc[(output_fp_test.species_id == species_id) & (output_fp_test.predicted_probability >= classification_threshold)])
    tn = len(output_fp_test.loc[(output_fp_test.species_id == species_id) & (output_fp_test.predicted_probability < classification_threshold)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    beta = 0.5
    f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    tpr = tp / positive_cnt
    tnr = tn / negative_cnt

    species_output_tp = output_tp_test.loc[output_tp_test.species_id == species_id]
    species_output_fp = output_fp_test.loc[output_fp_test.species_id == species_id]
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

