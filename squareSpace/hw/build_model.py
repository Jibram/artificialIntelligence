import tensorflow as tf
import numpy as np
import json
import math
import copy

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import keras
import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import TensorBoard

# Character occurance
def character_occ(data, characters):
  for i in range(0, len(data)):
    charList = [0]*len(characters)
    for char in data[i]:
      charList[characters.index(char)] += 1
    data[i] = charList
  return data

# Makes a unique set of all the languages in the data.
def get_languages(data):
  languages = dict()
  for line in data:
    if line['classification'] not in languages:
      languages[line['classification']] = len(languages)
  return languages

# Makes a list of all possible characters from data.
def get_characters(data):
  characters = []
  for line in data:
    for char in line['text']:
      if char not in characters:
        characters.append(char)
  return characters

# Makes a list of json objects into a list of strings.
def listify(data, key):
  newList = []
  for line in data:
    newList.append(line[key])
  return newList

# Convert list of lists to numpy array of dimension len(data) x 140
def trimData(data):
  for i in range(0,len(data)):
    if len(data[i]) < 140:
      data[i] = data[i].ljust(140,'?')
    else:
      data[i] = data[i][:140]
  return data

def get_data():
  # Prepare the data into lists, parsing the jsons. Since we are using python3, it allows unicode.
  data_str = []
  with open("train_X_languages_homework.json.txt") as f:
    for line in f:
      data_str.append(json.loads(line))

  data_ans = []
  with open("train_y_languages_homework.json.txt") as f:
    for line in f:
      data_ans.append(json.loads(line))

  # Set of all languages
  characters = get_characters(data_str)
  languages = get_languages(data_ans)
  
  # Remove all extra json formatting and just get list of all strings
  data_str = listify(data_str, 'text')
  data_ans = listify(data_ans, 'classification')
  
  # Standardize our data
  data_str = trimData(data_str)
  data_ans = [languages[langCode] for langCode in data_ans]

  return data_str, data_ans, characters, languages # Bag of characters size is 5697. Add 1 for bias, 5698 inputs per line.

  
def build_model():
  # data_str contains list of all strings
  # data_labels contains list of all language codes
  # characters is a map of all characters used in data_str
  # languages is a list with language codes. Useful to correspond with an index value
  data_str, data_labels, characters, languages = get_data()
  
  partialData = character_occ(data_str[:], characters)
  partialLabels = data_labels[:]
  input_size = len(partialData[0])
  del data_str, data_labels
  
  BATCH_SIZE = 512
  EPOCHS = 12
  
  NUM_SAMPLES = len(partialData)
  VOCAB_SIZE = len(characters)
  
  X = np.array(partialData, dtype=np.float32)
  Y = np.array(partialLabels, dtype=np.float32)
  del partialData, partialLabels
           
  
  standard_scaler = preprocessing.StandardScaler().fit(X)
  X = standard_scaler.transform(X)
  
  Y = keras.utils.to_categorical(Y, num_classes=len(languages))
 
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
  del X, Y
  
  model = Sequential()
  model.add(Dense(500, kernel_initializer="glorot_uniform", activation="sigmoid"))
  model.add(Dropout(0.5))
  model.add(Dense(300, kernel_initializer="glorot_uniform", activation="sigmoid"))
  model.add(Dropout(0.5))
  model.add(Dense(100, kernel_initializer="glorot_uniform", activation="sigmoid"))
  model.add(Dropout(0.5))
  model.add(Dense(len(languages), kernel_initializer="glorot_uniform", activation="softmax"))
  model_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(loss='categorical_crossentropy', 
                optimizer=model_optimizer,
                metrics=['accuracy'])
  
  tensorboard = TensorBoard(log_dir="run")
  
  history = model.fit(X_train,Y_train,
                     epochs=EPOCHS,
                     validation_split=0.1,
                     batch_size=BATCH_SIZE,
                     callbacks=[tensorboard],
                     verbose=2)
  
  scores = model.evaluate(X_test, Y_test, verbose=1)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  
  # 1 This is the build model script
  # 3 Serialize the Model
  
  # serialize model to JSON
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model.h5")
  
  # 5 Save the performance information
  with open("performance.txt", "w") as f:
    f.write("Model is expected to be correct roughly about " + str(scores[1]*100) + "%. ")
  with open("performance.txt", "a+") as f:
    f.write("Uses Dropout after each dense layer and an adam optimizer. 80/20")
    f.write("train/test ratio with sigmoid activations. Loss function is")
    f.write("categorical crossentropy. 12 Epochs of Batch Size 512.")
  
build_model()