import tensorflow as tf
import numpy as np
import json
import math
import copy

from sklearn import preprocessing

import keras
import keras.optimizers
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import TensorBoard

# Character occurance
def character_occ(data, characters):
  for i in range(0, len(data)):
    charList = [0]*len(characters)
    for char in data[i]:
      if char in characters:
        charList[characters.index(char)] += 1
      else:
        charList[characters.index('?')] += 1
    data[i] = charList
  return data

# Makes a list of all possible characters from data.
def get_characters(data):
  characters = []
  for line in data:
    for char in line['text']:
      if char not in characters:
        characters.append(char)
  return characters

# Makes a unique set of all the languages in the data.
def get_languages(data):
  languages = dict()
  for line in data:
    if line['classification'] not in languages:
      languages[line['classification']] = len(languages)
  return languages

# Convert list of lists to numpy array of dimension len(data) x 140
def trimData(data):
  for i in range(0,len(data)):
    if len(data[i]) < 140:
      data[i] = data[i].ljust(140,'?')
    else:
      data[i] = data[i][:140]
  return data

# Makes a list of json objects into a list of strings.
def listify(data, key):
  newList = []
  for line in data:
    newList.append(line[key])
  return newList

def get_data():
  # Prepare the data into lists, parsing the jsons. Since we are using python3, it allows unicode.
  data_str = []
  with open("test_X_languages_homework.json.txt") as f:
    for line in f:
      data_str.append(json.loads(line))
      
  train_str = []
  with open("train_X_languages_homework.json.txt") as f:
    for line in f:
      train_str.append(json.loads(line))
      
  train_ans = []
  with open("train_y_languages_homework.json.txt") as f:
    for line in f:
      train_ans.append(json.loads(line))
      
  # Set of all languages
  characters = get_characters(train_str)
  languages = get_languages(train_ans)
  languages = dict([[v,k] for k,v in languages.items()])
  
  # Remove all extra json formatting and just get list of all strings
  data_str = listify(data_str, 'text')
  train_str = listify(train_str, 'text')
  
  # Standardize our data
  data_str = trimData(data_str)
  train_str = trimData(train_str)

  return data_str, train_str, characters, languages

  
def make_predictions():
  
  # Prepare the test data.
  test_data, train_str, characters, languages = get_data()
  
  test_data = character_occ(test_data, characters)

  # load json and create model
  with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
  model = model_from_json(loaded_model_json)
                          
  # load weights into new model
  model.load_weights("model.h5")
  
  # evaluate loaded model on test data
  model_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(loss='categorical_crossentropy', 
                      optimizer=model_optimizer,
                      metrics=['accuracy'])
  
  X = np.array(test_data, dtype=np.float32)
  del test_data, train_str, characters
  standard_scaler = preprocessing.StandardScaler().fit(X)
  X = standard_scaler.transform(X)
  
  classes = model.predict_classes(X, batch_size=10)
  classes = classes.tolist()
  with open('predictions.txt', 'w') as f:
    for item in classes:
        f.write("%s\n" % languages[item])
  
make_predictions()