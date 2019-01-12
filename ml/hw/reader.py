import json
import math
import copy

# Character occurance
def character_occ(data, characters):
  for i in range(0, len(data)):
    charList = copy.deepcopy(characters)
    for char in data[i]:
      charList[char] += 1
    data[i] = charList
    if i % 500 == 0:
      print("Completed", i, "of", len(data))
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
  characters = dict()
  for line in data:
    for char in line['text']:
      if char not in characters:
        characters[char] = 0
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
  with open('train_str.json') as f:
    for line in f:
      data_str.append(json.loads(line))

  data_ans = []
  with open('train_ans.json') as f:
    for line in f:
      data_ans.append(json.loads(line))

  # Set of all languages
  characters = get_characters(data_str)
  languages = get_languages(data_ans)
  
  # Remove all extra json formatting and just get list of all strings
  data_str = listify(data_str, 'text')
  data_ans = listify(data_ans, 'classification')
  data_str = trimData(data_str)

  return data_str, data_ans, characters, languages # Bag of characters size is 5697. Add 1 for bias, 5698 inputs per line.
  
def main():
  scores = [0] * 2
  scores[1] = .50
  with open("performance.txt", "w") as f:
    f.write("Model is expected to be correct roughly about " + str(scores[1]*100))
  
  # Bag of characters size is 5697. Add 1 for bias, 5698 inputs per line.
  data_str, data_labels, characters, languages = get_data()

  partialData = character_occ(data_str[:25000], copy.deepcopy(characters))

  

main()