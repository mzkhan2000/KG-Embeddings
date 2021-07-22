# KG-Embeddings

```python
# Python program to generate embedding (word vectors) using Word2Vec

# importing necessary modules for embedding
pip install --upgrade gensim

import nltk
nltk.download('punkt')

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec

import re, string # using to remove regular expression, special characters in txt files
```

```python

# define a function for removing unnecessary/special characters and return a lower case plain texts/ Monir
def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+ |www\.\S+', '', text)
    text = re.sub('<.*?>', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

```

```python

# define a function to tokenize the sentence into words/ Monir
def word_tokenize(text):
    data = []

    # iterate through each sentence in the file
    for i in sent_tokenize(f):
        temp = []

        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())

        data.append(temp)
    return data
    
```


```python

preprocessed_text = word_drop(text)
text_to_token = word_tokenize(preprocessed_text)

```

```python
Code here............
```

# GA-themes extraction

```python

pip install rdflib
import rdflib
```

```python

from rdflib import Graph

g = Graph()
g.parse("/content/drive/My Drive/MonirResearchDatasets/surround-ga-records/ga-records.ttl", format='turtle')

print(len(g))

```


```python

import collections
from collections import Counter
```

```python
# a funtion for ga-themes extraction from GA-rdf repository separate and return a list all the ga-themes - Monir
def gaThemesExtraction(ga_record):
  gaThemes = []
  with open(ga_record, 'rt') as f:
    data = f.readlines()
  for line in data:
      # check if line contains "ga-themes" sub-string
      if line.__contains__('ga-themes'):
          # split the line contains from "ga-themes" sub-string
          stringTemp = line.split("ga-themes/",1)[1]
          # further split the line contains from "ga-themes" sub-string to delimiter
          stringTemp = stringTemp.split('>')[0]
          gaThemes.append(stringTemp)
  #print(dataLog)
  #print(gaThemes[:9])
  #print(len(gaThemes))
  return gaThemes

```

```python
# a funtion imput a list of ga-themes and return a list of unique ga-themes and another list of duplicate gaThemes - 
def make_unique_gaThemes(list_all_ga_themes):
  # find a a list of unique ga-themes
  unique_gaThemes = []
  unique_gaThemes = list(dict.fromkeys(gaThemes))
  #print(len(unique_gaThemes))

  # a list of duplicate gaThemes
  duplicate_gaThemes = []
  duplicate_gaThemes = [item for item, count in collections.Counter(gaThemes).items() if count > 1]
  #print(len(duplicate_gaThemes))

  return unique_gaThemes, duplicate_gaThemes
  
```

```python
# to connect to Google drive with colab
from google.colab import drive
drive.mount('/content/drive')

```

```python

# to get all the ga-themes from ga-records.ttl file
ga_record_datapath = "/content/drive/My Drive/MonirResearchDatasets/surround-ga-records/ga-records.ttl"
gaThemes = gaThemesExtraction(ga_record_datapath)
print(gaThemes[:10])
print(len(gaThemes))

```

```python

# to get all the ga-themes from all1K file 
ga_record_datapath = "/content/drive/My Drive/MonirResearchDatasets/surround-ga-records/all1k.ttl.txt"
gaThemes = gaThemesExtraction(ga_record_datapath)
print(gaThemes[:10])
print(len(gaThemes))

```


```python

# to get all unique ga-themes
unique_gaThemes, duplicate_gaThemes = make_unique_gaThemes(gaThemes)
print(unique_gaThemes[:100])
#print(duplicate_gaThemes[:100])
print(len(unique_gaThemes))

```


