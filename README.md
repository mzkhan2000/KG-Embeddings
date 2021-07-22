# Text-Embeddings

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
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import numpy
import multiprocessing
import re, string # using to remove regular expression, special characters in txt files
```

```python

# Read text file for embedding such as ‘Gravity_DBpedia.txt’ file
def input_text_file(file_path):   
   
   # Read text file
   sample = open(file_path, "r")
   s = sample.read()

   # Replaces escape character with space
   text = s.replace("\n", " ")
   return text
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
# Replaces escape character with space
f = s.replace("\n", " ")
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

# Create CBOW model
def create_embedding_model(data): 

  # Create CBOW Word2Ve model
  embedding_model = gensim.models.Word2Vec(data, min_count = 1, vector_size = 300, window = 10)

  return embedding_model

```

```python

def embedding_main:
    
    # Read text file for embedding such as ‘Gravity_DBpedia.txt’ file
    file_path = "/content/drive/My Drive/MonirResearchDatasets/Gravity_DBpedia.txt"
    
    text = input_text_file(file_path)
    preprocessed_text = word_drop(text)
    text_to_token = word_tokenize(preprocessed_text)
    embedding_model = create_embedding_model(text_to_token)

```
# KG-Embeddings

```python

def embedding_word_clusters(model, list_of_ga_themes, cluster_size):
  keys = list_of_ga_themes
  GoogleNews_model = model
  n = cluster_size

  embedding_clusters = []
  word_clusters = []
  for word in keys:
      embeddings = []
      words = []
      for similar_word, _ in GoogleNews_model.most_similar(word, topn=n):
          words.append(similar_word)
          embeddings.append(GoogleNews_model[similar_word])
      embedding_clusters.append(embeddings)
      word_clusters.append(words)

  return embedding_clusters, word_clusters
  
```


```python

Code here............

```

```python

# Script for creating a 2D scatter plot using Matplotlib library for data visualization in Python.
% matplotlib inline

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


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


