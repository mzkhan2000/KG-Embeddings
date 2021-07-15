#!/usr/bin/env python
# coding: utf-8

# In[6]:


### Load Google’s Word2Vec Embedding


# In[3]:


pip install --upgrade gensim


# In[1]:


# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec


# In[2]:


import numpy    


# In[3]:


import nltk
nltk.download('punkt')


# In[4]:


import re, string # using to remove regular expression, special characters in the csv files/ -Monir


# In[8]:


from gensim.models import KeyedVectors


# In[9]:


filename = 'GoogleNews-vectors-negative300.bin'


# In[12]:


model = KeyedVectors.load_word2vec_format(filename, binary=True)


# In[28]:


gravity20 = model.most_similar('gravity', topn=20)
print(gravity20)


# In[24]:


print(model.similarity('gravity', 'gravitation'))


# In[27]:


earthquakes10 = model.most_similar('earthquakes', topn=20)
print(earthquakes10)


# In[29]:


volcanology20 = model.most_similar('volcanology', topn=20)
print(volcanology20)


# In[30]:


seismics20 = model.most_similar('seismics', topn=20)
print(seismics20)


# In[33]:


marine20 = model.most_similar('marine', topn=20)
print(marine20)


# In[34]:


geophysics20 = model.most_similar('geophysics', topn=20)
print(geophysics20)


# In[35]:


palaeontology20 = model.most_similar('palaeontology', topn=20)
print(palaeontology20)


# In[ ]:


seismics20 = model.most_similar('seismics', topn=20)
print(seismics20)


# In[ ]:


seismics20 = model.most_similar('seismics', topn=20)
print(seismics20)


# In[2]:


conda list


# In[4]:


import gensim
from gensim.models import Word2Vec


# In[5]:


# Reads ‘Gravity_DBpedia.txt’ file
sample = open("Gravity_DBpedia.txt", "r")
s = sample.read()


# In[6]:


# Replaces escape character with space
f = s.replace("\n", " ")


# In[14]:


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


# In[15]:


f=word_drop(f)


# In[18]:


# Replaces escape character with space
f = s.replace("\n", " ")


# In[19]:


print(f)


# In[20]:


f=word_drop(f)


# In[21]:


print(f)


# In[22]:


# Replaces escape character with space
f = f.replace("\n", " ")


# In[23]:


print(f)


# In[26]:


data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
	temp = []
	
	# tokenize the sentence into words
	for j in word_tokenize(i):
		temp.append(j.lower())

	data.append(temp)


# In[55]:


# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 10000, window = 10)


# In[44]:


gravity5 = model1.wv.most_similar('gravity', topn=10)


# In[38]:


print(gravity5) # vector_size = 1000


# In[42]:


print(gravity5)  # vector_size = 100


# In[45]:


print(gravity5)  # vector_size = 200


# In[56]:


gravity5 = model1.wv.most_similar('gravity', topn=10)


# In[57]:


print(gravity5)  # vector_size = 10000 , window = 10


# In[39]:


# Print results
print("Cosine similarity between 'planets' " +
			"and 'gravitation' - CBOW : ",
	model1.wv.similarity('gravity', 'planets'))
	
#model1.similarity


# In[49]:


print(model1.wv.similarity('gravity', 'gravitation'))


# In[50]:


print(model1.wv.similarity('gravity', 'planets'))


# In[51]:


print(model1.wv.similarity('gravity', 'applications'))


# In[48]:


cosine_similarity = numpy.dot(model1['gravity'], model1['gravitation'])/(numpy.linalg.norm(model1['gravity'])* numpy.linalg.norm(model1['gravitation']))


# In[ ]:



    print("Cosine similarity between 'alice' " +
				"and 'gravitation' - CBOW : ",
	model1.similarity('alice', 'gravitation'))

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100,
											window = 5, sg = 1)

# Print results
print("Cosine similarity between 'alice' " +
		"and 'wonderland' - Skip Gram : ",
	model2.similarity('alice', 'wonderland'))
	
print("Cosine similarity between 'alice' " +
			"and 'machines' - Skip Gram : ",
	model2.similarity('alice', 'machines'))

