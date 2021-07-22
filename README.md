# KG-Embeddings

```python

pip install rdflib
import rdflib
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

# to get all the ga-themes
ga_record_datapath = "/content/drive/My Drive/MonirResearchDatasets/surround-ga-records/ga-records.ttl"
gaThemes = gaThemesExtraction(ga_record_datapath)
print(gaThemes[:10])
print(len(gaThemes))

```
