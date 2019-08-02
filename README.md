# Wikipedia2Image
Student project for the lecture [Deep Vision](https://hci.iwr.uni-heidelberg.de/content/deep-vision) summer term 2019.

## Problem Description


## Data Set

For obtaining the data set please change to the data/ directory.
Within the data/ directory we provide the json file contianing the entities we crawled from wikipedia.


## How to run the code
#### Get pretrained models for InferSent and GloVe
```
!wget https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
!wget http://nlp.stanford.edu/data/glove.840B.300d.zip && unzip glove.840B.300d.zip && rm glove.840B.300d.zip
```

### Requirements

See requirements.txt

## Evaluation

For the evaluation we provide a [Jupyter Notebook](Evaluator/Evaluation.ipynb) and pretrained weights within the Evaluator folder.
For more details, have a look at the notebook.