# MEL: Mannheim Entity Linking
MEL is a Python library whose goal is to provide an efficient and easy to use end-to-end Entity Linking system.
Entity Linking is the task of linking mentions in free text to entities in a Knowledge Base (in our case Wikipedia).
For example "Washington" can refer to *https://en.wikipedia.org/wiki/George_Washington* or 
*https://en.wikipedia.org/wiki/Washington,_D.C.* or even *https://en.wikipedia.org/wiki/Federal_government_of_the_United_States*.

MEL is comprised of three main components:
1. Mention detection using [spacy](https://spacy.io/).
2. Candidate generation based on [nel](https://github.com/wikilinks/nel).
3. Entity linking using an implementation of the approach described in [Yamada et al](https://github.com/studio-ousia/ntee).

By leveraging the best methods for each component, MEL is able to achieve close to state-of-the-art performance. An easy to
setup flask server is also included.

# Dependencies
* Python 3 with Numpy
* PyTorch
* Spacy
* Flask


# Setup 

1. Clone this repo.
2. We recommend creating a virtual enviroment for this project using conda or pipenv.
3. Install dependencies by running ```pip install -r requirements.txt```.
4. Install spacy model with ```python -m spacy download en```.
4. To use MEL, one needs several dicts that are stored as memory mapped files. These are hosted [here](mmap file link), 
we also provide a pre-trained model [here](conll model file link) trained on [CONLL](conll data link here).
Downloading these files along with setting up of the project's data structure can be done using a shell script:
```
chmod +x bin/setup.sh
bin/setup.sh
```
**Note**: This will download ~4G of data.

# Performance

We compare against the popular [TagMe](https://tagme.d4science.org/tagme/) system and report F1 scores on the combined
mention detection and entity linking task. For mention detection, any predicted mentions with over 80% overlap with 
a gold mention is considered a match. TagMe allows to filter Entity Linking using a threshold parameter, here we show
results for three different values for a fair comparison. Here we show overall **F1 score / linking accuracy**.

| Data Set  |        MEL      | TagMe - Threshold 0.1 | TagMe (Threshold 0.3) | TagMe (Threshold 0.5) |
|-----------|:---------------:|:---------------------:|:---------------------:|:---------------------:|
| Conll-Dev | **0.68** / 0.85 |        0.39 / 0.70    |     0.52 / 0.77       |     0.33 / 0.86       |
| MSNBC     | **0.65** / 0.82 |        0.28 / 0.80    |     0.46 / 0.87       |     0.23 / 0.90       |


# Train
If you want to train a new model, then you need to generate trainig, dev and test data in the format used by MEL.
To accomplish this, we provide a script: [gen_train_data.py](scripts/gen_train_data.py). The input to this file is data 
in [AIDA Conll-YAGO dataset](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/)
format which we can't distribute due to licensing issues. Each document must start like so:

-DOCSTART- ([DOC ID])

The doc id can be anything you choose, but for dev documents it should include 'testa' and for test documents it 
should include 'testb'. Each line thereafter, should have four feilds separated by tabs: Token,
[BIO tag](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)), full name of mention, 
title of entity it refers to and wikipedia page URL. For example:

German    B    German    German    http://en.wikipedia.org/wiki/Germany

European    B    European Commision    European_Commission     http://en.wikipedia.org/wiki/European_Commission

Commision    I    European Commision    European_Commission     http://en.wikipedia.org/wiki/European_Commission

Each document should be demarcated by a blank line. A default config file is provided and can be used to 
train a new model on CPU like so

```python train.py --my-config configs/default.yaml --use_cuda False --data_path data```.


# Flask server

Setting up a server is as easy as running

```python app.py --data_path data --model conll_v0.1.pt```

An example notebook on how to use the API is [here](notebooks/API%20example.ipynb).

# Speed

MEL is efficient as it spends most of its compute time running either spacy's cython code or PyTorch's C code. 
Here we compare MEL's server against TagMe using their [API](https://services.d4science.org/web/tagme/).

| API   | words / second |
|:-----:|:--------------:|
| MEL   |   10,997       |
| TagMe |   1,095        |

**Note**: our server was setup to run on a LAN and TagMe API likely has much higher network latency, still we are 
about ten times faster! 



# Contact

Rohit Gupta - [rohitg1594@gmail.com](rohitg1594@gmail.com)

Samuel Broscheit - [samuel.broscheit@gmail.com](samuel.broscheit@gmail.com )

# References

* Learning Distributed Representations of Texts and Entities from Knowledge Base, Yamada et al.

* Entity Disambiguation with Web Links, Chisholm et al.