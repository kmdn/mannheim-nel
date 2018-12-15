# MEL: Mannheim Entity Linking
MEL is a Python library whose goal is to provide an efficient and easy to use end-to-end Entity Linking system.
Entity Linking is the task of linking mentions in free text to entities in a Knowledge Base (in our case Wikipedia).
For example "Washington" can refer to *https://en.wikipedia.org/wiki/George_Washington* or 
*https://en.wikipedia.org/wiki/Washington,_D.C.* or even *https://en.wikipedia.org/wiki/Federal_government_of_the_United_States.
MEL uses [spacy](https://spacy.io/) for mention detection, Entitiy and Word Embeddings trained by 
[ntee](https://github.com/studio-ousia/ntee) and and implementation of their MLP for linking, 
and the candidate generation approach of
[nel](https://github.com/wikilinks/nel) to provide close to state of the art performance. An easy to
setup flask server is also included.

# Dependencies:
* Python 3 with Numpy
* PyTorch
* Spacy
* Flask


# Setup 

1. Clone this repo.
2. We recommend creating a virtual enviroment for this project using conda or pipenv.
3. Install dependencies by running ```pip install -r requirements.txt```.
4. To use MEL, one needs several dicts that are stored as memory mapped files. These are hosted [here](mmap file link), 
we also provide a pre-trained model [here](conll model file link) trained on [CONLL](conll data link here).
All these files along with setting up of the project's data structure can be done using a shell script
```
chmod +x bin/setup.sh
bin/setup.sh
```
**Note**: This may take a long time depending on your internet connection. ```setup.sh```
 will also create training files on conll data by running ```scripts/gen_conll_train.py```.


# Train
A default config file is provided and can be used to train a new model on CPU like so

```python train.py --my-config configs/default.yaml --use_cuda False --data_path data```

# Flask server

Setting up a server is as easy as running

```python app.py --data_path data --model conll_v0.1.pt```

# Performance

We compare against the popular [TagMe](https://tagme.d4science.org/tagme/) system and report F1 scores on the combined
mention detection and entity linking task. For mention detection, any predicted mentions with over 80% overlap with 
a gold mention is considered a match. TagMe allows to filter Entity Linking using a threshold parameter, here we show
results for three different values for a fair comparison:

| Data Set  |    MEL   | TagMe (Threshold 0.1) | TagMe (Threshold 0.3) | TagMe (Threshold 0.5) |
|-----------|:--------:|:---------------------:|:---------------------:|:---------------------:|
| Conll-Dev | **0.67** |        0.39           |        0.52           |        0.33           |
| MSNBC     | **0.64** |        0.28           |        0.46           |        0.23           |


# Contact

Rohit Gupta - [rohitg1594@gmail.com](rohitg1594@gmail.com)

Samuel Broscheit - [samuel.broscheit@gmail.com](samuel.broscheit@gmail.com )

# References

* Learning Distributed Representations of Texts and Entities from Knowledge Base, Yamada et al.

* Entity Disambiguation with Web Links, Chisholm et al.