# MEL: Mannheim Entity Linking
MEL is a Python library whose goal is to provide an efficient and easy to use end-to-end Entity Linking system.
Entity Linking is the task of linking mentions in free text to entities in a Knowledge Base (in our case Wikipedia).
MEL uses Entitiy and Word Embeddings trained by [ntee](https://github.com/studio-ousia/ntee) and candidate generation
approach of [nel](https://github.com/wikilinks/nel) to provide close to state of the art performance. An easy to
setup bare-bones flask server is also included.

# Dependencies:
* Python 3 with Numpy
* PyTorch
* Spacy
* Flask


# Setup 

1. Clone this repo.
2. We recommend creating a virtual enviroment for this project using conda or pipenv.
3. Install requirements by running
```pip install -r requirements.txt```
4. To use MEL, one needs several dicts that are stored as memory mapped files. These are hosted [here](mmap file link), 
we also provide a pre-trained model [here](conll model file link) trained on [CONLL](conll data link here).
All these filess along with setting up of the project's data structure can be done using a shell script
```
chmod +x bin/setup.sh
bin/setup.sh
```
Note: this may take a long time depending on your internet connection.



# References

* Learning Distributed Representations of Texts and Entities from Knowledge Base, Yamada et al.

* Entity Disambiguation with Web Links, Chisholm et al.