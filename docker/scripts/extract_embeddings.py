import joblib
import numpy as np
import pickle
import sys
from os.path import join


def extract(joblib_file):
    model_obj = joblib.load(joblib_file)
    word_embedding = model_obj['word_embedding']
    entity_embedding = model_obj['entity_embedding']

    new_word_emb = np.zeros((len(word_embedding) + 1, word_embedding.shape[1]), dtype=np.float32)
    new_ent_emb = np.zeros((len(entity_embedding) + 1, entity_embedding.shape[1]), dtype=np.float32)

    new_word_emb[1:] = word_embedding
    new_ent_emb[1:] = entity_embedding

    new_ent_emb.dump(join('/data', 'ent_embs.pickle'))
    new_word_emb.dump(join('/data', 'word_embs.pickle'))


if __name__ == '__main__':
    joblib_file = sys.argv[1]
