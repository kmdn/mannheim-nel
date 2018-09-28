from flask import Flask, request, jsonify

from utils import pickle_load, reverse_dict
from preprocessor import PreProcessor
from model import NEL

import numpy as np
import logging
import sys
from os.path import join

logging.basicConfig(level=logging.DEBUG)
np.warnings.filterwarnings('ignore')

app = Flask(__name__)
ENT_FILTER = {'CARDINAL', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL'}
MAX_CANDS = 100
MAX_CONTEXT = 200


def setup(data_path):
    app.logger.info('loading model params.....')
    model_params = pickle_load(join(data_path, 'models/wiki-model.pickle'))
    app.logger.info('yamada model loaded.')

    necounts = model_params['necounts']
    ent_conditionals = model_params['conditionals']
    ent_priors = model_params['priors']
    ent_dict = model_params['ent_dict']
    id2ent = reverse_dict(ent_dict)

    app.logger.info('loading entity embeddings.....')
    ent_embs = pickle_load(join(data_path, 'embs/ent_embs.pickle'), encoding='latin-1')
    app.logger.info('entity embeddings loaded.')

    app.logger.info('loading word embeddings.....')
    word_embs = pickle_load(join(data_path, 'embs/word_embs.pickle'), encoding='latin-1')
    app.logger.info('word embeddings loaded.')

    app.logger.info('creating preprocessor.....')
    processor = PreProcessor(model_params=model_params,
                             necounts=necounts,
                             ent_priors=ent_priors,
                             ent_conditionals=ent_conditionals,
                             max_cands=MAX_CANDS,
                             max_context=MAX_CONTEXT,
                             filter_out=ENT_FILTER)
    app.logger.info('preprocessor created.')

    app.logger.info('creating model.....')
    nel = NEL(ent_embs=ent_embs,
              word_embs=word_embs,
              params=model_params)
    app.logger.info('model created.')

    return processor, nel, id2ent


@app.route('/results', methods=['GET', 'POST'])
def get_entities():
    text = str(request.get_data().decode('utf-8'))

    input, mentions, mention_spans = processor.process(text)
    context, candidates, priors, conditionals, exact_match, contains = input
    scores = nel(input)
    pred_mask = np.argmax(scores, axis=1)
    preds = candidates[np.arange(len(candidates)), pred_mask]
    entities = [id2ent.get(pred, '') for pred in preds]

    for i, ent in enumerate(entities):
        if len(ent) == 0:
            entities.pop(i)
            mentions.pop(i)
            mention_spans.pop(i)

    assert len(mentions) == len(entities) == len(mention_spans)

    return jsonify({'mentions': mentions, 'entities': entities, 'spans': mention_spans}), 201


if __name__ == '__main__':
    Data_path = sys.argv[1]
    processor, nel, id2ent = setup(Data_path)
    app.logger.info('setup complete.')

    app.logger.info('app online.')

    app.run()
