from flask import Flask, request, jsonify

from utils import pickle_load, reverse_dict
from preprocessor import PreProcessor
from model import NEL

import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
np.warnings.filterwarnings('ignore')

app = Flask(__name__)
ENT_FILTER = {'CARDINAL', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL'}
MAX_CANDS = 100
MAX_CONTEXT = 200


def setup():
    app.logger.info('loading necounts.....', )
    necounts = pickle_load('data/necounts/normal_necounts.pickle')
    app.logger.info('necounts loaded.')

    app.logger.info('loading yamada model.....')
    yamada_model = pickle_load('data/yamada_model.pickle')
    app.logger.info('yamada model loaded.')

    app.logger.info('loading nel model.....')
    params = pickle_load('data/models/nel-ws-wiki.pickle')
    app.logger.info('nel model loaded.')

    app.logger.info('loading stat features.....')
    ent_conditionals = pickle_load('data/necounts/prior_prob.pickle')
    ent_priors, _ = pickle_load('data/necounts/stats.pickle')
    app.logger.info('stat features loaded.')

    app.logger.info('loading ent dict.....')
    ent_dict = yamada_model['ent_dict']
    id2ent = reverse_dict(ent_dict)
    app.logger.info('ent dict loaded.')

    app.logger.info('creating preprocessor.....')
    processor = PreProcessor(yamada_model=yamada_model,
                             necounts=necounts,
                             ent_priors=ent_priors,
                             ent_conditionals=ent_conditionals,
                             max_cands=MAX_CANDS,
                             max_context=MAX_CONTEXT,
                             filter_out=ENT_FILTER)
    app.logger.info('preprocessor created.')

    app.logger.info('creating model.....')
    nel = NEL(params=params)
    app.logger.info('model created.')

    return processor, nel, id2ent


@app.route('/')
def hello_world():
    return 'Hello World!'


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
    processor, nel, id2ent = setup()
    app.logger.info('setup complete.')

    app.logger.info('app online.')
    app.run()
