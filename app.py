from flask import Flask, request, jsonify

from utils import pickle_load, json_load
from preprocessor import PreProcessor
from model import NEL

import numpy as np
import logging
from sqlitedict import SqliteDict
from file import FileObjectStore
import sys
from os.path import join
from doc import Doc

np.warnings.filterwarnings('ignore')

app = Flask(__name__)


def setup(data_path):
    app.logger.info('loading model params.....')
    model_params = pickle_load(join(data_path, 'models/conll-model-256.pickle'))
    ent_embs = model_params['ent_embs.weight']
    word_embs = model_params['word_embs.weight']
    app.logger.info('yamada model loaded.')

    app.logger.info('creating file stores.....')
    dict_names = ['ent_dict', 'word_dict', 'redirects', 'str_prior', 'str_cond', 'disamb', 'str_necounts']
    file_stores = {}
    for dict_name in dict_names:
        file_stores[dict_name] = FileObjectStore(join(data_path, f'mmaps/{dict_name}'))

    app.logger.info('creating preprocessor.....')
    processor = PreProcessor(**file_stores)
    app.logger.info('preprocessor created.')

    app.logger.info('creating model.....')
    nel = NEL(ent_embs=ent_embs,
              word_embs=word_embs,
              params=model_params)
    app.logger.info('model created.')

    return processor, nel, file_stores


@app.route('/full', methods=['GET', 'POST'])
def mention_and_linking():

    content = request.get_json(force=True)
    text = content.get('text', '')
    doc = Doc(text,
              file_stores=File_stores)

    model_input = processor.process(doc)
    candidate_strs = model_input['candidate_strs']
    mentions = doc.mention_strings
    mention_spans = doc.mention_spans

    scores = nel(model_input)

    pred_mask = np.argmax(scores, axis=len(scores.shape) - 1)
    entities = candidate_strs[np.arange(len(candidate_strs)), pred_mask].tolist()

    for i, ent in enumerate(entities):
        if len(ent) == 0:
            entities.pop(i)
            mentions.pop(i)
            mention_spans.pop(i)

    assert len(mentions) == len(entities) == len(mention_spans)

    return jsonify({'mentions': mentions, 'entities': entities, 'spans': mention_spans}), 201


@app.route('/link', methods=['GET', 'POST'])
def linking():
    content = request.get_json(force=True)
    text = content.get('text', '')
    user_mentions = content.get('mentions', [])
    user_spans = content.get('spans', [])
    doc_id = content.get('doc_id')

    if not user_mentions:
        return jsonify({'mentions': user_mentions, 'entities': [], 'spans': user_spans}), 201

    doc = Doc(text,
              mentions=user_mentions,
              spans=user_spans,
              file_stores=File_stores,
              doc_id=doc_id)

    model_input = processor.process(doc)
    candidate_strs = model_input['candidate_strs']

    scores = nel(model_input)

    pred_mask = np.argmax(scores, axis=len(scores.shape) - 1)
    entities = candidate_strs[np.arange(len(candidate_strs)), pred_mask].tolist()

    assert len(user_mentions) == len(entities)

    return jsonify({'mentions': user_mentions, 'entities': entities, 'spans': user_spans}), 201


if __name__ == '__main__':
    Data_path = sys.argv[1]
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    processor, nel, File_stores = setup(Data_path)

    app.logger.info('setup complete.')
    app.logger.info('app online.')
    app.run()
