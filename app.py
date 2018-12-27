# Runs a simple flask service on a server
from flask import Flask, request, jsonify
import numpy as np
import argparse
from os.path import join
import torch
import logging

from src.pipeline.features import FeatureGenerator
from src.pipeline.detector import SpacyDetector
from src.pipeline.coref import HeuresticCorefResolver
from src.pipeline.candidates import NelCandidateGenerator
from src.models.mlpmodel import MLPModel
from src.utils.file import FileObjectStore
from src.repr.doc import Doc


np.warnings.filterwarnings('ignore')

app = Flask(__name__)

MAX_CANDS = 256

def setup(data_path, args):
    app.logger.info('loading models params.....')
    state_dict = torch.load(join(data_path, f'models/{args.model}'), map_location='cpu')['state_dict']
    ent_embs = state_dict['ent_embs.weight']
    word_embs = state_dict['word_embs.weight']
    app.logger.info('yamada models loaded.')

    app.logger.info('creating file stores.....')
    dict_names = ['ent_dict', 'word_dict', 'redirects', 'str_prior', 'str_cond', 'disamb', 'str_necounts']
    file_stores = {}
    for dict_name in dict_names:
        file_stores[dict_name] = FileObjectStore(join(data_path, f'mmaps/{dict_name}'))

    app.logger.info('creating preprocessor, respolver, detector and candidate generator.....')
    processor = FeatureGenerator(**file_stores)
    coref_resolver = HeuresticCorefResolver()
    detector = SpacyDetector()
    candidate_generator = NelCandidateGenerator(max_cands=MAX_CANDS,
                                                disamb=file_stores['disamb'],
                                                redirects=file_stores['redirects'],
                                                str_necounts=file_stores['str_necounts'])
    app.logger.info('created')

    args.hidden_size = state_dict['hidden.weight'].shape[0]

    app.logger.info('creating model.....')
    model = MLPModel(ent_embs=ent_embs,
                     word_embs=word_embs,
                     args=args)
    model.load_state_dict(state_dict)
    model.eval()

    app.logger.info('models created.')

    return processor, coref_resolver, detector, candidate_generator, model, file_stores


@app.route('/link', methods=['GET', 'POST'])
def linking():

    content = request.get_json(force=True)
    text = content.get('text', '')
    user_mentions = content.get('mentions', [])
    user_spans = content.get('spans', [])
    max_cands = content.get('max_cands', MAX_CANDS)

    Candidate_generator.max_cands = max_cands

    doc = Doc(text,
              file_stores=File_stores,
              text_spans=list(zip(user_mentions, user_spans)),
              coref_resolver=Coref_resolver,
              detector=Detector,
              candidate_generator=Candidate_generator)

    input_dict = processor.process(doc)
    candidate_strs = input_dict['candidate_strs']
    input_dict.pop('candidate_strs')

    for k, v in input_dict.items():
        print(k, v[:5])

    scores, _, _ = Model(input_dict)

    pred_mask = torch.argmax(scores, len(scores.shape) - 1)
    entities = candidate_strs[np.arange(len(candidate_strs)), pred_mask.numpy()].tolist()

    mentions = [mention.text for mention in doc.mentions]
    mention_spans = [(mention.begin, mention.end) for mention in doc.mentions]

    for i, ent in enumerate(entities):
        if len(ent) == 0:
            entities.pop(i)
            mentions.pop(i)
            mention_spans.pop(i)

    assert len(mentions) == len(entities) == len(mention_spans)

    return jsonify({'mentions': mentions, 'entities': entities, 'spans': mention_spans}), 201


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="flask app for mannheim-nel",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', required=True, help='path to data directory')
    parser.add_argument('--model', required=True, help='model name, must be in {data_path}/models')
    parser.add_argument('--port', type=int, default=5000, help='port of flask server')
    parser.add_argument('--host', default='127.0.0.1', help='server host')
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Hack: need this as it is included in model definition
    Args = parser.parse_args()
    Args.dp = 0.0

    Data_path = Args.data_path
    processor, Coref_resolver, Detector, Candidate_generator, Model, File_stores = setup(Data_path, Args)

    app.run(host=Args.host, port=Args.port)
    app.logger.info('Setup complete, app online.')
