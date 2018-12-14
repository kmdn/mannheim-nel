# Runs a simple flask service on a server
from flask import Flask, request, jsonify
import numpy as np
import argparse
from os.path import join
import torch
import logging

from src.utils.utils import pickle_load
from src.features.preprocessor import PreProcessor
from src.features.detector import SpacyDetector
from src.features.coref import HeuresticCorefResolver
from src.features.candidates import NelCandidateGenerator
from src.models.model import Model
from src.utils.file import FileObjectStore
from src.repr.doc import Doc


np.warnings.filterwarnings('ignore')

app = Flask(__name__)


def setup(data_path, args):
    app.logger.info('loading models params.....')
    model_params = pickle_load(join(data_path, f'models/{args.model}'))
    ent_embs = model_params['ent_embs.weight']
    word_embs = model_params['word_embs.weight']
    app.logger.info('yamada models loaded.')

    app.logger.info('creating file stores.....')
    dict_names = ['ent_dict', 'word_dict', 'redirects', 'str_prior', 'str_cond', 'disamb', 'str_necounts']
    file_stores = {}
    for dict_name in dict_names:
        file_stores[dict_name] = FileObjectStore(join(data_path, f'mmaps/{dict_name}'))

    app.logger.info('creating preprocessor, respolver, detector and candidate generator.....')
    processor = PreProcessor(**file_stores)
    coref_resolver = HeuresticCorefResolver()
    detector = SpacyDetector()
    candidate_generator = NelCandidateGenerator(max_cands=100,
                                                disamb=file_stores['disamb'],
                                                redirects=file_stores['redirects'],
                                                str_necounts=file_stores['str_necounts'])
    app.logger.info('created')

    args.hidden_size = model_params['hidden.weight'].shape[1]

    app.logger.info('creating models.....')
    nel = Model(ent_embs=ent_embs,
                word_embs=word_embs,
                W=model_params['orig_linear.weight'],
                b=model_params['orig_linear.bias'],
                args=args)
    nel.hidden.weight.data = torch.from_numpy(model_params['hidden.weight'])
    nel.hidden.bias.data = torch.from_numpy(model_params['hidden.bias'])
    nel.output.weight.data = torch.from_numpy(model_params['output.weight'])
    nel.output.bias.data = torch.from_numpy(model_params['output.bias'])
    nel.eval()

    app.logger.info('models created.')

    return processor, coref_resolver, detector, candidate_generator, nel, file_stores


@app.route('/link', methods=['GET', 'POST'])
def linking():

    content = request.get_json(force=True)
    text = content.get('text', '')
    user_mentions = content.get('mentions', [])
    user_spans = content.get('spans', [])
    max_cands = content.get('max_cands', 100)

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

    scores, _, _ = nel(input_dict)

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
    parser.add_argument('-d', '--data_path', required=True, help='path to data directory')
    parser.add_argument('-m', '--model', required=True, help='model name, must be in {data_path}/models')
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Hack: need this as it is included in model definition
    Args = parser.parse_args()
    Args.dp = 0.0

    Data_path = Args.data_path
    processor, Coref_resolver, Detector, Candidate_generator, nel, File_stores = setup(Data_path, Args)

    app.logger.info('Setup complete, app online.')
    app.run()
