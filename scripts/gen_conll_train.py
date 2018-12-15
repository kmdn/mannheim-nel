import argparse
import sys
import os
sys.path.extend('..')
from src.utils.utils import *
from src.features.detector import SpacyDetector
from src.features.coref import HeuresticCorefResolver
from src.features.candidates import NelCandidateGenerator
from src.utils.file import FileObjectStore
from src.repr.doc import Doc
from src.utils.iter_docs import iter_docs
from logging import getLogger


def load_file_stores(data_path):
    dict_names = ['ent_dict', 'word_dict', 'redirects', 'str_prior', 'str_cond', 'disamb', 'str_necounts']
    file_stores = {}
    for dict_name in dict_names:
        file_stores[dict_name] = FileObjectStore(join(data_path, f'mmaps/{dict_name}'))

    return file_stores


def gen_training_examples(aida_file, data_path):
    logger.info("Loading file stores....")
    file_stores = load_file_stores(data_path)
    logger.info("Loaded.")

    splits = ['train', 'dev', 'test']
    docid2context = {}
    split_examples = {split: [] for split in splits}

    logger.info("Creating split examples....")
    for split in splits:
        for context, mentions, doc_id in iter_docs(aida_file, split, redirects=file_stores['redirects']):
            docid2context[doc_id] = context
            split_examples[split].append([(doc_id, context[begin:end], (begin, end), ent_str) for ent_str, (begin, end) in mentions])
    logger.info("Created.")

    coref_resolver = HeuresticCorefResolver()
    detector = SpacyDetector()
    candidate_generator = NelCandidateGenerator(max_cands=256,
                                                disamb=file_stores['disamb'],
                                                redirects=file_stores['redirects'],
                                                str_necounts=file_stores['str_necounts'])

    logger.info("Creating training examples....")
    full_training_examples = {split: [] for split in splits}
    for split, doc_examples in split_examples.items():
        for examples in doc_examples:
            text_spans = [(text, span) for _, text, span, _ in examples]
            try:
                doc_id = examples[0][0]
            except:
                continue
            doc = Doc(docid2context[doc_id],
                      file_stores=file_stores,
                      detector=detector,
                      candidate_generator=candidate_generator,
                      coref_resolver=coref_resolver,
                      doc_id=doc_id,
                      text_spans=text_spans)
            doc.gen_cands()

            for idx, (doc_id, text, span, ent_str) in enumerate(examples):
                mention = doc.mentions[idx]
                assert mention.text == text, (mention.text, text)
                assert (mention.begin, mention.end) == span
                full_training_examples[split].append('||'.join((doc_id, text, ent_str, '||'.join(mention.cands))))
    logger.info("Created.")

    logger.info("Saving file stores...")
    for split in splits:
        f_name = join(data_path, f'training_files/mmaps/{split}')
        if os.path.exists(f_name):
            os.remove(f_name)
        f_store = FileObjectStore(f_name)
        split_examples = full_training_examples[split]
        f_store.save_many(zip(range(len(split_examples)), split_examples))

    f_name = join(data_path, 'training_files/mmaps/id2context')
    if os.path.exists(f_name):
        os.remove(f_name)
    f_store = FileObjectStore(f_name)
    f_store.save_many(docid2context.items())
    logger.info("Saved. Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="flask app for mannheim-nel",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_path', required=True, help='path to data directory')
    parser.add_argument('-a', '--aida_file', required=True, help='path to aida_yago file')
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = getLogger(__name__)
    args = parser.parse_args()
    gen_training_examples(args.aida_file, args.data_path)
