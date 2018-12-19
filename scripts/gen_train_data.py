# Script that reads in data in AIDA_YAGO format and saves training data, dev and test data
import argparse
import sys
import os
sys.path.extend('..')
from src.utils.utils import *
from src.pipeline.detector import SpacyDetector
from src.pipeline.coref import HeuresticCorefResolver
from src.pipeline.candidates import NelCandidateGenerator
from src.pipeline.features import FeatureGenerator
from src.utils.file import FileObjectStore
from src.repr.doc import Doc
from src.utils.iter_docs import iter_docs
from logging import getLogger


def gen_training_examples(train_file, data_path, dataset_name):
    logger.info("Loading file stores....")
    file_stores = load_file_stores(data_path)
    logger.info("Loaded.")

    splits = ['train', 'dev', 'test']
    docid2context = {}
    split_examples = {split: [] for split in splits}

    logger.info("Creating split examples....")
    for split in splits:
        for context, mentions, doc_id in iter_docs(train_file, split, redirects=file_stores['redirects']):
            docid2context[doc_id] = context
            split_examples[split].append([(doc_id, context[begin:end], (begin, end), ent_str) for ent_str, (begin, end) in mentions])
    logger.info("Created.")

    coref_resolver = HeuresticCorefResolver()
    detector = SpacyDetector()
    candidate_generator = NelCandidateGenerator(max_cands=256,
                                                disamb=file_stores['disamb'],
                                                redirects=file_stores['redirects'],
                                                str_necounts=file_stores['str_necounts'])
    feature_generator = FeatureGenerator(str_prior=file_stores['str_prior'],
                                         str_cond=file_stores['str_cond'],
                                         ent_dict=file_stores['ent_dict'])

    logger.info("Creating training examples....")
    full_training_examples = {split: [] for split in splits}
    for split, doc_examples in split_examples.items():
        for examples in doc_examples:
            text_spans = [(text, span) for _, text, span, _ in examples]
            try:
                doc_id = examples[0][0]
            except Exception as e:
                print(e)
                continue
            doc = Doc(docid2context[doc_id],
                      file_stores=file_stores,
                      detector=detector,
                      candidate_generator=candidate_generator,
                      coref_resolver=coref_resolver,
                      doc_id=doc_id,
                      text_spans=text_spans)
            doc.gen_cands()

            all_cand_cond_dict = {}
            for mention_idx, mention in enumerate(doc.mentions):
                _, _, cand_cond_dict = feature_generator.get_stat_feats(file_stores['str_prior'],
                                                                        file_stores['str_cond'],
                                                                        mention.text,
                                                                        mention.cands)
                for cand, cond in cand_cond_dict.items():
                    all_cand_cond_dict[cand] = max(all_cand_cond_dict.get(cand, 0), cond)

            for idx, (doc_id, text, span, ent_str) in enumerate(examples):
                mention = doc.mentions[idx]
                assert mention.text == text, (mention.text, text)
                assert (mention.begin, mention.end) == span
                cand_feature_list = ['@@'.join([cand, str(all_cand_cond_dict[cand])])
                                     for cand in mention.cands]
                full_training_examples[split].append('||'.join((doc_id, text, ent_str,
                                                                '||'.join(cand_feature_list))))
    logger.info("Created.")

    train_data_dir = join(data_path, f'training_files/{dataset_name}')
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)

    logger.info("Saving file stores...")
    for split in splits:
        f_name = join(train_data_dir, split)
        if os.path.exists(f_name):
            os.remove(f_name)
        f_store = FileObjectStore(f_name)
        split_examples = full_training_examples[split]
        f_store.save_many(zip(range(len(split_examples)), split_examples))

    f_name = join(train_data_dir, 'id2context')
    if os.path.exists(f_name):
        os.remove(f_name)
    f_store = FileObjectStore(f_name)
    f_store.save_many(docid2context.items())
    logger.info("Saved. Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="flask app for mannheim-nel",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', required=True, help='path to data directory')
    parser.add_argument('--train_file', required=True, help='path to training file')
    parser.add_argument('--dataset_name', required=True, help='name of directory under which to save generated data')
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = getLogger(__name__)
    args = parser.parse_args()
    gen_training_examples(args.train_file, args.data_path, args.dataset_name, )
