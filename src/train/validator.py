# Validator class for yamada model

from os.path import join
import numpy as np
from torch.autograd import Variable
from logging import getLogger

from src.utils.utils import reverse_dict

np.set_printoptions(threshold=10**6)

logger = getLogger()


class Validator:
    def __init__(self,
                 loader=None,
                 args=None,
                 data_type=None,
                 run=None,
                 file_stores=None):

        self.loader = loader
        self.args = args
        self.ent_dict = file_stores['ent_dict']
        self.word_dict = file_stores['word_dict']
        self.rev_ent_dict = reverse_dict(self.ent_dict)
        self.rev_word_dict = reverse_dict(self.word_dict)
        self.data_type = data_type
        self.run = run

        print(f'Validator has batch size: {self.loader.batch_size}')

    def _get_next_batch(self, data_dict):
        skip_keys = ['ent_strs', 'cand_strs', 'not_in_cand']
        for k, v in data_dict.items():
            try:
                if k not in skip_keys:
                    data_dict[k] = Variable(v)
            except:
                print(f'key - {k}, Value - {v}')

        ent_strs, cand_strs, not_in_cand = np.array(data_dict['ent_strs']),\
                                           np.array(data_dict['cand_strs']).T,\
                                           np.array(data_dict['not_in_cand'])
        for k in skip_keys:
            data_dict.pop(k)

        if self.args.use_cuda:
            device = self.args.device if isinstance(self.args.device, int) else self.args.device[0]
            for k, v in data_dict.items():
                data_dict[k] = v.cuda(device)

        return data_dict, ent_strs, cand_strs, not_in_cand

    def get_pred_str(self, batch_no, ids, context, scores, cand_strs, ent_strs):

        comp_str = ''
        for id in ids:
            word_tokens = context[id]
            mention_id = str(batch_no * self.args.batch_size + id)
            context_str = ' '.join([self.rev_word_dict.get(word_token, 'UNK_WORD') for word_token in word_tokens[:50]])
            pred_str = ','.join(cand_strs[id][(-scores[id]).argsort()][:10])
            comp_str += '||'.join([mention_id, ent_strs[id], pred_str, context_str]) + '\n'

        return comp_str

    def validate(self, model):
        model = model.eval()

        total_correct = 0
        total_not_in_cand = 0
        total_mentions = 0
        cor_adjust = 0

        pred_str = ''

        for batch_no, data in enumerate(self.loader, 0):
            data_dict, ent_strs, cand_strs, not_in_cand = self._get_next_batch(data)

            scores, _, _ = model(data_dict)
            scores = scores.cpu().data.numpy()

            # print(scores[:10, :10])

            preds_mask = np.argmax(scores, axis=1)
            preds = cand_strs[np.arange(len(preds_mask)), preds_mask]
            for i, pred in enumerate(preds):
                pred_str += '||'.join([str(batch_no * self.args.batch_size + i), ent_strs[i], pred]) + '\n'

            cor = preds == ent_strs
            num_cor = cor.sum()
            cor_idxs = np.where(cor)[0]

            total_correct += num_cor
            total_mentions += scores.shape[0]
            total_not_in_cand += not_in_cand.sum()
            cor_adjust += not_in_cand[cor_idxs].sum()

        with open(join(self.args.data_path, 'preds', 'conll.csv'), 'w') as f:
            f.write(pred_str)

        return total_mentions, total_not_in_cand, total_correct, cor_adjust

