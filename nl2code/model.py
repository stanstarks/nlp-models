import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from collections import OrderedDict
import logging
import copy
import heapq
import sys

from config import config_info
import config
from lang.grammar import Grammar
from parse import *
from astnode import *
from util import is_numeric
from .s2s_programmer import S2SProgrammer
from .compoments import Hyp

sys.setrecursionlimit(50000)

class Model:
    """High level models that handles decoding, training, predicting and saving

    TODO: use predefined embeddings
    """
    def __init__(self):

        # Building network
        self.network = S2SProgrammer()

        # Building optimizer
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if config.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, config.lr,
                                       momentum=config.momentum,
                                       weight_decay=config.weight_decay)
        elif config.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters, weight_decay=config.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % config.optimizer)

    def update(self, ex):
        # Train mode
        self.network.train()

        # Transfer to GPU
        if config.cuda:
            inputs = [Variable(e.cuda(async=True)) for e in ex[:-1]]
            # tgt_action_seq_type = Variable(ex[-1].cuda(async=True))

        # Run forward
        tgt_prob = self.network(*inputs)

        # Compute loss and accuracies
        tgt_action_seq_mask = inputs[-1]
        likelihood = torch.log(tgt_prob + 1.e-7 * (1 - tgt_action_seq_mask))
        loss = -(likelihood * tgt_action_seq_mask).sum(axis=-1)
        loss = torch.mean(loss)

        # clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # TODO: Clip gradients

        # Update parameters
        self.optimizer.step()

    def predict(self, ex):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if config.cuda:
            inputs = [Variable(e.cuda(async=True), volatile=True) for e in ex[:-1]]

        # Run forward


    def build_decoder(self, query_tokens, query_token_embed, query_token_embed_mask):
        logging.info('building decoder ...')

        # inputs: query_tokens, output: query_embed, query_token_embed_mask
        self.decoder_func_init = theano.function(inputs, outputs)

        # the actual decode step
        self.decoder_func_next_step = theano.function(inputs, outputs)

    def decode(self, example, grammar, terminal_vocab, beam_size, max_time_step, log=False):
        # beam search decoding

        eos = 1
        unk = terminal_vocab.unk
        vocab_embedding = self.vocab_embedding_W.get_value(borrow=True)
        rule_embedding = self.rule_embedding_W.get_value(borrow=True)

        query_tokens = example.data[0]

        query_embed, query_token_embed_mask = self.decoder_func_init(query_tokens)

        completed_hyps = []
        completed_hyp_num = 0
        live_hyp_num = 1

        root_hyp = Hyp(grammar)
        root_hyp.state = np.zeros(config.decoder_hidden_dim).astype('float32')
        root_hyp.cell = np.zeros(config.decoder_hidden_dim).astype('float32')
        root_hyp.action_embed = np.zeros(config.rule_embed_dim).astype('float32')
        root_hyp.node_id = grammar.get_node_type_id(root_hyp.tree.type)
        root_hyp.parent_rule_id = -1

        hyp_samples = [root_hyp]  # [list() for i in range(live_hyp_num)]

        # source word id in the terminal vocab
        src_token_id = [terminal_vocab[t] for t in example.query][:config.max_query_length]
        unk_pos_list = [x for x, t in enumerate(src_token_id) if t == unk]

        # sometimes a word may appear multi-times in the source, in this case,
        # we just copy its first appearing position. Therefore we mask the words
        # appearing second and onwards to -1
        token_set = set()
        for i, tid in enumerate(src_token_id):
            if tid in token_set:
                src_token_id[i] = -1
            else: token_set.add(tid)

        for t in xrange(max_time_step):
            hyp_num = len(hyp_samples)
            # print 'time step [%d]' % t
            decoder_prev_state = np.array([hyp.state for hyp in hyp_samples]).astype('float32')
            decoder_prev_cell = np.array([hyp.cell for hyp in hyp_samples]).astype('float32')

            hist_h = np.zeros((hyp_num, max_time_step, config.decoder_hidden_dim)).astype('float32')

            if t > 0:
                for i, hyp in enumerate(hyp_samples):
                    hist_h[i, :len(hyp.hist_h), :] = hyp.hist_h
                    # for j, h in enumerate(hyp.hist_h):
                    #    hist_h[i, j] = h

            prev_action_embed = np.array([hyp.action_embed for hyp in hyp_samples]).astype('float32')
            node_id = np.array([hyp.node_id for hyp in hyp_samples], dtype='int32')
            parent_rule_id = np.array([hyp.parent_rule_id for hyp in hyp_samples], dtype='int32')
            parent_t = np.array([hyp.get_action_parent_t() for hyp in hyp_samples], dtype='int32')
            query_embed_tiled = np.tile(query_embed, [live_hyp_num, 1, 1])
            query_token_embed_mask_tiled = np.tile(query_token_embed_mask, [live_hyp_num, 1])

            inputs = [np.array([t], dtype='int32'), decoder_prev_state, decoder_prev_cell, hist_h, prev_action_embed,
                      node_id, parent_rule_id, parent_t,
                      query_embed_tiled, query_token_embed_mask_tiled]

            decoder_next_state, decoder_next_cell, \
            rule_prob, gen_action_prob, vocab_prob, copy_prob  = self.decoder_func_next_step(*inputs)

            new_hyp_samples = []

            cut_off_k = beam_size
            score_heap = []

            # iterating over items in the beam
            # print 'time step: %d, hyp num: %d' % (t, live_hyp_num)

            word_prob = gen_action_prob[:, 0:1] * vocab_prob
            word_prob[:, unk] = 0

            hyp_scores = np.array([hyp.score for hyp in hyp_samples])

            # word_prob[:, src_token_id] += gen_action_prob[:, 1:2] * copy_prob[:, :len(src_token_id)]
            # word_prob[:, unk] = 0

            rule_apply_cand_hyp_ids = []
            rule_apply_cand_scores = []
            rule_apply_cand_rules = []
            rule_apply_cand_rule_ids = []

            hyp_frontier_nts = []
            word_gen_hyp_ids = []
            cand_copy_probs = []
            unk_words = []

            for k in xrange(live_hyp_num):
                hyp = hyp_samples[k]

                # if k == 0:
                #     print 'Top Hyp: %s' % hyp.tree.__repr__()

                frontier_nt = hyp.frontier_nt()
                hyp_frontier_nts.append(frontier_nt)

                assert hyp, 'none hyp!'

                # if it's not a leaf
                if not grammar.is_value_node(frontier_nt):
                    # iterate over all the possible rules
                    rules = grammar[frontier_nt.as_type_node] if config.head_nt_constraint else grammar
                    assert len(rules) > 0, 'fail to expand nt node %s' % frontier_nt
                    for rule in rules:
                        rule_id = grammar.rule_to_id[rule]

                        cur_rule_score = np.log(rule_prob[k, rule_id])
                        new_hyp_score = hyp.score + cur_rule_score

                        rule_apply_cand_hyp_ids.append(k)
                        rule_apply_cand_scores.append(new_hyp_score)
                        rule_apply_cand_rules.append(rule)
                        rule_apply_cand_rule_ids.append(rule_id)

                else:  # it's a leaf that holds values
                    cand_copy_prob = 0.0
                    for i, tid in enumerate(src_token_id):
                        if tid != -1:
                            word_prob[k, tid] += gen_action_prob[k, 1] * copy_prob[k, i]
                            cand_copy_prob = gen_action_prob[k, 1]

                    # and unk copy probability
                    if len(unk_pos_list) > 0:
                        unk_pos = copy_prob[k, unk_pos_list].argmax()
                        unk_pos = unk_pos_list[unk_pos]

                        unk_copy_score = gen_action_prob[k, 1] * copy_prob[k, unk_pos]
                        word_prob[k, unk] = unk_copy_score

                        unk_word = example.query[unk_pos]
                        unk_words.append(unk_word)

                        cand_copy_prob = gen_action_prob[k, 1]

                    word_gen_hyp_ids.append(k)
                    cand_copy_probs.append(cand_copy_prob)

            # prune the hyp space
            if completed_hyp_num >= beam_size:
                break

            word_prob = np.log(word_prob)

            word_gen_hyp_num = len(word_gen_hyp_ids)
            rule_apply_cand_num = len(rule_apply_cand_scores)

            if word_gen_hyp_num > 0:
                word_gen_cand_scores = hyp_scores[word_gen_hyp_ids, None] + word_prob[word_gen_hyp_ids, :]
                word_gen_cand_scores_flat = word_gen_cand_scores.flatten()

                cand_scores = np.concatenate([rule_apply_cand_scores, word_gen_cand_scores_flat])
            else:
                cand_scores = np.array(rule_apply_cand_scores)

            top_cand_ids = (-cand_scores).argsort()[:beam_size - completed_hyp_num]

            # expand_cand_num = 0
            for cand_id in top_cand_ids:
                # cand is rule application
                new_hyp = None
                if cand_id < rule_apply_cand_num:
                    hyp_id = rule_apply_cand_hyp_ids[cand_id]
                    hyp = hyp_samples[hyp_id]
                    rule_id = rule_apply_cand_rule_ids[cand_id]
                    rule = rule_apply_cand_rules[cand_id]
                    new_hyp_score = rule_apply_cand_scores[cand_id]

                    new_hyp = Hyp(hyp)
                    new_hyp.apply_rule(rule)

                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.hist_h.append(copy.copy(new_hyp.state))
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = rule_embedding[rule_id]
                else:
                    tid = (cand_id - rule_apply_cand_num) % word_prob.shape[1]
                    word_gen_hyp_id = (cand_id - rule_apply_cand_num) / word_prob.shape[1]
                    hyp_id = word_gen_hyp_ids[word_gen_hyp_id]

                    if tid == unk:
                        token = unk_words[word_gen_hyp_id]
                    else:
                        token = terminal_vocab.id_token_map[tid]

                    frontier_nt = hyp_frontier_nts[hyp_id]
                    # if frontier_nt.type == int and (not (is_numeric(token) or token == '<eos>')):
                    #     continue

                    hyp = hyp_samples[hyp_id]
                    new_hyp_score = word_gen_cand_scores[word_gen_hyp_id, tid]

                    new_hyp = Hyp(hyp)
                    new_hyp.append_token(token)

                    if log:
                        cand_copy_prob = cand_copy_probs[word_gen_hyp_id]
                        if cand_copy_prob > 0.5:
                            new_hyp.log += ' || ' + str(new_hyp.frontier_nt()) + '{copy[%s][p=%f]}' % (token ,cand_copy_prob)

                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.hist_h.append(copy.copy(new_hyp.state))
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = vocab_embedding[tid]
                    new_hyp.node_id = grammar.get_node_type_id(frontier_nt)


                # get the new frontier nt after rule application
                new_frontier_nt = new_hyp.frontier_nt()

                # if new_frontier_nt is None, then we have a new completed hyp!
                if new_frontier_nt is None:
                    # if t <= 1:
                    #     continue

                    new_hyp.n_timestep = t + 1
                    completed_hyps.append(new_hyp)
                    completed_hyp_num += 1

                else:
                    new_hyp.node_id = grammar.get_node_type_id(new_frontier_nt.type)
                    # new_hyp.parent_rule_id = grammar.rule_to_id[
                    #     new_frontier_nt.parent.to_rule(include_value=False)]
                    new_hyp.parent_rule_id = grammar.rule_to_id[new_frontier_nt.parent.applied_rule]

                    new_hyp_samples.append(new_hyp)

                # expand_cand_num += 1
                # if expand_cand_num >= beam_size - completed_hyp_num:
                #     break

                # cand is word generation

            live_hyp_num = min(len(new_hyp_samples), beam_size - completed_hyp_num)
            if live_hyp_num < 1:
                break

            hyp_samples = new_hyp_samples
            # hyp_samples = sorted(new_hyp_samples, key=lambda x: x.score, reverse=True)[:live_hyp_num]

        completed_hyps = sorted(completed_hyps, key=lambda x: x.score, reverse=True)

        return completed_hyps

    @property
    def params_name_to_id(self):
        name_to_id = dict()
        for i, p in enumerate(self.params):
            assert p.name is not None
            # print 'parameter [%s]' % p.name

            name_to_id[p.name] = i

        return name_to_id

    @property
    def params_dict(self):
        assert len(set(p.name for p in self.params)) == len(self.params), 'param name clashes!'
        return OrderedDict((p.name, p) for p in self.params)

    def pull_params(self):
        return OrderedDict([(p_name, p.get_value(borrow=False)) for (p_name, p) in self.params_dict.iteritems()])

    def save(self, model_file, **kwargs):
        logging.info('save model to [%s]', model_file)

        weights_dict = self.pull_params()
        for k, v in kwargs.iteritems():
            weights_dict[k] = v

        np.savez(model_file, **weights_dict)

    def load(self, model_file):
        logging.info('load model from [%s]', model_file)
        weights_dict = np.load(model_file)

        # assert len(weights_dict.files) == len(self.params_dict)

        for p_name, p in self.params_dict.iteritems():
            if p_name not in weights_dict:
                raise RuntimeError('parameter [%s] not in saved weights file', p_name)
            else:
                logging.info('loading parameter [%s]', p_name)
                assert np.array_equal(p.shape.eval(), weights_dict[p_name].shape), \
                    'shape mis-match for [%s]!, %s != %s' % (p_name, p.shape.eval(), weights_dict[p_name].shape)

                p.set_value(weights_dict[p_name])
