import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from . import seq2seq
from . import layers
from .utils import load_embeddings

from config import config_info
import config

from collections import deque

class S2SProgrammer():
    """Simple seq2seq network

    Adopted from parlai s2s
    truncate and history not implemented

    share the linear layer's weights with the embedding layer for:
    * vocab_embedding
    * rule_embedding

    """

    RNN_OPTS = {'lstm': nn.LSTM}

    def __init__(self):
        # all instances needs truncate param

        # check for cuda
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print('[ Using CUDA ]')
            torch.cuda.set_device(config.gpu)
        # query embedding
        self.query_embedding = nn.Embedding(config.source_vocab_size,
                                            config.word_embed_dim,
                                            padding_idx=0)
        self.query_encoder = seq2seq.Encoder(dropout=config.dropout)
        self.decoder = seq2seq.Decoder(dropout=config.dropout)

        # currently trained from scratch
        if config.fix_embeddings:
            for p in self.query_embedding.parameters():
                p.requires_grad = False

        # or keep some fixed with a register buffer

        self.src_ptr_net = layers.PointerNet()

        self.terminal_gen_linear = nn.Linear(config.decoder_hidden_dim, 2)
        # need softmax after this

        # init rule embedding W, b
        self.rule_embedding = nn.Embedding(config.rule_num, config.rule_embed_dim,
                                           padding_idx=0)
        self.rule_embedding_trans = Linear(config.rule_embed_dim, config.rule_num,
                                           shared_weight=self.rule_embedding.weight)
        # init node embedding
        self.node_embedding = nn.Embedding(config.node_num, config.node_embed_dim,
                                           padding_idx=0)
        self.node_embedding_trans = Linear(config.node_embed_dim, config.node_num, bias=False,
                                           shared_weight=self.node_embedding.weight)
        # init vocab_embedding W, b
        self.vocab_embedding = nn.Embedding(config.target_vocab_size, config.rule_embed_dim,
                                            padding_idx=0)
        self.vocab_embedding_trans = Linear(config.target_vocab_size, config.rule_embed_dim,
                                            shared_weight=self.vocab_embedding.weight)

        # decoder_hidden_dim -> action embed
        self.decoder_hidden_state_W_rule = nn.Linear(config.decoder_hidden_dim,
                                                     config.rule_embed_dim)

        self.decoder_hidden_state_W_token = nn.Linear(config.decoder_hidden_dim
                                                      + config.encoder_hidden_dim,
                                                      config.rule_embed_dim)

        # set up tensors once


    def train(self, query_tokens, tgt_action_seq, tgt_action_seq_type, tgt_node_seq,
                tgt_par_rule_seq, tgt_par_t_seq):
        """
        all inputs should be transferred to GPU
        """
        query_token_embed = self.query_embedding(query_tokens)
        query_lens = [x for x in torch.sum((query_tokens > 0).int(), dim=1).data]

        query_embed = self.query_encoder(query_token_embed,
                                         x_lens=query_lens)

        tgt_node_embed = self.node_embedding(tgt_node_seq)

        # previous action embeddings
        tgt_action_seq_mask = tgt_action_seq_type.sum(2) > 0
        ta_mask = tgt_action_seq[:, :, 0].ge(0)
        indices = ta_mask.nonzeros()
        ta_rule = self.rule_embedding(tgt_action_seq[:, :, 0])
        tgt_action_seq_embed = self.vocab_embedding(tgt_action_seq[:, :, 1])
        tgt_action_seq_embed.data[indices] = ta_rule.data[indices]

        # right shift to get current target
        tgt_action_seq_embed_tm1 = torch.zeros_like(tgt_action_seq_embed)
        tgt_action_seq_embed_tm1.data[:, 1:, :] = tgt_action_seq_embed.data[:, :-1, :]

        # parent rule application embeddings
        tgt_par_rule_embed = self.rule_embedding(tgt_par_rule_seq)
        tgt_par_rule_mask = tgt_par_rule_seq.ge(0).unsqueeze(2)
        tgt_par_rule_embed.data.masked_fill_(tgt_par_rule_mask.data, 0.)

        decoder_input = torch.cat([tgt_action_seq_embed_tm1,
                                   tgt_node_embed,
                                   tgt_par_rule_embed], -1)

        decoder_hidden_states, _, ctx_vectors = self.decoder(decoder_input,
                                             context=query_embed,
                                             context_mask=query_token_embed_mask,
                                             mask=tgt_action_seq_mask,
                                             parent_t_seq=tgt_par_t_seq)

        ptr_net_decoder_state = torch.cat([decoder_hidden_states, ctx_vectors], -1)

        decoder_hidden_states_trans_rule = self.decoder_hidden_state_W_rule(decoder_hidden_states)
        decoder_hidden_states_trans_token = self.decoder_hidden_state_W_token(
            ptr_net_decoder_state)

        rule_predict = F.softmax(self.rule_embedding_trans(decoder_hidden_states_trans_rule))
        terminal_gen_action_prob = F.softmax(self.terminal_gen_linear(decoder_hidden_states))
        vocab_predict = F.softmax(self.vocab_embedding_trans(decoder_hidden_states_trans_token))


    def embed_input(self, xs):
        xes = F.dropout(self.lt(xs), )
    def decoder_func_next_step(self, time_steps, decoder_prev_state, decoder_prev_cell,
                               hist_h, prev_action_embed, node_id, par_rule_id, parent_t,
                               query_embed, query_token_em)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 shared_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shared = shared_weight is not None

        # init weight
        if not self.shared:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
        else:
            if (shared_weight.size(0) != out_features or
                    shared_weight.size(1) != in_features):
                raise RuntimeError('wrong dimensions for shared weights')
            self.weight = shared_weight

        # init bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        if not self.shared:
            # weight is shared so don't overwrite it
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # detach weight to prevent gradients from changing weight when shared
        weight = self.weight
        if self.shared:
            weight = weight.detach()
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'







