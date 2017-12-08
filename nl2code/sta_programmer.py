import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers
from .utils import load_embeddings

from config import config_info
import config

class S2AProgrammer(nn.Module):
    """A simple seq2seq network

    Currently using greedy decoding. Adopted from parlai seq2seq agent. Shared
    not implemented.
    """
    RNN_TYPES = {'lstm': nn.LSTM}

    def __init__(self):
        # check for cuda
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print('[ Using CUDA ]')
            torch.cuda.set_device(config.gpu)
        # query embedding
        # TODO: use load_embedding
        self.query_embedding = nn.Embedding(config.source_vocab_size,
                                            config.word_embed_dim,
                                            padding_idx=0)

        # currently trained from scratch
        if config.fix_embeddings:
            for p in self.query_embedding.parameters():
                p.requires_grad = False

        # or keep some fixed with a register buffer

        # RNN programmer encoder
        self.query_encoder = layers.StackedBRNN(
            input_size=config.word_embed_dim,
            hidden_size=config.encoder_hidden_dim / 2,
            num_layers=1)

        # RNN programmer decoder
        self.decoder = layers.CondAttLSTM(
            config.rule_embed_dim*2 + config.node_embed_dim,
            config.decoder_hidden_dim, config.encoder_hidden_dim,
            config.attention_hidden_dim)

        # RNN pointer network
        self.src_ptr_net = layers.PointerNet()

        self.terminal_gen_linear = nn.Linear(config.decoder_hidden_dim, 2)
        # need softmax after this

        # init rule embedding W, b
        self.rule_embedding_W = torch.Tensor(config.rule_num,
                                             config.rule_embed_dim).normal_(0, 0.1)
        self.rule_embedding_b = torch.Tensor(config.rule_num).zero_()
        # init node embedding
        self.node_embedding = torch.Tensor(config.node_num,
                                           config.node_embed_dim).normal_(0, 0.1)
        # init vocab_embedding W, b
        self.vocab_embedding_W = torch.Tensor(config.target_vocab_size,
                                              config.rule_embed_dim).normal_(0, 0.1)
        self.vocab_embedding_b = torch.Tensor(config.target_vocab_size).zero_()

        # decoder_hidden_dim -> action embed
        self.decoder_hidden_state_W_rule = nn.Linear(config.decoder_hidden_dim,
                                                     config.rule_embed_dim)

        self.decoder_hidden_state_W_token = nn.Linear(config.decoder_hidden_dim
                                                      + config.encoder_hidden_dim,
                                                      config.rule_embed_dim)

        # set up tensors once


    def forward(self, query_tokens, tgt_action_seq, tgt_action_seq_type, tgt_node_seq,
                tgt_par_rule_seq, tgt_par_t_seq):
        query_token_embed = self.query_embedding(query_tokens)
        query_token_embed_mask = 1 * (query_tokens > 0)

        query_embed = self.query_encoder(query_token_embed,
                                         mask=query_token_embed_mask,
                                         dropout=config.dropout)

        tgt_node_embed = self.node_embedding[tgt_node_seq]

        # previous action embeddings
        tgt_action_seq_mask = tgt_action_seq_type.sum(2) > 0
        ta_mask = (tgt_action_seq[:, :, 0] > 0).unsqueeze(2)
        tgt_action_seq_embed = ta_mask * self.rule_embedding_W[tgt_action_seq[:, :, 0]] +\
                               (1 - ta_mask) * self.vocab_embedding_W[tgt_action_seq[:, :, 1]]

        tgt_action_seq_embed_tm1 = variable_shift_right(tgt_action_seq_embed)


        # parent rule application embeddings
        tgt_par_rule_embed = self.rule_embedding_W[tgt_par_rule_seq]
        tgt_par_rule_mask = (tgt_par_rule_seq > 0).unsqueeze(2).expand(tgt_par_rule_embed.size())
        tgt_par_rule_embed.data.masked_fill_(tgt_par_rule_mask, 0.)

        decoder_input = torch.cat([tgt_action_seq_embed_tm1,
                                   tgt_node_embed,
                                   tgt_par_rule_embed], -1)

        decoder_hidden_states, _, ctx_vectors = self.decoder(decoder_input,
                                             context=query_embed,
                                             context_mask=query_token_embed_mask,
                                             mask=tgt_action_seq_mask,
                                             parent_t_seq=tgt_par_t_seq,
                                             dropout=config.dropout)







