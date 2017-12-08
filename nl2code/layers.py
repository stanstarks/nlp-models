import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable

import numpy as np
import logging
import copy

import config
from parse import *

class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # No padding necessary.
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        # Pad if we care or if its during eval.
        if self.padding or not self.training:
            return self._forward_padded(x, x_mask)
        # We don't care.
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

class PointerNet(nn.Module):
    """
    * c = dense2*tanh(dense1_input*e + dense1_h*d)
    * o = softmax(c)

    TODO: add relu activation
    """
    def __init__(self, opt):
        super(PointerNet, self).__init__()

        self.dense1_input = nn.Linear(config.encoder_hidden_dim, config.ptrnet_hidden_dim)

        self.dense1_h = nn.Linear(config.decoder_hidden_dim + config.encoder_hidden_dim, config.ptrnet_hidden_dim)

        self.dense2 = nn.Linear(config.ptrnet_hidden_dim, 1)

    def forward(self, query_embed, query_token_embed_mask, decoder_states):
        query_embed_trans = self.dense1_input(query_embed)
        h_trans = self.dense1_h(decoder_states)

        query_embed_trans = query_embed_trans.unsqueeze(1)
        h_trans = h_trans.unsqueeze(2)

        # (batch_size, max_decode_step, query_token_num, ptr_net_hidden_dim)
        dense1_trans = torch.tanh(query_embed_trans + h_trans)

        scores = self.dense2(dense1_trans)
        scores = scores.view(scores.size(0), scores.size(1), -1)

        # mask padding
        # using the torch softmax instead of defining manually
        query_token_embed_mask = query_token_embed_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(query_token_embed_mask.data, -float('inf'))
        scores = F.softmax(scores, dim=-1)

        """ or try this one
        scores = torch.exp(scores - torch.max(scores, -1, keepdim=True))
        scores *= query_token_embed_mask.unsqueeze(1)
        scores = scores / T.sum(scores, -1, keepdims=True)
        """

        return scores

class CondAttLSTM(nn.Module):
    """
    Conditional LSTM with Attention
    """
    def __init__(self, input_dim, output_dim,
                 context_dim, att_hidden_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='sigmoid', name='CondAttLSTM'):

        super(CondAttLSTM, self).__init__()

        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.context_dim = context_dim
        self.input_dim = input_dim

        # regular LSTM layer
        self.lstm = nn.LSTMCell(input_dim, output_dim)

        # attention layer
        self.att_ctx = nn.Linear(context_dim, att_hidden_dim)
        self.att_h = nn.Linear(output_dim, att_hidden_dim, bias=False)
        self.att = nn.Linear(att_hidden_dim, 1)

        # attention over history
        self.hatt_h = nn.Linear(output_dim, att_hidden_dim, bias=False)
        self.hatt_hist = nn.Linear(output_dim, att_hidden_dim)
        self.hatt = nn.Linear(att_hidden_dim, 1)

    def _step(self,
              t, xi_t, xf_t, xo_t, xc_t, mask_t, parent_t,
              h_tm1, c_tm1, hist_h,
              u_i, u_f, u_o, u_c,
              c_i, c_f, c_o, c_c,
              h_i, h_f, h_o, h_c,
              p_i, p_f, p_o, p_c,
              att_h_w1, att_w2, att_b2,
              context, context_mask, context_att_trans,
              b_u):

        # context: (batch_size, context_size, context_dim)

        # (batch_size, att_layer1_dim)
        h_tm1_att_trans = T.dot(h_tm1, att_h_w1)

        # h_tm1_att_trans = theano.printing.Print('h_tm1_att_trans')(h_tm1_att_trans)

        # (batch_size, context_size, att_layer1_dim)
        att_hidden = T.tanh(context_att_trans + h_tm1_att_trans[:, None, :])
        # (batch_size, context_size, 1)
        att_raw = T.dot(att_hidden, att_w2) + att_b2
        att_raw = att_raw.reshape((att_raw.shape[0], att_raw.shape[1]))

        # (batch_size, context_size)
        if context_mask:
            att_raw.data.masked_fill_(context_mask.data, -float('inf'))
        ctx_att = F.softmax(att_raw, dim=-1)

        # (batch_size, context_dim)
        ctx_vec = T.sum(context * ctx_att[:, :, None], axis=1)

        # t = theano.printing.Print('t')(t)

        ##### attention over history #####

        def _attention_over_history():
            hist_h_mask = T.zeros((hist_h.shape[0], hist_h.shape[1]), dtype='int8')
            hist_h_mask = T.set_subtensor(hist_h_mask[:, T.arange(t)], 1)

            hist_h_att_trans = self.hatt_hist(hist_h)
            h_tm1_hatt_trans = self.hatt_h(h_tm1)

            hatt_hidden = T.tanh(hist_h_att_trans + h_tm1_hatt_trans[:, None, :])
            hatt_raw = T.dot(hatt_hidden, self.hatt_W2) + self.hatt_b2
            hatt_raw = hatt_raw.reshape((hist_h.shape[0], hist_h.shape[1]))
            # hatt_raw = theano.printing.Print('hatt_raw')(hatt_raw)
            hatt_exp = T.exp(hatt_raw - T.max(hatt_raw, axis=-1, keepdims=True)) * hist_h_mask
            # hatt_exp = theano.printing.Print('hatt_exp')(hatt_exp)
            # hatt_exp = hatt_exp.flatten(2)
            h_att_weights = hatt_exp / (T.sum(hatt_exp, axis=-1, keepdims=True) + 1e-7)
            # h_att_weights = theano.printing.Print('h_att_weights')(h_att_weights)

            # (batch_size, output_dim)
            _h_ctx_vec = T.sum(hist_h * h_att_weights[:, :, None], axis=1)

            return _h_ctx_vec

        # maybe autograd supports if statements
        # the first step don't have history
        h_ctx_vec = T.switch(t,
                             _attention_over_history(),
                             T.zeros_like(h_tm1))

        # h_ctx_vec = theano.printing.Print('h_ctx_vec')(h_ctx_vec)

        ##### attention over history #####

        ##### feed in parent hidden state #####

        if not config.parent_hidden_state_feed:
            t = 0

        par_h = T.switch(t,
                         hist_h[T.arange(hist_h.shape[0]), parent_t, :],
                         T.zeros_like(h_tm1))

        ##### feed in parent hidden state #####
        if config.tree_attention:
            i_t = self.inner_activation(
                xi_t + T.dot(h_tm1 * b_u[0], u_i) + T.dot(ctx_vec, c_i) + T.dot(par_h, p_i) + T.dot(h_ctx_vec, h_i))
            f_t = self.inner_activation(
                xf_t + T.dot(h_tm1 * b_u[1], u_f) + T.dot(ctx_vec, c_f) + T.dot(par_h, p_f) + T.dot(h_ctx_vec, h_f))
            c_t = f_t * c_tm1 + i_t * self.activation(
                xc_t + T.dot(h_tm1 * b_u[2], u_c) + T.dot(ctx_vec, c_c) + T.dot(par_h, p_c) + T.dot(h_ctx_vec, h_c))
            o_t = self.inner_activation(
                xo_t + T.dot(h_tm1 * b_u[3], u_o) + T.dot(ctx_vec, c_o) + T.dot(par_h, p_o) + T.dot(h_ctx_vec, h_o))
        else:
            i_t = self.inner_activation(
                xi_t + T.dot(h_tm1 * b_u[0], u_i) + T.dot(ctx_vec, c_i) + T.dot(par_h, p_i))  # + T.dot(h_ctx_vec, h_i)
            f_t = self.inner_activation(
                xf_t + T.dot(h_tm1 * b_u[1], u_f) + T.dot(ctx_vec, c_f) + T.dot(par_h, p_f))  # + T.dot(h_ctx_vec, h_f)
            c_t = f_t * c_tm1 + i_t * self.activation(
                xc_t + T.dot(h_tm1 * b_u[2], u_c) + T.dot(ctx_vec, c_c) + T.dot(par_h, p_c))  # + T.dot(h_ctx_vec, h_c)
            o_t = self.inner_activation(
                xo_t + T.dot(h_tm1 * b_u[3], u_o) + T.dot(ctx_vec, c_o) + T.dot(par_h, p_o))  # + T.dot(h_ctx_vec, h_o)
        h_t = o_t * self.activation(c_t)

        h_t = (1 - mask_t) * h_tm1 + mask_t * h_t
        c_t = (1 - mask_t) * c_tm1 + mask_t * c_t

        new_hist_h = T.set_subtensor(hist_h[:, t, :], h_t)

        return h_t, c_t, ctx_vec, new_hist_h

    def _for_step(self,
                  xi_t, xf_t, xo_t, xc_t, mask_t,
                  h_tm1, c_tm1,
                  context, context_mask, context_att_trans,
                  hist_h, hist_h_att_trans,
                  b_u):

        # context: (batch_size, context_size, context_dim)

        # (batch_size, att_layer1_dim)
        h_tm1_att_trans = T.dot(h_tm1, self.att_h_W1)

        # (batch_size, context_size, att_layer1_dim)
        att_hidden = T.tanh(context_att_trans + h_tm1_att_trans[:, None, :])

        # (batch_size, context_size, 1)
        att_raw = self.att(att_hidden)

        # (batch_size, context_size)
        ctx_att = T.exp(att_raw).reshape((att_raw.shape[0], att_raw.shape[1]))

        if context_mask:
            ctx_att = ctx_att * context_mask

        ctx_att = ctx_att / T.sum(ctx_att, axis=-1, keepdims=True)

        # (batch_size, context_dim)
        ctx_vec = T.sum(context * ctx_att[:, :, None], axis=1)

        ##### attention over history #####

        if hist_h:
            hist_h = T.stack(hist_h).dimshuffle((1, 0, 2))
            hist_h_att_trans = T.stack(hist_h_att_trans).dimshuffle((1, 0, 2))
            h_tm1_hatt_trans = T.dot(h_tm1, self.hatt_h_W1)

            hatt_hidden = T.tanh(hist_h_att_trans + h_tm1_hatt_trans[:, None, :])
            hatt_raw = T.dot(hatt_hidden, self.hatt_W2) + self.hatt_b2
            hatt_raw = hatt_raw.flatten(2)
            h_att_weights = T.nnet.softmax(hatt_raw)

            # (batch_size, output_dim)
            h_ctx_vec = T.sum(hist_h * h_att_weights[:, :, None], axis=1)
        else:
            h_ctx_vec = T.zeros_like(h_tm1)

        ##### attention over history #####

        i_t = self.inner_activation(xi_t + T.dot(h_tm1 * b_u[0], self.U_i) + T.dot(ctx_vec, self.C_i) + T.dot(h_ctx_vec, self.H_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1 * b_u[1], self.U_f) + T.dot(ctx_vec, self.C_f) + T.dot(h_ctx_vec, self.H_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1 * b_u[2], self.U_c) + T.dot(ctx_vec, self.C_c) + T.dot(h_ctx_vec, self.H_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1 * b_u[3], self.U_o) + T.dot(ctx_vec, self.C_o) + T.dot(h_ctx_vec, self.H_o))
        h_t = o_t * self.activation(c_t)

        h_t = (1 - mask_t) * h_tm1 + mask_t * h_t
        c_t = (1 - mask_t) * c_tm1 + mask_t * c_t

        # ctx_vec = theano.printing.Print('ctx_vec')(ctx_vec)

        return h_t, c_t, ctx_vec

    def forward(self, X, context, parent_t_seq, init_state=None, init_cell=None, hist_h=None,
                 mask=None, context_mask=None,
                 dropout=0, train=True, srng=None,
                 time_steps=None):
        assert context_mask.dtype == 'int8', 'context_mask is not int8, got %s' % context_mask.dtype

        # (n_timestep, batch_size)
        mask = self.get_mask(mask, X)
        # (n_timestep, batch_size, input_dim)
        X = X.dimshuffle((1, 0, 2))

        retain_prob = 1. - dropout
        B_w = np.ones((4,), dtype=theano.config.floatX)
        B_u = np.ones((4,), dtype=theano.config.floatX)
        if dropout > 0:
            logging.info('applying dropout with p = %f', dropout)
            if train:
                B_w = srng.binomial((4, X.shape[1], self.input_dim), p=retain_prob,
                                    dtype=theano.config.floatX)
                B_u = srng.binomial((4, X.shape[1], self.output_dim), p=retain_prob,
                                    dtype=theano.config.floatX)
            else:
                B_w *= retain_prob
                B_u *= retain_prob

        # (n_timestep, batch_size, output_dim)
        xi = T.dot(X * B_w[0], self.W_i) + self.b_i
        xf = T.dot(X * B_w[1], self.W_f) + self.b_f
        xc = T.dot(X * B_w[2], self.W_c) + self.b_c
        xo = T.dot(X * B_w[3], self.W_o) + self.b_o

        # (batch_size, context_size, att_layer1_dim)
        context_att_trans = T.dot(context, self.att_ctx_W1) + self.att_b1

        if init_state:
            # (batch_size, output_dim)
            first_state = T.unbroadcast(init_state, 1)
        else:
            first_state = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        if init_cell:
            # (batch_size, output_dim)
            first_cell = T.unbroadcast(init_cell, 1)
        else:
            first_cell = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        if not hist_h:
            # (batch_size, n_timestep, output_dim)
            hist_h = alloc_zeros_matrix(X.shape[1], X.shape[0], self.output_dim)

        if train:
            n_timestep = X.shape[0]
            time_steps = T.arange(n_timestep, dtype='int32')

        # (n_timestep, batch_size)
        parent_t_seq = parent_t_seq.dimshuffle((1, 0))

        [outputs, cells, ctx_vectors, hist_h_outputs], updates = theano.scan(
            self._step,
            sequences=[time_steps, xi, xf, xo, xc, mask, parent_t_seq],
            outputs_info=[
                first_state,  # for h
                first_cell,  # for cell
                None, # T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.context_dim), 1),  # for ctx vector
                hist_h,  # for hist_h
            ],
            non_sequences=[
                self.U_i, self.U_f, self.U_o, self.U_c,
                self.C_i, self.C_f, self.C_o, self.C_c,
                self.H_i, self.H_f, self.H_o, self.H_c,
                self.P_i, self.P_f, self.P_o, self.P_c,
                self.att_h_W1, self.att_W2, self.att_b2,
                context, context_mask, context_att_trans,
                B_u
            ])

        outputs = outputs.dimshuffle((1, 0, 2))
        ctx_vectors = ctx_vectors.dimshuffle((1, 0, 2))
        cells = cells.dimshuffle((1, 0, 2))

        return outputs, cells, ctx_vectors

    def get_mask(self, mask, X):
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)
        mask = mask.astype('int8')

        return mask
