import tensorflow as tf
import numpy as np

from Layer import *

class Support_Attention_Model(object):
    def __init__(self,word_to_idx,feature_dim,emb_dim,hidden_dim,n_time_steps):
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i:w for w,i in word_to_idx.iteritems()}
        self.voc_size = len(word_to_idx)
        self.input_feature_num = feature_dim[0]
        self.input_feature_dim = feature_dim[1]
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_time_steps = n_time_steps
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']
        self.loss_weight = (1,1)
        self.alpha_c = 1.0

        self._build_input()
        self._build_variables()
        self._build_model()

    def _build_input(self):
        self.img_feature = tf.placeholder(tf.float32,[None,self.input_feature_num,self.input_feature_dim])
        self.captions = tf.placeholder(tf.int32,[None,self.n_time_steps+1])
        self.support_context = tf.placeholder(tf.float32,[None,self.n_time_steps,self.input_feature_dim])

    def _build_variables(self):
        self.embedding = Embedding(self.voc_size,self.emb_dim,name = 'word_emb')
        self.feature_to_key = FeatureProj(self.input_feature_dim,self.input_feature_num,self.input_feature_dim,name='feature_proj')
        self.batch_norm = BatchNorm(name = 'batch_norm')

        self.beginner = InitStateGen(self.input_feature_dim,self.hidden_dim,name='beginner')
        self.controller_h1 = Attention_Product(self.input_feature_dim,self.hidden_dim,self.input_feature_num,name='controller_h1')

        self.decoder_h1 = LSTM(self.emb_dim+self.input_feature_dim,self.hidden_dim,activation=None,name='decoder_h1')
        self.decoder_h2 = Dense(self.hidden_dim,self.voc_size,activation=None,name = 'decoder_h2')
        
        self.reconstructer_h1 = Dense(self.input_feature_dim,512,activation=tf.nn.relu,name = 'reconstructer')
        self.reconstructer_h2 = Dense(512,self.input_feature_dim,activation=None,name = 'reconstructer_h2')
    
    def _build_model(self):
        caption_in = self.captions[:,:self.n_time_steps]
        caption_out = self.captions[:,1:]
        mask = tf.to_float(tf.not_equal(caption_out,self._null))
        word_vec = self.embedding(caption_in) # (None,n_time_steps,emb_dim)
        features = self.batch_norm(self.img_feature,'train')
        key = self.feature_to_key(features) # (None,input_feature_num,input_feature_dim)
        cell, state = self.beginner(features) # (None,hidden_dim)
        cell = tf.nn.dropout(cell,0.5)
        state = tf.nn.dropout(state,0.5)

        pred_loss = 0.0
        recon_loss = 0.0
        loss = 0.0
        alpha_list = []
        for t in xrange(self.n_time_steps):
            # alpha=(None,input_feature_num), context=(None,input_feature_dim)
            alpha,context = self.controller_h1(features,key,state)
            alpha_list.append(alpha)

            input_vec = tf.concat(axis = 1,values = [word_vec[:,t,:],context])
            state,cell = self.decoder_h1(input_vec,state,cell)
            cell = tf.nn.dropout(cell,0.5)
            state = tf.nn.dropout(state,0.5)
            logit = self.decoder_h2(state)

            recon = self.reconstructer_h1(context)
            recon = self.reconstructer_h2(recon)

            recon_loss += tf.reduce_sum(tf.squared_difference(self.support_context[:,t,:],recon))
            pred_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,labels=caption_out[:,t])*mask[:,t])

        if self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list),(1,0,2))
            alphas_all = tf.reduce_sum(alphas,1)
            alphas_reg = self.alpha_c*tf.reduce_sum((16./196-alphas_all)**2)
        self.loss = self.loss_weight[0]*recon_loss+self.loss_weight[1]*pred_loss+alphas_reg
        self.pretrain_loss = recon_loss+alphas_reg


    def build_sampler(self):
        sample_word_list = []
        alpha_list = []
        batch_size = tf.shape(self.img_feature)[0]

        features = self.batch_norm(self.img_feature,'test')
        cell, state = self.beginner(features)
        key = self.feature_to_key(features)

        for t in xrange(self.n_time_steps):
            if t == 0:
                x = self.embedding(tf.fill([batch_size],self._start))
            else:
                x = self.embedding(sample_word)

            alpha,context = self.controller_h1(features,key,state)
            alpha_list.append(alpha)

            input_vec = tf.concat(axis = 1,values = [x,context])
            state,cell = self.decoder_h1(input_vec,state,cell)
            logit = self.decoder_h2(state)
            sample_word = tf.arg_max(logit,1)
            sample_word_list.append(sample_word)

        alphas = tf.transpose(tf.stack(alpha_list),(1,0,2))
        sample_caption = tf.transpose(tf.stack(sample_word_list),(1,0))

        return alphas, sample_caption
