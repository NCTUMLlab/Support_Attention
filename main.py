from __future__ import print_function

import numpy as np
import tensorflow as tf

from core.utils import *
from core.Models import *
from core.Trainer import *

support_context = np.load('data/context/context_model_19_len16.npy')
data = load_coco_data(data_path = './data', split='train')
val_data = load_coco_data(data_path = './data', split='val')
word_to_idx = data['word_to_idx']

model = Support_Attention_Model(word_to_idx=word_to_idx,
                                feature_dim=[196,512],
                                emb_dim = 512,
                                hidden_dim = 1024,
                                n_time_steps = 16)
print('Model is built')
trainer = Trainer(model,optimizer='adam', learning_rate = 1e-3)
trainer.train(data = data,
              support_data = support_context,
              val_data = val_data,
              epochs = 20,
              pretrain_epochs = 10,
              batch_size = 100)
