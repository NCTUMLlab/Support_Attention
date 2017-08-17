from __future__ import print_function

import tensorflow as tf
import numpy as np

from utils import *
from bleu import evaluate

class Evaluater(object):
    def __init__(self,model,batch_size):
        self.model = model
        self.batch_size = batch_size

    def test(self,data):

        features = data['features']

        alphas, sample_caption = self.model.build_sampler()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,'model_ckpt/model-20')
            print('model is restored.')
            feature_batch, image_file = sample_coco_minibatch(data,self.batch_size)
            feed_dict = {self.model.img_feature:feature_batch}
            alps,smp_cap = sess.run([alphas,sample_caption],feed_dict=feed_dict)
            decoded = decode_captions(smp_cap,self.model.idx_to_word)

            np.save('data/visual/alps.npy',alps)
            np.save('data/visual/sample_captions.npy',decoded)
            np.save('data/visual/image_name.npy',image_file)
