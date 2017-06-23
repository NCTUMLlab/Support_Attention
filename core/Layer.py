import tensorflow as tf

def get_weight(shape,name=None,initializer=tf.contrib.layers.xavier_initializer()):
    return tf.get_variable(name=name,shape=shape,initializer=initializer)

def get_bias(shape,name=None):
    return tf.get_variable(name=name,shape=shape,initializer=tf.random_normal_initializer())

class Dense(object):
    def __init__(self,input_dim,output_dim,activation,name=None):
        with tf.variable_scope(name):
            self.weight     = get_weight(shape=[input_dim,output_dim],name='dense_weight')
            self.bias       = get_bias(shape=[output_dim],name='dense_bias')
            self.activation = activation

    def __call__(self,input_tensor):
        self.logit  = tf.matmul(input_tensor,self.weight)+self.bias
        if self.activation == None:
            self.output = self.logit
        else:
            self.output = self.activation(self.logit)
        return self.output

class Recurrent(object):
    def __init__(self,input_dim, output_dim, activation, name=None):
        with tf.variable_scope(name):
            self.weight_in   = get_weight(shape = [input_dim,output_dim],name='recurrent_input_weight')
            self.weight_time = get_weight(shape=[output_dim,output_dim],name='recurrent_time_weight')
            self.bias        = get_bias(shape=[output_dim],name='recurrent_bias')
            self.activation  = activation
    
    def __call__(self,input_tensor,last_time_step):
        if last_time_step == 'initial_state':
            self.logit = tf.matmul(input_tensor,self.weight_in)+self.bias
        else:
            self.logit = tf.matmul(input_tensor,self.weight_in)+tf.matmul(last_time_step,self.weight_time)+self.bias

        if self.activation == None:
            self.output = self.logit
        else:
            self.output = self.activation(self.logit)
        return self.output

class LSTM(object):
    def __init__(self,input_dim, output_dim, activation, name=None):
        with tf.variable_scope(name):
            self.w_i   = get_weight(shape = [input_dim,output_dim],name='lstm_w_i')
            self.u_i   = get_weight(shape=[output_dim,output_dim],name='lstm_u_i')
            self.b_i   = get_bias(shape=[output_dim],name='lstm_b_i')
            self.w_f   = get_weight(shape = [input_dim,output_dim],name='lstm_w_f')
            self.u_f   = get_weight(shape=[output_dim,output_dim],name='lstm_u_f')
            self.b_f   = get_bias(shape=[output_dim],name='lstm_b_f')
            self.w_o   = get_weight(shape=[input_dim,output_dim],name='lstm_w_o')
            self.u_o   = get_weight(shape=[output_dim,output_dim],name='lstm_u_o')
            self.b_o   = get_bias(shape=[output_dim],name='lstm_b_o')
            self.w_c   = get_weight(shape=[input_dim,output_dim],name='lstm_w_c')
            self.u_c   = get_weight(shape=[output_dim,output_dim],name='lstm_u_c')
            self.b_c   = get_weight(shape=[output_dim],name='lstm_b_c')
            self.activation  = activation
    
    def __call__(self,input_tensor,last_time_step,last_time_cell):
        if last_time_step == 'initial_state' and last_time_cell=='initial_cell':
            self.f_gate = tf.nn.sigmoid(tf.matmul(input_tensor,self.w_f)+self.b_f)
            self.i_gate = tf.nn.sigmoid(tf.matmul(input_tensor,self.w_i)+self.b_i)
            self.o_gate = tf.nn.sigmoid(tf.matmul(input_tensor,self.w_o)+self.b_o)
            self.cell   = self.i_gate*tf.nn.tanh(tf.matmul(input_tensor,self.w_c)+self.b_c)
            self.hid    = self.o_gate*self.cell
        else:
            self.f_gate = tf.nn.sigmoid(tf.matmul(input_tensor,self.w_f)+tf.matmul(last_time_step,self.u_f)+self.b_f)
            self.i_gate = tf.nn.sigmoid(tf.matmul(input_tensor,self.w_i)+tf.matmul(last_time_step,self.u_i)+self.b_i)
            self.o_gate = tf.nn.sigmoid(tf.matmul(input_tensor,self.w_o)+tf.matmul(last_time_step,self.u_o)+self.b_o)
            self.cell   = self.f_gate*last_time_cell+self.i_gate*tf.nn.tanh(tf.matmul(input_tensor,self.w_c)+self.b_c)
            self.hid    = self.o_gate*self.cell

        if self.activation == None:
            self.output = self.hid
        else:
            self.output = self.activation(self.hid)

        return self.hid, self.cell

class Convolution(object):
    def __init__(self, kernel_shape, activation, name=None):
        with tf.variable_scope(name):
            self.kernel = get_weight(shape=kernel_shape,name='convolution_kernel')
            self.bias   = get_bias(shape = [kernel_shape[-1]],name='convolution_bias')
            self.activation = activation

    def __call__(self,input_tensor,padding='SAME',stride=1):
        self.feature_map = tf.nn.conv2d(input_tensor,self.kernel,strides=[1,stride,stride,1],padding=padding)
        self.feature_map = tf.nn.bias_add(self.feature_map,self.bias)
        if self.activation == None:
            self.output = self.feature_map
        else:
            self.output = self.activation(self.feature_map)
        return self.output

class MaxPooling(object):
    def __init__(self,pool_size=[2,2]):
        self.pool_size = [1,pool_size[0],pool_size[1],1]

    def __call__(self,input_tensor,stride=2,padding='SAME'):
        return tf.nn.max_pool(input_tensor,ksize=self.pool_size,strides=[1,stride,stride,1],padding=padding)

class Embedding(object):
    def __init__(self,input_dim,emb_dim,name=None):
        with tf.variable_scope(name):
            self.emb_matrix = get_weight([input_dim,emb_dim],name='emb_matrix',initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
    
    def __call__(self,input_tensor):
        return tf.nn.embedding_lookup(self.emb_matrix,input_tensor)

class Attention(object):
    def __init__(self,data_dim,state_dim,data_num,name = None):
        with tf.variable_scope(name):
            self.data_dim =data_dim
            self.state_dim = state_dim
            self.data_num = data_num
            self.w = get_weight([state_dim,data_dim],name='attention_weight')
            self.b = get_bias([data_dim],name = 'attention_bias')
            self.w_att = get_weight([data_dim,1],name='attention_output')

    def __call__(self,value,key,hidden_state):
        h_att = tf.nn.relu(key+tf.expand_dims(tf.matmul(hidden_state,self.w),1)+self.b)
        out_att = tf.reshape(tf.matmul(tf.reshape(h_att,[-1,self.data_dim]),self.w_att),[-1,self.data_num])
        alpha = tf.nn.softmax(out_att)
        context = tf.reduce_sum(value*tf.expand_dims(alpha,2),1,name = 'context')
        return alpha, context

class Attention_Product(object):
    def __init__(self,data_dim,state_dim,data_num,name = None):
        with tf.variable_scope(name):
            self.data_dim =data_dim
            self.state_dim = state_dim
            self.data_num = data_num
            self.w = get_weight([state_dim,data_dim],name='attention_weight')
            self.b = get_bias([data_dim],name = 'attention_bias')

    def __call__(self,value,key,hidden_state):
        h_att = tf.nn.relu(tf.matmul(hidden_state,self.w)+self.b)
        match = tf.reduce_sum(tf.multiply(tf.expand_dims(h_att,axis=1),key),axis=2)
        match = tf.divide(match,self.data_dim)
        alpha = tf.nn.softmax(match)
        context = tf.reduce_sum(value*tf.expand_dims(alpha,2),1,name = 'context')
        return alpha, context

class FeatureProj(object):
    def __init__(self,data_dim,data_len,output_dim,name=None):
        with tf.variable_scope(name):
            self.data_dim = data_dim
            self.data_len = data_len
            self.w = get_weight([data_dim,output_dim],name='feat_proj_w')

    def __call__(self,input_tensor):
        feat_flatten = tf.reshape(input_tensor,[-1,self.data_dim])
        feat_proj = tf.matmul(feat_flatten,self.w)
        feat_proj = tf.reshape(feat_proj,[-1,self.data_len,self.data_dim])
        return feat_proj

class InitStateGen(object):
    def __init__(self,data_dim,hidden_dim,name=None):
        with tf.variable_scope(name):
            self.data_dim = data_dim
            self.hid_dim = hidden_dim
            self.w_h = get_weight([data_dim,hidden_dim],name='w_h')
            self.b_h = get_bias([hidden_dim],name='b_h')
            self.w_c = get_weight([data_dim,hidden_dim],name = 'w_c')
            self.b_c = get_bias([hidden_dim],name='b_c')
    
    def __call__(self,feature):
        feat_mean = tf.reduce_mean(feature,axis=1)
        h = tf.nn.tanh(tf.matmul(feat_mean,self.w_h)+self.b_h)
        c = tf.nn.tanh(tf.matmul(feat_mean,self.w_c)+self.b_c)
        return c,h

class BatchNorm(object):
    def __init__(self,name = None):
        self.name = name
    def __call__(self,input_tensor, mode = 'train'):
        if mode == 'train':
            is_training = True
            reuse = None
        else:
            is_training = False
            reuse = True
        
        return tf.contrib.layers.batch_norm(inputs = input_tensor,
                                            is_training = is_training,
                                            updates_collections=None,
                                            reuse = reuse,
                                            scope = self.name)
