# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

from Config import Config


class NetworkVP:
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device), tf.variable_scope('net_') as self.scope:
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
                

    def _create_graph(self):
        self.x = tf.placeholder(
            tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')
        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
        self.global_step = tf.Variable(0, trainable=False, name='step')
        self.is_training = tf.placeholder(tf.bool)
        self.action_index = tf.placeholder(tf.float32, [None, self.num_actions])
        self.step_sizes = tf.placeholder(tf.int32, [None], name='stepsizes')

        #for fast convergence on atari
        self.d1 = self.jchoi_cnn(self.x)

        self.state_in = []  # LSTM input state
        self.state_out = []  # LSTM output state
        input = self.d1

        for i in range(Config.NUM_LSTMS):
            c0 = tf.placeholder(tf.float32, [None, Config.NCELLS])
            h0 = tf.placeholder(tf.float32, [None, Config.NCELLS])
            self.state_in.append((c0,h0))
            rnn_out, rnn_state = self.lstm_layer(input, Config.NCELLS, rnn.LSTMStateTuple(c0,h0), self.step_sizes, 'rnn_'+str(i))
            self.state_out.append(rnn_state)
            input = rnn_out # + input #if residual


        self.logits_v = tf.squeeze(self.dense_layer(input, 1, 'logits_v', func=None), axis=[1])
        self.logits_p = self.dense_layer(input, self.num_actions, 'logits_p', func=None)
        
        self.softmax_p = tf.nn.softmax(self.logits_p)
        self.log_softmax_p = tf.nn.log_softmax(self.logits_p)
        self.log_selected_action_prob = tf.reduce_sum(self.log_softmax_p * self.action_index, axis=1)


        self.sample_action_index = tf.multinomial(self.logits_p - tf.reduce_max(self.logits_p, 1, keep_dims=True), 1) # take 1 sample

        self.cost_p_1 = self.log_selected_action_prob * (self.y_r - tf.stop_gradient(self.logits_v))
        self.cost_p_2 = -1 * self.var_beta * \
                    tf.reduce_sum(self.log_softmax_p * self.softmax_p, axis=1)
        
        mask = tf.reduce_max(self.action_index,axis=1)
        self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v) * mask, axis=0)
        self.cost_p_1_agg = tf.reduce_sum(self.cost_p_1 * mask, axis=0)
        self.cost_p_2_agg = tf.reduce_sum(self.cost_p_2 * mask, axis=0)
        self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg)


        self.cost_all = self.cost_p + self.cost_v

        #self.opt = tf.train.AdamOptimizer(learning_rate=self.var_learning_rate)
        self.opt = tf.train.RMSPropOptimizer(learning_rate=self.var_learning_rate,
                                            decay=Config.RMSPROP_DECAY,
                                            momentum=Config.RMSPROP_MOMENTUM,
                                            epsilon=Config.RMSPROP_EPSILON)

        self.opt_grad = self.opt.compute_gradients(self.cost_all)
        self.opt_grad_clipped = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM),v) for g,v in self.opt_grad]
        self.train_op = self.opt.apply_gradients(self.opt_grad_clipped)


    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_1_agg))
        summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_2_agg))
        summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        #summaries.append(tf.summary.histogram("activation_n1", self.n1))
        #summaries.append(tf.summary.histogram("activation_n2", self.n2))
        summaries.append(tf.summary.histogram("activation_d1", self.d1))
        summaries.append(tf.summary.histogram("activation_v", self.logits_v))
        summaries.append(tf.summary.histogram("activation_p", self.softmax_p))

        #self.summary_op = tf.summary.merge(summaries)
        self.summary_op = tf.summary.merge_all()
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def lstm_layer(self, input, out_dim, initial_state_input, step_sizes, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            cell = rnn.LSTMCell(out_dim, state_is_tuple=True)  # or Basic
            batch_size = tf.shape(self.step_sizes)[0]
            input_reshaped = tf.reshape(input, [batch_size, -1, out_dim])
            outputs, state = tf.nn.dynamic_rnn( cell,
                                                input_reshaped,
                                                initial_state=initial_state_input,
                                                sequence_length=step_sizes,
                                                time_major=False)
            # scope=scope)
            outputs = tf.reshape(outputs, [-1, out_dim])
        return outputs, state


    def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
        #flatten
        if len(input.get_shape().as_list()) > 2:
            flatten_input_shape = input.get_shape()
            nb_elements = flatten_input_shape[1] * flatten_input_shape[2] * flatten_input_shape[3]
            input = tf.reshape(input, shape=[-1, nb_elements._value])
            
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output

    def jchoi_cnn(self, _input):    
       self.n1 = self.conv2d_layer(_input, 3, 32, 'conv1', strides=[1, 2, 2, 1],func=tf.nn.elu)
       self.n2 = self.conv2d_layer(self.n1, 3, 32, 'conv2', strides=[1, 2, 2, 1],func=tf.nn.elu)
       self.n3 = self.conv2d_layer(self.n2, 3, 32, 'conv3', strides=[1, 2, 2, 1],func=tf.nn.elu)
       self.n4 = self.conv2d_layer(self.n3, 3, 32, 'conv4', strides=[1, 2, 2, 1],func=tf.nn.elu)
       self.d1 = self.dense_layer(self.n4, 256, 'dense0')     
       return self.d1	
    
    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate, self.is_training: Config.TRAIN_MODELS}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict_single(self, x):
        return self.predict_p(x[None, :])[0]

    def predict_v(self, x):
        prediction = self.sess.run(self.logits_v, feed_dict={self.x: x})
        return prediction

    def predict_p(self, x):
        prediction = self.sess.run(self.softmax_p, feed_dict={self.x: x})
        return prediction
    
    #rnn version
    # def predict_a_and_v(self, x, c, h):
    #     feed_dict = self.__get_base_feed_dict()
    #     if Config.USE_RNN == False:
    #         feed_dict.update({self.x: x, self.is_training: False})
    #         a, v = self.sess.run([self.sample_action_index, self.logits_v], feed_dict=feed_dict)
    #         return a, v, c, h
    #     else:
    #         step_sizes = np.ones((c.shape[0],),dtype=np.int32)
    #         feed_dict = self.__get_base_feed_dict()
    #         feed_dict.update({self.x: x, self.step_sizes:step_sizes, self.c0:c, self.h0:h, self.is_training: False})
    #         a, v, rnn_state = self.sess.run([self.sample_action_index, self.logits_v, self.lstm_state], feed_dict=feed_dict)
    #         return a, v, rnn_state.c, rnn_state.h

    def predict_a_and_v(self, x, cs, hs):
        batch_size = x.shape[0]
        step_sizes = np.ones((x.shape[0],), dtype=np.int32)
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.step_sizes: step_sizes, self.is_training: False})
        for i in range(Config.NUM_LSTMS):
            c = cs[:, i, :] if i == 1 else cs[:, i]
            h = hs[:, i, :] if i == 1 else hs[:, i]
            feed_dict.update({self.state_in[i]: (c, h)})
        a, v, rnn_out = self.sess.run([self.sample_action_index,  self.logits_v, self.state_out], feed_dict=feed_dict)
        c = np.zeros((batch_size, Config.NUM_LSTMS, Config.NCELLS),dtype=np.float32)
        h = np.zeros((batch_size, Config.NUM_LSTMS, Config.NCELLS),dtype=np.float32)
        for i in range(Config.NUM_LSTMS):
                c[:, i, :] = rnn_out[i].c
                h[:, i, :] = rnn_out[i].h
        return a, v, c, h

    # def train(self, x, y_r, a, c, h, l):
    #     # TODO : define a new OP which dynamically pad tensor
    #     # https://www.tensorflow.org/extend/adding_an_op
    #     r = np.reshape(y_r,(y_r.shape[0],))
    #     feed_dict = self.__get_base_feed_dict()
    #
    #     if Config.USE_RNN == False:
    #         feed_dict.update({self.x: x, self.y_r: r, self.action_index: a, self.is_training: True})
    #     else:
    #         step_sizes = np.array(l)
    #         feed_dict.update({self.x: x, self.y_r: r, self.action_index: a, self.step_sizes:step_sizes, self.c0:c, self.h0:h, self.is_training: True})
    #     self.sess.run(self.train_op, feed_dict=feed_dict)

    def train(self, x, y_r, a, c, h, l):
        r = np.reshape(y_r, (y_r.shape[0],))
        step_sizes = np.array(l)
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y_r: r, self.action_index: a, self.step_sizes: step_sizes, self.is_training: True})

        for i in range(Config.NUM_LSTMS):
            cb = np.array(c[i]).reshape((-1, Config.NCELLS))
            hb = np.array(h[i]).reshape((-1, Config.NCELLS))
            cb = c[i]
            hb = h[i]
            feed_dict.update({self.state_in[i]: (cb, hb)})

        self.sess.run(self.train_op, feed_dict=feed_dict)

    def log(self, x, y_r, a, c, h, l):
        r = np.reshape(y_r, (y_r.shape[0],))
        step_sizes = np.array(l)
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update(
            {self.x: x, self.y_r: r, self.action_index: a, self.step_sizes: step_sizes, self.is_training: True})

        for i in range(Config.NUM_LSTMS):
            cb = np.array(c[i]).reshape((-1, Config.NCELLS))
            hb = np.array(h[i]).reshape((-1, Config.NCELLS))
            cb = c[i]
            hb = h[i]
            feed_dict.update({self.state_in[i]: (cb, hb)})
            
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        print('loaded : ', filename)
        return self._get_episode_from_filename(filename)
       
    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))
