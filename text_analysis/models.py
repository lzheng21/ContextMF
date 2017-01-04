'''
Created on Dec 8, 2015

@author: donghyun
'''
import numpy as np
np.random.seed(1337)

from keras.callbacks import EarlyStopping
from keras.models import *
from keras.layers.convolutional import *
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


class CNN_module():
    '''
    classdocs
    '''
    batch_size = 128
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters, init_W=None):

        self.max_len = max_len
        max_features = vocab_size
        vanila_dimension = 200
        projection_dimension = output_dimesion

        '''
        filter_lengths = [3, 4, 5]
        self.model = Graph()


        self.model.add_input(name='input', input_shape=(max_len,), dtype='int')

        if init_W is None:
            self.model.add_node(Embedding(
                max_features, emb_dim, input_length=max_len), name='sentence_embeddings', input='input')
        else:
            self.model.add_node(Embedding(max_features, emb_dim, input_length=max_len, weights=[
                                init_W / 20]), name='sentence_embeddings', input='input')


        for i in filter_lengths:
            model_internal = Sequential()
            model_internal.add(
                Reshape(dims=(1, self.max_len, emb_dim), input_shape=(self.max_len, emb_dim)))
            model_internal.add(Convolution2D(
                nb_filters, i, emb_dim, activation="relu"))
            model_internal.add(MaxPooling2D(
                pool_size=(self.max_len - i + 1, 1)))
            model_internal.add(Flatten())

            self.model.add_node(model_internal, name='unit_' +
                                str(i), input='sentence_embeddings')


        self.model.add_node(Dense(vanila_dimension, activation='tanh'),
                            name='fully_connect', inputs=['unit_' + str(i) for i in filter_lengths])
        self.model.add_node(Dropout(dropout_rate),
                            name='dropout', input='fully_connect')

        self.model.add_node(Dense(projection_dimension, activation='tanh'),
                            name='projection', input='dropout')

        # Output Layer
        self.model.add_output(name='output', input='projection')
        '''
        self.model = Sequential()  # or Graph or whatever
        self.model.add(Embedding(output_dim=emb_dim, input_dim=vocab_size,
                                        mask_zero=False))
        # user_review_model.add(Reshape((1,word_dim,seq_len)))
        self.model.add(Convolution1D(nb_filter=64,
                                filter_length=3,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1))  # we use max over time pooling by defining a python function to use

        # in a Lambda layer
        def max_1d(X):
            return K.max(X, axis=1)

        self.model.add(Lambda(max_1d, output_shape=(64,)))

        self.model.add(Dense(output_dimesion, activation='relu',name='output'))
        #self.model.add(Dropout(0.2))

        #self.model.compile('rmsprop', {'output': 'mse'})
        self.model.compile(loss='mse',optimizer='rmsprop')

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def qualitative_CNN(self, vocab_size, emb_dim, max_len, nb_filters):
        self.max_len = max_len
        max_features = vocab_size

        filter_lengths = [3, 4, 5]
        print("Build model...")
        self.qual_model = Graph()
        self.qual_conv_set = {}
        '''Embedding Layer'''
        self.qual_model.add_input(
            name='input', input_shape=(max_len,), dtype=int)

        self.qual_model.add_node(Embedding(max_features, emb_dim, input_length=max_len, weights=self.model.nodes['sentence_embeddings'].get_weights()),
                                 name='sentence_embeddings', input='input')

        '''Convolution Layer & Max Pooling Layer'''
        for i in filter_lengths:
            model_internal = Sequential()
            model_internal.add(
                Reshape(dims=(1, max_len, emb_dim), input_shape=(max_len, emb_dim)))
            self.qual_conv_set[i] = Convolution2D(nb_filters, i, emb_dim, activation="relu", weights=self.model.nodes[
                                                  'unit_' + str(i)].layers[1].get_weights())
            model_internal.add(self.qual_conv_set[i])
            model_internal.add(MaxPooling2D(pool_size=(max_len - i + 1, 1)))
            model_internal.add(Flatten())

            self.qual_model.add_node(
                model_internal, name='unit_' + str(i), input='sentence_embeddings')
            self.qual_model.add_output(
                name='output_' + str(i), input='unit_' + str(i))

        self.qual_model.compile(
            'rmsprop', {'output_3': 'mse', 'output_4': 'mse', 'output_5': 'mse'})

    def train(self, X_train, V, seed):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        np.random.seed(seed)
        X_train = np.random.permutation(X_train)
        np.random.seed(seed)
        V = np.random.permutation(V)

        print("Train...CNN module")
        #history = self.model.fit({'input': X_train, 'output': V},
        #                         verbose=0, batch_size=self.batch_size, nb_epoch=self.nb_epoch, shuffle=True, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=0)])
        history = self.model.fit(X_train,y=V,batch_size=self.batch_size,nb_epoch=self.nb_epoch, shuffle=True, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=0)])

        cnn_loss_his = history.history['loss']
        cmp_cnn_loss = sorted(cnn_loss_his)[::-1]
        if cnn_loss_his != cmp_cnn_loss:
            self.nb_epoch = 1
        return history

    def get_projection_layer(self, X_train):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        #Y = self.model.predict(
        #    {'input': X_train}, batch_size=len(X_train))['output']

        intermediate_layer_model = Model(input=self.model.input,
                                         output=self.model.get_layer('output').output)
        Y = intermediate_layer_model.predict(X_train,batch_size=len(X_train))
        return Y
