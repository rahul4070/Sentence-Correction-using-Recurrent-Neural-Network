print('importing libraries...')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed
tf.keras.backend.clear_session()
from tensorflow.keras.layers import Input, Softmax, RNN, Dense, Embedding, LSTM, concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from multiprocessing import Pool
import os
import joblib
import pickle
from tqdm import tqdm
# import streamlit as st
from flask import Flask, jsonify, request
import datetime


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)

print('libraries imported!')

print('-'*120)


tknizer_corr = joblib.load('tknizer_corr')
tknizer_eng  = joblib.load('tknizer_eng')


print('loading embedding vector...')
embeddings_index = dict()
f = open('glove.6B.300d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('embedding vector loaded!')

print('-'*120)


print('loading encoder_embedding_matrix...')
global encoder_embedding_matrix
encoder_embedding_matrix = np.zeros((20146+1, 300))
for word, i in tknizer_corr.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        encoder_embedding_matrix[i] = embedding_vector
print('encoder_embedding_matrix loaded!')
    
print('-'*120)    
    
print('decoder_embedding_matrix...')
global decoder_embedding_matrix
decoder_embedding_matrix = np.zeros((2993+1, 300))
for word, i in tknizer_eng.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        decoder_embedding_matrix[i] = embedding_vector
print('decoder_embedding_matrx loaded!')


print('-'*120)

print('loading the model...')
class Encoder(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns output sequence
    '''

    def __init__(self,inp_vocab_size,embedding_size,lstm_size,input_length):
        super().__init__()

        #Initialize Embedding layer
        #Intialize Encoder LSTM layer
        self.inp_vocab_size = inp_vocab_size
        self.embedding_size = embedding_size
        self.input_length = input_length
        self.lstm_size = lstm_size

        self.embedding = Embedding(input_dim=self.inp_vocab_size, output_dim=self.embedding_size, input_length=self.input_length,
                    mask_zero=True, name="embedding_layer_encoder",
                     weights=[encoder_embedding_matrix], trainable=False
                     )
        self.lstm = LSTM(self.lstm_size, return_state=True, return_sequences=True, name="Encoder_LSTM", 
                             activation ='tanh', kernel_regularizer=l2(1e-3)
                             )

    def call(self,input_sequence,states):

        #   This function takes a sequence input and the initial states of the encoder.
        #   Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to encoder_lstm
        #   returns -- All encoder_outputs, last time steps hidden and cell state

        embedding_layer = self.embedding(input_sequence)

        lstm_output, lstm_state_h, lstm_state_c = self.lstm(embedding_layer)
        
        return lstm_output, lstm_state_h, lstm_state_c

    
    def initialize_states(self,batch_size):
    #   '''
    #   Given a batch size it will return intial hidden state and intial cell state.
    #   If batch size is 32- Hidden state is zeros of size [32,lstm_units], cell state zeros is of size [32,lstm_units]
    #   '''
        hidden_state = np.zeros((batch_size, self.lstm_size))
        cell_state = np.zeros((batch_size, self.lstm_size))

        return hidden_state, cell_state


class Attention(tf.keras.layers.Layer):
  '''
    Class the calculates score based on the scoring_function using Bahdanu attention mechanism.
  '''
  def __init__(self,scoring_function, att_units):
    # Please go through the reference notebook and research paper to complete the scoring functions
    super().__init__()  

    self.scoring_function = scoring_function
    self.att_units = att_units
    
    if self.scoring_function=='dot':
      # Intialize variables needed for Dot score  function here
      pass
    
    if scoring_function == 'general':
      # Intialize variables needed for General score function here
        self.w1 = tf.keras.layers.Dense(self.att_units)

    elif scoring_function == 'concat':
      # Intialize variables needed for Concat score function here
        self.w11 = tf.keras.layers.Dense(self.att_units)
        self.w21 = tf.keras.layers.Dense(self.att_units)
        self.v = tf.keras.layers.Dense(1)
  
  
  def call(self,decoder_hidden_state,encoder_output):
    '''
      Attention mechanism takes two inputs current step -- decoder_hidden_state and all the encoder_outputs.
      * Based on the scoring function we will find the score or similarity between decoder_hidden_state and encoder_output.
        Multiply the score function with your encoder_outputs to get the context vector.
        Function returns context vector and attention weights(softmax - scores)
    '''
    
    if self.scoring_function == 'dot':
        # Implement Dot score function here
        decoder_hidden_state = tf.expand_dims(decoder_hidden_state, axis=-1)
        score = tf.matmul(encoder_output,decoder_hidden_state)

    elif self.scoring_function == 'general':
        # Implement General score function here
        decoder_hidden_state = tf.expand_dims(decoder_hidden_state, axis=-1)
        score = tf.matmul(self.w1(encoder_output), decoder_hidden_state)

    elif self.scoring_function == 'concat':
        # Implement General score function here
        decoder_hidden_state = tf.expand_dims(decoder_hidden_state, axis=1)
        score = self.v(tf.nn.tanh(self.w11(decoder_hidden_state) + self.w21(encoder_output)))

    weights = tf.nn.softmax(score,axis=1)
    context_vector = weights * encoder_output
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, weights

class OneStepDecoder(tf.keras.Model):
  def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):

    super().__init__()
      # Initialize decoder embedding layer, LSTM and any other objects needed
    self.tar_vocab_size = tar_vocab_size
    self.att_units = att_units
    self.score_fun = score_fun
    self.input_length = input_length
    self.embedding_dim = embedding_dim
    self.dec_units = dec_units

    self.embedding = Embedding(input_dim=self.tar_vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                            mask_zero=True, name="embedding_layer_step_decoder", 
                            weights=[decoder_embedding_matrix], trainable=False
                            )
    
    self.lstm = LSTM(self.dec_units, return_state=True, return_sequences=True, name="One_Step_decoder_LSTM",
                         activation = 'tanh', kernel_regularizer=l2(1e-3)
                         )

    self.attention = Attention(self.score_fun, self.att_units)

    self.dense = Dense(self.tar_vocab_size)


  def call(self,input_to_decoder, encoder_output, state_h,state_c):
    #     One step decoder mechanisim step by step:
    #   A. Pass the input_to_decoder to the embedding layer and then get the output(batch_size,1,embedding_dim)

    # print(input_to_decoder.shape)
    embedding_layer = self.embedding(input_to_decoder)


    #   B. Using the encoder_output and decoder hidden state, compute the context vector.
    context_vector,attention_weights = self.attention(state_h, encoder_output)
    context_vector = tf.expand_dims(context_vector, axis=1)

    # print('embedding layer - ',embedding_layer.shape)
    # print('attention - ', attention.shape)

    #   C. Concat the context vector with the step A output
    concat_layer = concatenate(inputs=[embedding_layer, context_vector])
    #   D. Pass the Step-C output to LSTM/GRU and get the decoder output and states(hidden and cell state)
    lstm_out, state_hidden, state_cell = self.lstm(concat_layer, initial_state=[state_h, state_c])
    lstm_out = tf.squeeze(lstm_out, axis=1)
    # context_vector = tf.squeeze(context_vector, axis=1)
    # states = [state_hidden, state_cell]
    #   E. Pass the decoder output to dense layer(vocab size) and store the result into output.
    output = self.dense(lstm_out)
    #   F. Return the states from step D, output from Step E, attention weights from Step -B
    return output, state_hidden, state_cell, attention_weights, context_vector


class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
      #Intialize necessary variables and create an object from the class onestepdecoder
        super().__init__()
        self.out_vocab_size = out_vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.dec_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units
        self.onestepdecoder = OneStepDecoder(out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units)

        
    def call(self, input_to_decoder,encoder_output,decoder_hidden_state,decoder_cell_state):


        #Initialize an empty Tensor array, that will store the outputs at each and every time step
        all_outputs = tf.TensorArray(tf.float32, size=tf.shape(input_to_decoder)[1], name='output_arrays')

        #Create a tensor array as shown in the reference notebook
        #Iterate till the length of the decoder input
        length =tf.shape(input_to_decoder)[1]
        # print('----', length)
        for i in range(length):
            # print(tf.reshape(input_to_decoder[:,i], [-1,1]))
            # print(input_to_decoder[:].shape)
            # Call onestepdecoder for each token in decoder_input
            output, decoder_hidden_state, decoder_cell_state, _, _ = self.onestepdecoder(input_to_decoder[:,i:i+1], encoder_output, decoder_hidden_state, decoder_cell_state)

            # Store the output in tensorarray
            all_outputs = all_outputs.write(i, output)
        
        all_outputs = tf.transpose(all_outputs.stack(), [1,0,2])
        # Return the tensor array
        return all_outputs
        
        
    

class encoder_decoder(tf.keras.Model):
  def __init__(self, vocab_encoder_len, vocab_decoder_len, embedding_dim, lstm_size, input_length, score_fun, att_units):
    #Intialize objects from encoder decoder
    super().__init__()
    self.vocab_encoder_size = vocab_encoder_len
    self.vocab_decoder_size = vocab_decoder_len
    self.embedding_dim = embedding_dim
    self. lstm_size = lstm_size
    self. input_length = input_length
    self.socre_fun = score_fun
    self.att_units = att_units
    self.encoder = Encoder(self.vocab_encoder_size, self.embedding_dim , lstm_size,input_length)
    self.decoder = Decoder(self.vocab_decoder_size, self.embedding_dim, input_length, lstm_size ,score_fun ,att_units)

  
  def call(self,data):
    #Intialize encoder states, Pass the encoder_sequence to the embedding layer
    initial_state=self.encoder.initialize_states(100)
    encoder_output,state_h,state_c=self.encoder(data[0],initial_state)


    # Decoder initial states are encoder final states, Initialize it accordingly
    # Pass the decoder sequence,encoder_output,decoder states to Decoder
    output=self.decoder(data[1],encoder_output, state_h, state_c)
    # return the decoder output
    return output


class Dataset:
    def __init__(self, data, tknizer_corr, tknizer_eng, max_len):
        self.encoder_inps = data['corrupted_text'].values
        self.decoder_inps = data['english_inp'].values
        self.decoder_outs = data['english_out'].values
        self.tknizer_eng = tknizer_eng
        self.tknizer_corr = tknizer_corr
        self.max_len = max_len

    def __getitem__(self, i):
        self.encoder_seq = self.tknizer_corr.texts_to_sequences([self.encoder_inps[i]]) # need to pass list of values
        self.decoder_inp_seq = self.tknizer_eng.texts_to_sequences([self.decoder_inps[i]])
        self.decoder_out_seq = self.tknizer_eng.texts_to_sequences([self.decoder_outs[i]])

        self.encoder_seq = pad_sequences(self.encoder_seq, maxlen=self.max_len, dtype='int32', padding='post')
        self.decoder_inp_seq = pad_sequences(self.decoder_inp_seq, maxlen=self.max_len, dtype='int32', padding='post')
        self.decoder_out_seq = pad_sequences(self.decoder_out_seq, maxlen=self.max_len, dtype='int32', padding='post')
        return self.encoder_seq, self.decoder_inp_seq, self.decoder_out_seq

    def __len__(self): # your model.fit_gen requires this function
        return len(self.encoder_inps)

    
class Dataloder(tf.keras.utils.Sequence):    
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.dataset.encoder_inps))


    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.squeeze(np.stack(samples, axis=1), axis=0) for samples in zip(*data)]
        # we are creating data like ([italian, english_inp], english_out) these are already converted into seq
        return tuple([[batch[0],batch[1]],batch[2]])

    def __len__(self):  # your model.fit_gen requires this function
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')

# Refer https://www.tensorflow.org/tutorials/text/nmt_with_attention#define_the_optimizer_and_the_loss_function
def custom_lossfunction(targets,logits):

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    loss_ = loss_object(targets, logits)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def learning_rate_scheduler(epoch, learning_rate):
  if epoch % 10 == 0:
    learning_rate = learning_rate * 0.95
    return learning_rate
  else:
    return learning_rate


print('-'*120)


def load_model():
    
    model = encoder_decoder(20146 + 1, 2993 + 1, 300, 300, 50, 'concat', 32)
    optimizer = tf.keras.optimizers.Adam(0.01)
    model.compile(optimizer=optimizer,loss=custom_lossfunction)
    train_dataloader = joblib.load('train_dataloader.pkl')
    validation_dataloader = joblib.load('validation_dataloader.pkl')  
    learningratescheduler = LearningRateScheduler(learning_rate_scheduler)   
    model.fit(x = train_dataloader, steps_per_epoch =129, epochs=1, validation_data=validation_dataloader, 
                validation_steps = 16,
              callbacks =[
                      tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='val_loss', factor=0.2, patience=1),
                    #   tensorboard_callback,
                      learningratescheduler
          ]
                )
    model.load_weights("attention_weights/model.h5")

    return model

# reference: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
    
def preprocess(text):
    """
        Function to clean the strings containing special characters and converts them to lowercase characters.

        input: string
        output: string which contains number and lower character.
    """

    # convert the string to lowercase
    text = text.lower()
    # decontraction - expanding the words like : i'll -> i will, he'd -> he would
    text = decontracted(text)
    text = re.sub('[^A-Za-z0-9]',' ',text)
    text = re.sub('\s_\s', ' ', text)   #  replace strings like  ' _ ' with ' ' (string with a space)
    text = re.sub('\s+', ' ', text).strip()  # replace more than one_space_character to single_space_character

    return text



model = load_model()

print('model loaded!')


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def prediction():

    start_time = datetime.datetime.now()

    data = request.form.to_dict(flat=False)

    print('data------------', data)

    input_sentence = data['corrupted_text'][0]

    input_sentence = preprocess(input_sentence)
    
    print('input_sentence------', input_sentence)

    inp_seq = tknizer_corr.texts_to_sequences([input_sentence])
    inp_seq = pad_sequences(inp_seq,padding='post',maxlen=50)

    encoder = Encoder(inp_vocab_size=20146, embedding_size=100, lstm_size=256, input_length=50)
    states = encoder.initialize_states(32)

    en_outputs,state_h , state_c = model.layers[0](tf.constant(inp_seq), states)
    cur_vec = tf.constant([[tknizer_eng.word_index['<start>']]])
    pred = []
    
    #Here 50 is the max_length of the sequence
    for i in range(50):
        infe_output, state_h, state_c, attention_weights, _ = model.layers[1].layers[0](cur_vec,en_outputs,state_h,state_c)
        # storing the attention weights to plot later on
        cur_vec = np.reshape(np.argmax(infe_output), (1, 1))
        if cur_vec[0][0]:
          pred.append(tknizer_eng.index_word[cur_vec[0][0]])
        else: 
          continue
        if(pred[-1]=='<end>'):
            break
    translated_sentence = ' '.join(pred)


    print('time taken to execute : ', datetime.datetime.now()- start_time)

    print('translated_sentence:- ', translated_sentence)
    
    return jsonify({'translated_sentence': translated_sentence[:-5]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

