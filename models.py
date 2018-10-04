
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, TimeDistributed
from keras.layers import Concatenate, Flatten
from keras.layers import GRU, Conv2D, MaxPooling2D
from keras.layers import Input, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
# from keras.utils.vis_utils import plot_model
import keras

# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units, n_feature):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input), name="root_word_input")
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    print(state_h.shape)
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output), name="target_word_input")
    
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    
    feature_input = Input(shape=(None, n_feature), name="word_feature_input")
    feat_out = Dense(10, activation="relu", name="feature_output")(feature_input)
    print(feat_out.shape)
    x = Concatenate(name="feature_merge")([decoder_outputs, feat_out])
    decoder_dense = Dense(n_output, activation='softmax', name="train_output")
    decoder_outputs = decoder_dense(x)
    
    model = Model([encoder_inputs, decoder_inputs, feature_input], decoder_outputs)
#     define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    
    word_feature_input = Input(shape=(None, n_feature), name="word_feature_input")
    word_feat_out = Dense(10, activation="relu")(word_feature_input)
    word_out = Concatenate()([decoder_outputs, word_feat_out])
    
    decoder_outputs = decoder_dense(word_out)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs + [word_feature_input] , [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


# In[44]:


def seq(n_input, n_output, n_units, n_feature):
    # define training encoder
    feat_units = 15
    encoder_inputs = Input(shape=(None, n_input), name="root_word_input")
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    
    feature_input = Input(shape=(n_feature,), name="word_feature_input")
    feat_out = Dense(feat_units, activation="relu", name="feature_output")(feature_input)
    x = Concatenate()([state_h, feat_out])
    x2 = Concatenate()([state_c, feat_out])
    state_h = Dense(n_units, activation='relu')(x)
    state_c = Dense(n_units, activation='relu')(x2)
    encoder_states = [state_h, state_c]
    
    decoder_inputs = Input(shape=(None, n_output), name="target_word_input")
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
#     decoder_outputs, _, _ = decoder_lstm(decoder_inputs)
    decoder_dense = Dense(n_output, activation='softmax', name="train_output")
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs, feature_input], decoder_outputs)
#     define inference encoder
    encoder_model = Model([encoder_inputs, feature_input], encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


# In[ ]:


def seq2(n_input, n_output, n_feature, n_enc_units, n_dec_units):
    # define training encoder
    feat_units = 15
    encoder_inputs = Input(shape=(None, n_input), name="root_word_input")
    encoder = LSTM(n_enc_units, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    
    feature_input = Input(shape=(n_feature,), name="word_feature_input")
    feat_out = Dense(feat_units, activation="relu", name="feature_output")(feature_input)
    x = Concatenate()([state_h, feat_out])
    x2 = Concatenate()([state_c, feat_out])
    state_h = Dense(n_dec_units, activation='relu')(x)
    state_c = Dense(n_dec_units, activation='relu')(x2)
    encoder_states = [state_h, state_c]
    
    decoder_inputs = Input(shape=(None, n_output), name="target_word_input")
    decoder_lstm = LSTM(n_dec_units, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax', name="train_output")
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs, feature_input], decoder_outputs)

    encoder_model = Model([encoder_inputs, feature_input], encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_dec_units,))
    decoder_state_input_c = Input(shape=(n_dec_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


# In[45]:


# train, infenc, infdec = seq(27, 27, 128, 29)


# In[21]:


def conv_model(n_root_input, n_output, n_word_feature):
    root_word_input = Input(shape=(15, 27, 1), name="root_word_input")
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(root_word_input)
    x = MaxPooling2D(2, 2)(x)
    flat_output = Flatten()(x)
    print(flat_output.shape)


# In[22]:


# conv_model(10, 10, 10)


# In[1]:


def predict(infenc, infdec, source, feat, n_steps, cardinality):
    # encode
    state = infenc.predict([source, feat])
    # start of sequence input
    start = [0.0 for _ in range(cardinality)]
#     start[0] = 1
    target_seq = np.array(start).reshape(1, 1, cardinality)
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
    return np.array(output)

