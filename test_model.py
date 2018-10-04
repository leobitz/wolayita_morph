import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import text_processor as tp
from models import *
from training_handler import TrainingHandler
from models import *
from data_gen import *
from keras.models import load_model

dg = DataGen()
test_n_batches, test_batch_size = 30, 100 
test_gen = dg.gen(batch_size=test_batch_size, n_batches=test_n_batches, trainset=False)

n_features = len(char2int)
n_steps_in = dg.max_root_len
n_steps_out = dg.max_output_len

corpus = "dataset_train.txt"
tag_name = "double_GRU_128"
model_name = "morpholizer"

model = load_model("model_weights/morpholizer_enc_dec_256_512/model_weight-02-0.0095.hdf5")
# model.summary()
feat_units = 15
n_dec_units = 512
encoder_inputs = model.input[0]

encoder_outputs, state_h, state_c = model.get_layer('encoder_lstm').output
feature_input = model.input[2]
feat_out =  model.get_layer('feature_output').output
state_h = model.get_layer('dense_1').output
state_c = model.get_layer('dense_2').output
encoder_states = [state_h, state_c]
encoder_model = Model([encoder_inputs, feature_input], encoder_states)


decoder_inputs =  model.input[1]
# # define inference decoder
decoder_state_input_h = Input(shape=(n_dec_units,))
decoder_state_input_c = Input(shape=(n_dec_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm = model.get_layer('decoder_lstm')
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]

decoder_dense = model.get_layer('train_output')
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


total, correct = 0, 0
sims = []
for b in range(test_n_batches):
    [X1, X2, X3], y = next(test_gen)
    for j in range(test_batch_size):
        X33 = X3[j].reshape((1, X3.shape[1])) 
        X11 = X1[j].reshape((1, X1.shape[1], X1.shape[2]))
        target = predict(encoder_model, decoder_model, X11, X33, n_steps_out, n_features)
        root = ''.join(dg.one_hot_decode(X1[j])).replace('&', ' ')
        word = ''.join(dg.one_hot_decode(y[j])).replace('&', ' ')
        targetS = ''.join(dg.one_hot_decode(target)).replace('&', ' ')
        sims.append(dg.word_sim(word, targetS))
        if dg.one_hot_decode(y[j]) == dg.one_hot_decode(target):
            correct += 1
    print(b, root, word, targetS)
    total += test_batch_size
    
word_sim_average = sum(sims)/len(sims)
print('Word Similarity Average: {0:.2f}%'.format(word_sim_average))
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
