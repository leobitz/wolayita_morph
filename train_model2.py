import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import text_processor as tp
from models import *
from data_gen import *
from training_handler import TrainingHandler

dg = DataGen()

n_features = len(char2int)
n_steps_in = dg.max_root_len
n_steps_out = dg.max_output_len

n_batches = 1000
batch_size = 128
n_batches = int(len(dg.words) * .8) // batch_size
gen = dg.gen(batch_size=batch_size, n_batches=n_batches)


epoches = 5

tag_name = "enc_dec_512_512_te"
save_on_every = 10

train, infenc, infdec = seq2(n_features, n_features, dg.word_feat_len, 512, 512)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model_name = "morpholizer"
trainer = TrainingHandler(train, model_name)
trainer.train(tag_name, gen, epoches, n_batches, save_on_every, save_model=True)