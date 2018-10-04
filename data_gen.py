import numpy as np
from text_processor import *

class DataGen:

    def __init__(self, *args, **kwargs):
        self.roots, self.words, self.featArray, mr, mw = get_feature_array()
        self.n_chars = len(char2int)
        self.max_root_len = mr
        self.max_output_len = mw
        self.word_feat_len = len(self.featArray[0])

    def gen(self, batch_size=100, n_batches=-1, trainset=True):
        max_batch, min_batch = 0, 0
        if trainset == True:
            max_batch = int(len(self.words) * .7) // batch_size
            min_batch = 0
        else:
            max_batch = len(self.words)/ batch_size
            min_batch = int(len(self.words) * .7 // batch_size)
        
        total_batchs = max_batch
        batch = min_batch
        while True:
            rootX, target_inX, featX, y = list(), list(), list(), list()
            for i in range(batch * batch_size, (1 + batch) * batch_size):
                root = self.roots[i]
                word = self.words[i]
                word_feature = self.featArray[i]
                root_encoded, target_encoded, target_in_encoded = self.encond_input_output(root, word)
                rootX.append(root_encoded)
                target_inX.append(target_in_encoded)
                featX.append(word_feature)
                y.append(target_encoded)
            yield [np.array(rootX), np.array(target_inX), np.array(featX)], np.array(y)
            batch += 1
            if batch == total_batchs or batch == n_batches:
                batch = min_batch


    def word2vec(self, word, max_chars):
        vec = np.zeros((max_chars, self.n_chars))
        for i in range(len(word)):
            vec[i][char2int[word[i]]] = 1
        vec[len(word):, 1] = 1
        return vec
    
    def char2vec(self, char):
        vec = np.zeros((self.n_chars,))
        vec[char2int[char]] = 1
        return vec
    
    def encond_input_output(self, root_word, target_word):
        root_word = list(root_word)
        target_word = list(target_word)
#         target_word.reverse()
        target_word_in = [" "] + target_word[:-1]
        root_encoded = self.word2vec(root_word, self.max_root_len)
        target_encoded = self.word2vec(target_word, self.max_output_len)
        target_in_encoded = self.word2vec(target_word_in, self.max_output_len)
        return root_encoded, target_encoded, target_in_encoded

        
    def one_hot_decode(self, vec):
        return [int2char[np.argmax(v)] for v in vec]
    
    def word_sim(self, word1, word2):
        c = 0
        for i in range(len(word1)):
            if word1[i] == word2[i]:
                c += 1
        return c/len(word1)
            

    def get_dataset(self, n=100):
        j = 0
        rootX, target_inX, featX, y = list(), list(), list(), list()
        for i in range(len(self.words)):
            root = self.roots[i]
            word = self.words[i]
            word_feature = self.featArray[i]
            root_encoded, target_encoded, target_in_encoded = self.encond_input_output(root, word)
            rootX.append(root_encoded)
            target_inX.append(target_in_encoded)
            featX.append(word_feature)
            y.append(target_encoded)
            j += 1
            if j == n: break
        return np.array(rootX), np.array(target_inX), np.array(featX), np.array(y)
            
