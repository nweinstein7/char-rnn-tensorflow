import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import re
from custom_parser import custom_parse
import nltk

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tags_file = os.path.join(data_dir, "tags.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tags_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tags_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tags_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tags_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        clean_data = nltk.word_tokenize(data)
        tagged_data = [t[1] for t in nltk.pos_tag(clean_data)]
        wordcounter = collections.Counter(clean_data)
        tagcounter = collections.Counter(tagged_data)
        wordcount_pairs = sorted(wordcounter.items(), key=lambda x: -x[1])
        tagcount_pairs = sorted(tagcounter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*wordcount_pairs)
        self.tags, _ = zip(*tagcount_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tag_dict = dict(zip(self.tags, range(len(self.tags))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        with open(tags_file, 'wb') as f:
            cPickle.dump(self.tags, f)
        self.tensor = np.array(zip(list(map(self.vocab.get, clean_data)),list(map(self.tag_dict.get, tagged_data))))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tags_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        with open(tags_file, 'rb') as f:
            self.tags = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tag_dict = dict(zip(self.tags, range(len(self.tags))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int((self.tensor.size / 2) / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int((self.tensor.size / 2) / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."
        end_index = self.num_batches * self.batch_size * self.seq_length
        self.tensor = self.tensor[:end_index, :end_index]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        print("SHAPE: {}".format(xdata.shape))
        xdata = xdata.reshape(-1, self.batch_size, 2)
        ydata = ydata.reshape(-1, self.batch_size, 2)
        self.x_batches = np.vsplit(xdata, self.num_batches)
        self.y_batches = np.vsplit(ydata, self.num_batches)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
