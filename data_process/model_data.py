import config.file_handing as fh
import numpy as np
import os
import gc
import copy

class DataIter(object):
    def __init__(self, document_list, label_list, batch_size, padded_value):
        self.document_list = document_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.padded_value = padded_value
        self.batch_starting_point_list = range(0, len(document_list), batch_size)

    def sample_document(self, index):
        return self.document_list[index]

    def __iter__(self):
        self.current_batch_starting_point_list = copy.copy(
            self.batch_starting_point_list)
        self.current_batch_starting_point_list = np.random.permutation(
            self.current_batch_starting_point_list)
        self.batch_index = 0
        return self

    def __next__(self):
        if self.batch_index >= len(self.current_batch_starting_point_list):
            raise StopIteration
        batch_starting = self.current_batch_starting_point_list[
            self.batch_index]
        batch_end = batch_starting + self.batch_size
        raw_batch = self.document_list[batch_starting:batch_end]
        label_batch = self.label_list[batch_starting:batch_end]

        transeposed_batch = map(list, zip(*raw_batch))
        padded_batch = []
        length_batch = []
        for transeposed_doc in transeposed_batch:
            length_list = [len(sent) for sent in transeposed_doc]
            max_length = max(length_list)
            new_doc = [sent + [self.padded_value] * (max_length - len(sent)) for sent in transeposed_doc]
            padded_batch.append(np.asarray(new_doc, dtype=np.int32).transpose(1, 0))
            length_batch.append(length_list)
        padded_length = np.asarray(length_batch)
        padded_label = np.asarray(label_batch, dtype=np.int32) - 1
        original_index = np.arange(batch_starting, batch_end)
        self.batch_index += 1
        #         print(batch_starting,batch_end)
        return padded_batch, padded_label, length_batch, original_index

def load_topic(index_list, input_dir, input_prefix, log_file, vocab=None):
    print("Loading topic data")
    temp = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz')).todense()
    X = np.array(temp, dtype='float32')
    X = np.array([X[index] for index in index_list], dtype='float32')
    del temp
    temp2 = fh.load_sparse(os.path.join(input_dir, input_prefix + '_X_indices.npz')).todense()
    indices = np.array(temp2, dtype='float32')
    indices = np.array([indices[index] for index in index_list], dtype='float32')
    del temp2
    lists_of_indices = fh.read_json(os.path.join(input_dir, input_prefix + '.indices.json'))
    index_arrays = [np.array(l, dtype='int32') for l in lists_of_indices]
    index_arrays = [index_arrays[index] for index in index_list]
    del lists_of_indices
    if vocab is None:
        vocab = fh.read_json(os.path.join(input_dir, input_prefix + '.vocab.json'))
    n_items, vocab_size = X.shape
    vocab_size == len(vocab)
    print("Loaded %d topic documents with %d features" % (n_items, vocab_size))

    label_file = os.path.join(input_dir, input_prefix + '.labels.npz')
    if os.path.exists(label_file):
        print("Loading topic labels")
        temp = fh.load_sparse(label_file).todense()
        labels = np.array(temp, dtype='float32')
    else:
        print("Label topic file not found")
        labels = np.zeros([n_items, 1], dtype='float32')
    assert len(labels) == n_items
    gc.collect()

    return X, vocab, labels, indices, index_arrays

