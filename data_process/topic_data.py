import os
import re
import codecs
import copy
from collections import Counter
import numpy as np
from scipy import sparse
from spacy.lang.en import English
import config.file_handing as fh

def load_and_process_data(infile, vocab_size, parser, vocab=None, log_transform=None, label_list=None):
    with codecs.open('/data/lengjia/topic_model/mallet_stopwords.txt', 'r', encoding='utf-8') as input_file:
        mallet_stopwords = input_file.readlines()
    mallet_stopwords = { s.strip() for s in mallet_stopwords}

    with codecs.open(infile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    n_items = len(lines)
    print("Parsing %d documents" % n_items)

    parsed = []
    labels = []
    word_counts = Counter()

    for i, line in enumerate(lines):

        if i % 1000 == 0:
            print(i)
        _, _, label, text = line.strip().split("\t\t")
        labels.append(label)
        text = re.sub('<[^>]+>', '', text)
        parse = parser(text)
        words = [re.sub('\s', '', token.orth_) for token in parse]
        words = [word.lower() for word in words if len(word) >= 1]
        words = [word for word in words if len(word) <= 20]

        words = [word for word in words if word not in mallet_stopwords]
        #         words = [word for word in words if re.match('^[a-zA-A]*$', word) is not None]
        words = [word for word in words if re.match('[a-zA-A0-9]', word) is not None]
        # convert numbers to a number symbol
        words = [word for word in words if re.match('[0-9]', word) is None]
        words = [word for word in words if re.search('@', word) is None]  ##delete string with @
        parsed.append(words)
        word_counts.update(words)

    print("Size of full vocabulary=%d" % len(word_counts))

    if vocab is None:
        initial_vocab = {}
        vocab = copy.copy(initial_vocab)
        #         for w in word_counts.most_common(len(word_counts))[3258:]:
        for w in word_counts.most_common(vocab_size):
            if w[0] not in vocab:
                vocab[w[0]] = len(vocab)
                if len(vocab) >= vocab_size:
                    break
        total_words = np.sum(list(vocab.values()))
        word_freqs = np.array([vocab[v] for v in vocab.keys()]) / float(total_words)  # 词频
    else:
        word_freqs = None

    if label_list is None:
        label_list = list(set(labels))
        label_list.sort()
    n_labels = len(label_list)
    label_index = dict(zip(label_list, range(n_labels)))

    X = np.zeros([n_items, vocab_size], dtype=int)
    y = []
    lists_of_indices = []
    delect_index = []
    counter = Counter()
    print("Converting to count representations")
    count = 0
    total_tokens = 0
    for i, words in enumerate(parsed):
        indices = [vocab[word] for word in words if word in vocab]
        word_subset = [word for word in words if word in vocab]
        counter.clear()
        counter.update(indices)
        if len(counter.keys()) > 0:
            values = list(counter.values())
            if log_transform:
                values = np.array(np.round(np.log(1 + np.array(values, dtype='float'))), dtype=int)
            X[np.ones(len(counter.keys()), dtype=int) * count, list(counter.keys())] += values
            total_tokens += len(word_subset)
            y_vector = np.zeros(n_labels)
            y_vector[label_index[labels[i]]] = 1
            y.append(y_vector)
            lists_of_indices.append(indices)
            count += 1
        else:
            delect_index.append(str(i))

    print("Found %d non-empty documents" % count)
    print("Total tokens = %d" % total_tokens)

    # drop the items that don't have any words in the vocabualry
    X = np.array(X[:count, :], dtype=int)
    X_indices = X.copy()
    X_indices[X_indices > 0] = 1
    print(X.shape)
    temp = np.array(y)
    y = np.array(temp[:count], dtype=int)
    print(y.shape)
    sparse_y = sparse.csr_matrix(y)
    sparse_X = sparse.csr_matrix(X)
    sparse_X_indices = sparse.csr_matrix(X_indices)

    return sparse_X, vocab, lists_of_indices, sparse_y, word_freqs, label_list, sparse_X_indices, delect_index

def preprocess_data(train_infile, test_infile, output_dir, vocab_size, log_transform=False):
    print("Loading Spacy")
    parser = English()

    with codecs.open(train_infile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    train_n_items = len(lines)
    train_indices = list(set(range(train_n_items)))
    with codecs.open(test_infile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    test_n_items = len(lines)
    test_indices = list(set(range(test_n_items)))  # [0,1,...,n_items-1]

    train_X, train_vocab, train_indices, train_y, word_freqs, label_list, train_X_indices, train_delect_index = load_and_process_data(
        train_infile, vocab_size, parser, log_transform=log_transform)
    test_X, _, test_indices, test_y, _, _, test_X_indices, test_delect_index = load_and_process_data(test_infile,
                                                                                                     vocab_size, parser,
                                                                                                     vocab=train_vocab,
                                                                                                     log_transform=log_transform,
                                                                                                     label_list=label_list)
    fh.save_sparse(train_X, os.path.join(output_dir, 'train.npz'))
    fh.write_to_json(train_vocab, os.path.join(output_dir, 'train.vocab.json'))
    fh.write_to_json(train_indices, os.path.join(output_dir, 'train.indices.json'))
    fh.save_sparse(train_y, os.path.join(output_dir, 'train.labels.npz'))
    fh.save_sparse(train_X_indices, os.path.join(output_dir, 'train_X_indices.npz'))
    fh.write_list_to_text(train_delect_index, os.path.join(output_dir, 'train_delect_index.txt'))

    fh.save_sparse(test_X, os.path.join(output_dir, 'test.npz'))
    fh.write_to_json(test_indices, os.path.join(output_dir, 'test.indices.json'))
    fh.save_sparse(test_y, os.path.join(output_dir, 'test.labels.npz'))
    fh.save_sparse(test_X_indices, os.path.join(output_dir, 'test_X_indices.npz'))
    fh.write_list_to_text(test_delect_index, os.path.join(output_dir, 'test_delect_index.txt'))

    n_labels = len(label_list)
    label_dict = dict(zip(range(n_labels), label_list))
    fh.write_to_json(label_dict, os.path.join(output_dir, 'train.label_list.json'))
    fh.write_to_json(list(word_freqs.tolist()), os.path.join(output_dir, 'train.word_freq.json'))
