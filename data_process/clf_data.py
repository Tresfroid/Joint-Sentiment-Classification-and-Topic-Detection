import copy
from collections import Counter
import numpy as np
from progressbar import ProgressBar
import config.file_handing as fh

def get_vocabulary(textfile, vocab_file, topic_vocabsize, initial_vocab={}, vocabsize=0):
    vocab_topic = fh.read_json(vocab_file)
    vocab_topic['<unk>'] = int(topic_vocabsize)
    vocab_topic['<sssss>'] = int(topic_vocabsize) + 1
    vocab = copy.copy(vocab_topic)
    word_count = Counter()
    for line in open(textfile,'r').readlines():
        _,_,label,text = line.strip().split("\t\t")
        for w in text.split(): # skip speaker indicator
            word_count[w] += 1
    # if vocabulary size is specified, most common words are selected
    if vocabsize > 0:
        for w in word_count.most_common(vocabsize):
            if w[0] not in vocab:
                vocab[w[0]] = len(vocab)
                if len(vocab) >= vocabsize:
                    break
    else: # all observed words are stored
        for w in word_count:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab

def load_embedding(log_file,word_id_dict,embedding_file_name,embedding_size=300):
    """ Return:
            matrix : vocab_length * embedding_size
                    the i line is the embedding of index i word
    """
    embedding_length = len(word_id_dict)
    embedding_matrix = np.random.uniform(-1e-2,1e-2,size=(embedding_length,embedding_size))
    embedding_matrix[0] = 0
    hit = 0
    with open(embedding_file_name,"r", encoding='utf-8') as f:
        for line in f:
            splited_line = line.strip().split(" ")
            word,embeddings = splited_line[0],splited_line[1:]
            if word in word_id_dict:
                word_index = word_id_dict[word]
                embedding_array = np.fromstring("\n".join(embeddings),dtype=np.float32,sep="\n")
                embedding_matrix[word_index] = embedding_array
                hit += 1
    hit_rate = float(hit)/embedding_length
    print(("The hit rate is {}".format(hit_rate)))
    fh.log(log_file, "The hit rate is : %6f" % hit_rate)
    return embedding_matrix

def convert_words2ids(words, vocab, unk, max_value, sos=None, eos=None):
    id_list = [ vocab[w] if w in vocab else unk for w in words ]
    if sos is not None:
        id_list.insert(0, sos)
    if eos is not None:
        id_list.append(eos)
    return id_list[:max_value]

def load_clf(textfile, vocab, max_value, max_utterance, delect_index):
    """ Load a dialog text corpus as word Id sequences
        Args:
            textfile (str): filename of a dialog corpus
            vocab (dict): word-id mapping
        Return:
            list of dialogue : dialogue is (input_id_list,output_id_list)
    """
    document_list = []
    label_list = []

    def filter_key(sent):
        unk_count = sent.count(vocab['<unk>'])
        return unk_count / len(sent) < 0.3

    with open(textfile, "r", encoding='utf-8') as f:
        line_list = f.readlines()
        line_len = len(line_list)
        progressbar = ProgressBar()
        word_list_buffer = []
        for i, line in enumerate(line_list):
            if (i not in delect_index):
                _, _, label, text = line.strip().split("\t\t")
                sent_list = text.strip().split("<sssss>")
                sent_list = [sent.strip().split(" ") for sent in sent_list]
                sent_id_list = [convert_words2ids(sent, vocab, max_value=max_value, unk=vocab['<unk>']) for sent in
                                sent_list]
                new_sent_id_list = []
                previous_sent = []
                for sent in sent_id_list:
                    if len(previous_sent) != 0:
                        new_sent = previous_sent + sent
                    else:
                        new_sent = sent
                    if len(new_sent) < 3:
                        previous_sent = new_sent
                    else:
                        new_sent_id_list.append(new_sent)
                        previous_sent = []
                if len(previous_sent) > 0:
                    new_sent_id_list.append(previous_sent)
                if len(new_sent_id_list) > 0:
                    document_list.append(new_sent_id_list[:max_utterance])
                    label_list.append(int(label))

    # document_list [[[each word index in one clause ],[each word index in one clause ]],[[],[]],..]  doc-clause
    # label_list [3,5,...] each doc label
    def sort_key(document_with_label):
        document = document_with_label[0]
        first_key = len(document)  # The first key is the number of utterance of input
        second_key = np.max(
            [len(utterance) for utterance in document])  # The third key is the max number of word in input
        third_key = np.mean(
            [len(utterance) for utterance in document])  # The third key is the max number of word in input
        return first_key, second_key, third_key

    index_list = [i for i in range(len(document_list))]
    document_original_list = document_list
    label_original_list = label_list
    document_with_label_list = list(zip(*[document_list, label_list, index_list]))
    document_with_label_list = sorted(document_with_label_list, key=sort_key)
    document_list, label_list, index_list = list(zip(*document_with_label_list))
    return document_list, label_list, index_list


