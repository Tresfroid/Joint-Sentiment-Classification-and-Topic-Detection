import numpy as np
import os
from optparse import OptionParser
import torch
import torch.nn as nn
from model.model import JointModel
from train_model import train_model
from config.file_handing import log
import config.file_handing as fh
from data_process.topic_data import preprocess_data
from data_process.clf_data import get_vocabulary,load_embedding,load_clf
from data_process.model_data import DataIter,load_topic


def main():

    ########################           option            #############################################################
    usage = "%prog input_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--train_data_file', dest='train_data_file', default="/data/lengjia/topic_model/yelp_2013_39923/yelp/yelp-2013-train.txt.ss")
    parser.add_option('--test_data_file', dest='test_data_file', default="/data/lengjia/topic_model/yelp_2013_39923/yelp/yelp-2013-test.txt.ss")
    parser.add_option('--num_label', dest='num_label', default=5)
    parser.add_option('--cuda', dest='cuda', default=0)
    parser.add_option('--maxmin', dest='maxmin', default=3)
    parser.add_option('--batchsize', dest='batchsize', default=1)
    parser.add_option('--temp_batch_num', dest='temp_batch_num', default=5000)
    parser.add_option('--dt', dest='dt', default=30, help='Number of topics: default=%default')
    parser.add_option('--vocab', dest='topic_vocabsize', default=5000)
    parser.add_option('--clf_weight', dest='clf_weight', default=9)
    parser.add_option('--tm_weight', dest='tm_weight', default=1)
    parser.add_option('--max_epochs', dest='max_epochs', default=50)
    parser.add_option('--min_epochs', dest='min_epochs', default=10)

    parser.add_option('--lr', dest='lr', default=1e-4)
    parser.add_option('--de', dest='de', default=500)
    parser.add_option('--encoder_layers', dest='encoder_layers', default=1)
    parser.add_option('--generator_layers', dest='generator_layers', default=4)
    parser.add_option('--topic_num', dest='topic_num', default=30)
    parser.add_option('--clf_vocabsize', dest='clf_vocabsize', default=15000)
    parser.add_option('--embedding_size', dest='embedding_size', default=300)
    parser.add_option('--max_value', dest='max_value', default=50)
    parser.add_option('--max_utterance', dest='max_utterance', default=25)
    parser.add_option('--padded_value', dest='padded_value', default=0)
    parser.add_option('--input', dest='input_prefix', default='train')
    parser.add_option('--output', dest='output_prefix', default='output')
    parser.add_option('--test', dest='test_prefix', default='test')
    #######################################         parameter            #########################################################
    options, args = parser.parse_args()
    input_dir = args[0]
    train_data_file = options.train_data_file
    test_data_file = options.test_data_file
    num_label= int(options.num_label)
    cuda = int(options.cuda)
    maxmin = int(options.maxmin)
    batchsize = int(options.batchsize)
    temp_batch_num = int(options.temp_batch_num)
    dt = int(options.dt)
    topic_vocabsize = int(options.topic_vocabsize)
    clf_weight = float(options.clf_weight)
    tm_weight = float(options.tm_weight)
    max_epochs = int(options.max_epochs)
    min_epochs = int(options.min_epochs)

    lr = float(options.lr)
    de = int(options.de)
    encoder_layers = int(options.encoder_layers)
    generator_layers = int(options.generator_layers)
    topic_num = int(options.topic_num)
    clf_vocabsize = int(options.clf_vocabsize)
    embedding_size = int(options.embedding_size)
    max_value = int(options.max_value)
    max_utterance = int(options.max_utterance)
    padded_value = int(options.padded_value)

    input_prefix = options.input_prefix
    output_prefix = options.output_prefix
    test_prefix = options.test_prefix
    l1_strength = np.array(0.0, dtype=np.float32)
    vocab_file = os.path.join(input_dir, input_prefix + '.vocab.json' )
    word2vec_file = "/data/lengjia/topic_model/yelp/yelp_embedding_300.txt"
    model_file = os.path.join(input_dir, 'JointModel_maxmin'+ str(maxmin) + '_batchsize' + str(batchsize) + '_vocab' + str(topic_vocabsize) + '_' + str(clf_weight) + '+' + str(tm_weight) +  '_dt' + str(dt)+ '.pkl')
    log_file = os.path.join(input_dir, 'JointModel_maxmin'+ str(maxmin) + '_batchsize' + str(batchsize) + '_vocab' + str(topic_vocabsize) + '_' + str(clf_weight) + '+' + str(tm_weight) +  '_dt' + str(dt)+ '.log')
    topic_file = os.path.join(input_dir, 'JointModel_maxmin'+ str(maxmin) + '_batchsize' + str(batchsize) + '_vocab' + str(topic_vocabsize) + '_' + str(clf_weight) + '+' + str(tm_weight) + '_dt' + str(dt)+ '.txt')


#*********************************************************************************************************************************************************

    #### topic: data->one_hot
    # preprocess_data(train_data_file, test_data_file, input_dir, topic_vocabsize, log_transform=False )
    ### clf: data->embedding
    vocab_clf = get_vocabulary(train_data_file, vocab_file, topic_vocabsize, vocabsize=clf_vocabsize)  # dict of word:id
    pretrained_embedding = None
    pretrained_embedding = load_embedding(log_file, vocab_clf, word2vec_file, embedding_size)
    train_delect_index = fh.read_text(os.path.join(input_dir, input_prefix + '_delect_index.txt'))
    test_delect_index = fh.read_text(os.path.join(input_dir, test_prefix + '_delect_index.txt'))
    train_data_clf, train_label_clf, train_index_list = load_clf(train_data_file, vocab_clf, max_value, max_utterance,train_delect_index)
    test_data_clf, test_label_clf, test_index_list = load_clf(test_data_file, vocab_clf, max_value, max_utterance,test_delect_index)
    print(len(train_data_clf), len(test_data_clf))
    train_batch_clf = DataIter(train_data_clf,train_label_clf,batchsize,0)
    test_batch_clf = DataIter(test_data_clf,test_label_clf,batchsize,0)
    #### topic: read one_hot
    train_X_topic, vocab_topic, train_labels_topic,train_indices,train_index_arrays = load_topic(train_index_list, input_dir, input_prefix, log_file)
    test_X_topic, _, test_labels_topic,test_indices,test_index_arrays = load_topic(test_index_list, input_dir, test_prefix, log_file, vocab_topic)

    # model
    model = JointModel( d_v=topic_vocabsize, d_e=de, d_t=dt, encoder_layers=encoder_layers, generator_layers=generator_layers,
        encoder_shortcut=False, generator_shortcut=False, generator_transform=None,
        num_word=clf_vocabsize, emb_size=embedding_size, word_rnn_size=300, word_rnn_num_layer=2, word_rnn_dropout=0.3,
        word_rnn_bidirectional=True,
        word_attention_size=150, context_rnn_size=150, context_rnn_dropout=0.3, context_rnn_bidirectional=True,
        context_attention_size=150, mlp_size=100, num_label=num_label, context_rnn_num_layer=1,
        pretrained_embedding=pretrained_embedding
    )
    # model = torch.nn.DataParallel(model,device_ids=[0,1,2,3])
    # model.load_state_dict(torch.load(os.path.join(input_dir + 'JointModel_maxmin1_batchsize1_vocab5000_9+1.pkl' )))
    ##train
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_attention = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function_clf = nn.CrossEntropyLoss()
    loss_function_attention = nn.KLDivLoss()
    cv_list = train_model(log_file, model_file, model, optimizer, optimizer_attention,loss_function_clf, loss_function_attention,
                                          train_batch_clf, test_batch_clf,vocab_topic,
                                          train_X_topic, train_indices, train_index_arrays,test_X_topic, test_indices, test_index_arrays,
                                          max_epochs, clf_weight,tm_weight, topic_num, temp_batch_num, maxmin,topic_file, cuda=cuda)

    max_cv = np.max(np.array(cv_list))
    log(log_file, "The best cv = %0.3f" % max_cv)

if __name__ == '__main__':
    main()