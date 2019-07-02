from config.file_handing import log
import numpy as np
import re
import time
ISOTIMEFORMAT='%Y-%m-%d %X'
from subprocess import Popen, PIPE
import torch
import torch.nn as nn
from model.loss import loss_function_topic as loss_function_topic,loss_function_share


def print_topics(model, vocab, log_file, topic_num, write_to_log=True):
    vocab = {value:key for key,value in vocab.items()}
    n_topics = model.d_t
    if n_topics > 1:
        log(log_file, "Topics:", write_to_log)
        weights = model.de.weight.data.cpu().numpy()
        for j in range(n_topics):
            order = list(np.argsort(weights[:, j]).tolist()) #返回的是数组值从小到大的索引值
            order.reverse()
            highest_list = [vocab[i] for i in order[:topic_num]]
            highest = ' '.join(highest_list)
            print("%d %s" % (j, highest))
            log(log_file, "%s" % highest,  write_to_log)

def get_reward_cv(model, vocab, log_file, topic_file, topic_num, cuda):
    vocab = {value: key for key, value in vocab.items()}
    n_topics = model.d_t
    cv_list = []
    topic_list = []
    if n_topics > 1:
        weights = model.de.weight.detach().cpu().numpy()
        for j in range(n_topics):
            order = list(np.argsort(weights[:, j]).tolist())
            order.reverse()
            highest_list = [vocab[i] for i in order[:topic_num]]
            topic_list.append(highest_list)
        f = open(topic_file, 'w')
        for topic in topic_list:
            for word in topic:
                f.write(word + ' ')
            f.write('\n')
        f.close()
        p = Popen(['/data/lengjia/jdk1.8.0_201/bin/java', '-jar',
                   '/data/lengjia/topic_model/palmetto-0.1.0-jar-with-dependencies.jar',
                   '/data/lengjia/topic_model/wiki/wikipedia_bd/', 'C_V', topic_file], stdin=PIPE, stdout=PIPE,
                  stderr=PIPE)
        output, err = p.communicate(b"input data that is passed to subprocess' stdin")
        temp_list = output.decode("utf-8").split("\n")
        for line in temp_list:
            if re.search(r"\t(.*)\t", line):
                co_herence_cv = re.search(r"\t(.*)\t", line)
                cv = float(co_herence_cv.group(1))
                cv_list.append(cv)
        cv_vector = torch.Tensor(cv_list).unsqueeze(0).cuda(cuda)
    return cv_vector


def evaluate(log_file, model_file, model, loss_function_clf, test_batch_clf, test_X_topic, test_indices,
             test_index_arrays, cuda):
    model.train(False)
    total_loss = 0
    total_hit = 0
    total_sample = 0
    batch_i = 0
    no_hit_index_list = []
    true_label_list = []
    predicted_label_list = []
    #     test_X_topic = normalize(np.array(test_X_topic, dtype='float32'), axis=1)
    #     n_items, dv = test_X_topic.shape
    bound = 0
    batch_num = 0

    for batch in test_batch_clf:
        data, target, length, original_index = batch[0], batch[1], batch[2], batch[3]
        if cuda is None:
            data_var_list = [torch.LongTensor(chunk) for chunk in data]
            target_var = torch.LongTensor(target)
            length_var = torch.LongTensor(length)
            x = torch.from_numpy(test_X_topic[original_index[0]:original_index[-1] + 1])
            x_indices = torch.from_numpy(test_indices[original_index[0]:original_index[-1] + 1])
        else:
            data_var_list = [(torch.LongTensor(chunk).cuda(cuda)) for chunk in data]
            target_var = (torch.LongTensor(target).cuda(cuda))
            length_var = (torch.LongTensor(length))
            x = (torch.from_numpy(test_X_topic[original_index[0]:original_index[-1] + 1]).cuda(cuda))
            x_indices = (torch.from_numpy(test_indices[original_index[0]:original_index[-1] + 1]).cuda(cuda))
        #         index_arrays = test_index_arrays[original_index[0]:original_index[-1]+1]
        mean, logvar, p_x_given_h, predicted_target, word_attention_dict = model(
            x, x_indices, data_var_list, length_var, cuda)
        loss_clf = loss_function_clf(predicted_target, target_var)
        _, predicted_label = torch.max(predicted_target, dim=1)
        no_hit_mask = predicted_label.data != target_var.data
        no_hit_mask = no_hit_mask.cpu()
        no_hit_index = torch.masked_select(torch.arange(predicted_label.data.size(0)), no_hit_mask).tolist()
        no_hit_index = np.asarray(no_hit_index, dtype=np.int32)
        total_hit += torch.sum(predicted_label.data == target_var.data)
        total_loss += loss_clf.item()
        total_sample += data[0].shape[1]
        batch_i += 1
        no_hit_index_list.append(original_index[no_hit_index])
        true_label_list.append(target)
        predicted_label_array = np.asarray(predicted_label.cpu().data.tolist())
        predicted_label_list.append(predicted_label_array)

        loss_topic, nll_term, KLD, penalty = loss_function_topic(x, mean, logvar, p_x_given_h, x_indices)
        #         print(nll_term)

        counts_list = []
        for i in x:
            counts_list.append(torch.sum(i).detach().cpu())
        if np.mean(counts_list) != 0:
            bound += (loss_topic.detach().cpu().numpy() / np.array(counts_list)).mean()
            batch_num += 1

    not_hit_index_array = np.concatenate(no_hit_index_list)
    acc = float(total_hit) / float(total_sample)
    returned_document_list = [test_batch_clf.sample_document(index) for index in original_index]

    bound = np.exp(bound / float(batch_num))
    print("Estimated perplexity upper bound on test set = %0.3f" % bound)
    log(log_file, "Estimated perplexity upper bound on test set = %0.3f" % bound)

    return total_loss / (
                batch_i + 1), acc, not_hit_index_array, target, predicted_label_array, returned_document_list, bound

def train_model(log_file, model_file, model, optimizer, optimizer_attention, loss_function_clf, loss_function_attention,
                train_batch_clf, test_batch_clf, vocab,
                train_X_topic, train_indices, train_index_arrays, test_X_topic, test_indices, test_index_arrays,
                max_epochs, clf_w, tm_w, topic_num, temp_batch_num, maxmin, topic_file, cuda=None):
    print("Start Tranining")
    log(log_file, "Start Tranining")
    if cuda != None:
        model.cuda(cuda)
    # normalize input vectors
    #     train_X_topic = normalize(np.array(train_X_topic, dtype='float32'), axis=1)
    epochs_since_improvement = 0
    min_bound = np.inf
    _, dv = train_X_topic.shape
    cv_list = []

    for epoch_i in range(max_epochs):
        print("\nEpoch %d" % epoch_i)
        print(time.strftime(ISOTIMEFORMAT, time.localtime()))
        log(log_file, "\nEpoch %d" % epoch_i)
        log(log_file, time.strftime(ISOTIMEFORMAT, time.localtime()))

        temp_batch_index = 0
        for train_batch in train_batch_clf:
            model.train(True)
            train_data, train_target, length_data, index_range = train_batch[0], train_batch[1], train_batch[2], train_batch[3]
            if cuda is None:
                train_data_var_list = [(torch.LongTensor(chunk)) for chunk in train_data]
                train_target_var = (torch.LongTensor(train_target))
                length_var = (torch.LongTensor(length_data))
                x = (torch.from_numpy(train_X_topic[index_range[0]:index_range[-1] + 1]))
                x_indices = (torch.from_numpy(train_indices[index_range[0]:index_range[-1] + 1]))
            else:
                train_data_var_list = [(torch.LongTensor(chunk).cuda(cuda)) for chunk in train_data]
                train_target_var = (torch.LongTensor(train_target).cuda(cuda))
                length_var = (torch.LongTensor(length_data))
                x = (torch.from_numpy(train_X_topic[index_range[0]:index_range[-1] + 1]).cuda(cuda))
                x_indices = (torch.from_numpy(train_indices[index_range[0]:index_range[-1] + 1]).cuda(cuda))

            mean, logvar, p_x_given_h, predicted_train_target, word_attention_dict= model(
                x, x_indices, train_data_var_list, length_var, cuda)
            optimizer.zero_grad()
            loss_clf = loss_function_clf(predicted_train_target, train_target_var)  # batch_size
            loss_topic, nll_term, KLD, penalty = loss_function_topic(x, mean, logvar, p_x_given_h, x_indices)
            loss = clf_w * loss_clf + tm_w * loss_topic
            loss.mean().backward()
            nn.utils.clip_grad_norm(model.continuous_parameters(), max_norm=5)
            optimizer.step()

            mean, logvar, p_x_given_h, predicted_train_target, word_attention_dict= model(
                x, x_indices, train_data_var_list, length_var, cuda)
            if len(word_attention_dict) != 0:
                optimizer_attention.zero_grad()
                loss_attention = loss_function_share(word_attention_dict, model, loss_function_attention, maxmin)
                loss_attention.backward()
                nn.utils.clip_grad_norm(model.continuous_parameters(), max_norm=5)
                optimizer_attention.step()

            temp_batch_index += 1
            if temp_batch_index % temp_batch_num == 0:
                print("\nepoch batch nll_term KLD l1p loss")
                print("%d %d %0.4f %0.4f %0.4f %0.4f" % (
                epoch_i, temp_batch_index, nll_term.mean(), KLD.mean(), penalty, loss_topic.mean()))
                print("loss_attention:", loss_attention.item())
                log(log_file, "\nepoch batch nll_term KLD l1p loss")
                log(log_file, "%d %d %0.4f %0.4f %0.4f %0.4f" % (
                epoch_i, temp_batch_index, nll_term.mean(), KLD.mean(), penalty, loss_topic.mean()))
                log(log_file, "%0.4f" % loss_attention.item())
                # train_loss,train_acc = evaluate(model,loss_function,train_batch_generator,cuda=cuda)
                train_loss, train_acc = 0, 0
                test_loss, test_acc, wrong_index, true_label_array, predicted_label_array, document_list, bound = evaluate(
                    log_file, model_file, model, loss_function_clf, test_batch_clf, test_X_topic, test_indices,
                    test_index_arrays, cuda)

                if bound < min_bound:
                    print("New best dev bound = %0.3f" % bound)
                    log(log_file, "New best dev bound = %0.3f" % bound)
                    min_bound = bound
                    print("Saving model")
                    torch.save(model.state_dict(), model_file)
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    print("No improvement in %d epoch(s)" % epochs_since_improvement)
                    log(log_file, "No improvement in %d epoch(s)" % epochs_since_improvement)

                print("Batch :%d\ttrain_loss:%0.6f\ttrain_acc:%0.6f\ntest_loss:%0.6f\ttest_acc:%0.6f" % (
                temp_batch_index, train_loss, train_acc, test_loss, test_acc))
                log(log_file, "Batch\t train_loss\t train_acc\t test_loss\t test_acc")
                log(log_file, "%d\t %0.6f\t %0.6f\t %0.6f\t %0.6f" % (
                temp_batch_index, train_loss, train_acc, test_loss, test_acc))
                print("True : %d \t Predicted : %d" % (true_label_array[0], predicted_label_array[0]))
                log(log_file, "True : %d " % true_label_array[0])
                log(log_file, "Predicted : %d" % predicted_label_array[0])
                # error_analysis(test_batch_generator, wrong_index, predicted_label_array, true_label_array)

        batch_cv = get_reward_cv(model, vocab, log_file, topic_file,topic_num,cuda)
        print('co_herence_cv: ' + str(torch.mean(batch_cv).detach().cpu().numpy()))
        log(log_file, "co_herence_cv: %0.3f" % float(float(torch.mean(batch_cv).detach().cpu().numpy())))
        cv_list.append(torch.mean(batch_cv).detach().cpu().numpy())
        ###converge  topicmodel:20   clfmodel:6
        if epochs_since_improvement >= 20 and epoch_i >= 6:
            break

    print("The best dev bound = %0.3f" % min_bound)
    log(log_file, "The best dev bound = %0.3f" % min_bound)
    log(log_file, "Final topics:")
    print_topics(model, vocab, log_file, topic_num, write_to_log=True)

    return cv_list




