import torch
import sys

def loss_function_topic(x, mean, logvar, p_x_given_h, indices):
    KLD = -0.5 * torch.sum(1 + logvar - (mean ** 2) - torch.exp(logvar), 1)
    nll_term = -torch.sum(torch.mul(indices, torch.log(torch.mul(indices, p_x_given_h) + 1e-32)), 1)
    # nll_term = -torch.sum(torch.mul(x, torch.log(p_x_given_h + 1e-10)), 1)
    loss = nll_term + KLD
    penalty = 0.0
    return loss, nll_term, KLD, penalty

def loss_function_share(word_attention_dict, model, loss_function_attention, maxmin):
    clf_dic = sorted(word_attention_dict.items(), key=lambda x: x[0], reverse=False)
    kld_all = 0.0
    de_parameter = model.de.weight  # 2000*50

    if maxmin == 1:
        for d in clf_dic:
            index = d[0]
            HAN_attention = d[1]
            TM_attention = de_parameter[index]
            sum_h = HAN_attention.norm()
            sum_t = TM_attention.norm()
            kld_all += -torch.abs(torch.dot(TM_attention, HAN_attention)) / (sum_h ** 2 * sum_t ** 2)
    elif maxmin == 2:
        for d in clf_dic:
            index = d[0]
            HAN_attention = d[1]
            TM_attention = de_parameter[index]
            kl1 = loss_function_attention(TM_attention, HAN_attention.detach())
            kl2 = loss_function_attention(HAN_attention, TM_attention.detach())
            kld_all += -2 / (2 + kl1 + kl2)
    elif maxmin == 3:
        for d in clf_dic:
            index = d[0]
            HAN_attention = d[1]
            TM_attention = de_parameter[index]
            kl1 = loss_function_attention(TM_attention, HAN_attention.detach())
            kl2 = loss_function_attention(HAN_attention, TM_attention.detach())
            kld_all += -1 / (1 + 1 / (1 / (kl1 + 0.01) + 1 / (kl2 + 0.01)))
    elif maxmin == 4:
        for d in clf_dic:
            index = d[0]
            HAN_attention = d[1]
            TM_attention = de_parameter[index]
            sum_h = HAN_attention.norm()
            sum_t = TM_attention.norm()
            kld_all += -(torch.abs(torch.dot(TM_attention, HAN_attention)) - sum_h ** 2 - sum_t ** 2)
    else:
        print("maxmin loss function is not set!")
        sys.exit(0)

    return kld_all / len(clf_dic)
