import json
import codecs
import numpy as np
from scipy import sparse

def read_json(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file, encoding='utf-8')
    return data

def write_to_json(data, output_filename, indent=2, sort_keys=False):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, indent=indent, sort_keys=sort_keys)

def load_sparse(input_filename):
    npy = np.load(input_filename)
    coo_matrix = sparse.coo_matrix((npy['data'], (npy['row'], npy['col'])), shape=npy['shape'])
    return coo_matrix.tocsc()

def save_sparse(sparse_matrix, output_filename):
    assert sparse.issparse(sparse_matrix)
    if sparse.isspmatrix_coo(sparse_matrix):
        coo = sparse_matrix
    else:
        coo = sparse_matrix.tocoo()
    row = coo.row
    col = coo.col
    data = coo.data
    shape = coo.shape
    np.savez(output_filename, row=row, col=col, data=data, shape=shape)

def log(logfile, text, write_to_log=True):
    if write_to_log:
        with codecs.open(logfile, 'a', encoding='utf-8') as f:
            f.write(text + '\n')

def read_text(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
        lines_list = []
        for line in lines:
            lines_list.append(int(line.strip()))
    return lines_list

def write_list_to_text(lines, output_filename, add_newlines=True, add_final_newline=False):
    if add_newlines:
        lines = '\n'.join(lines)
        if add_final_newline:
            lines += '\n'
    else:
        lines = ''.join(lines)
        if add_final_newline:
            lines[-1] += '\n'

    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.writelines(lines)