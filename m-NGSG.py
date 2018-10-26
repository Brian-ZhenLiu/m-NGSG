"""

This code is for the final project of biomedical course
Implementation of paper "protein classification using modified NGSG model"
Input file is 'fasta' format file

Student: Zhen Liu
NID: zh116855

"""
import numpy as np
from numpy import *
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn import svm, metrics

def split_data(feature, size):
    """

    This function is for integrating all features into a big list
    Transform feature list into array
    Split all feature from dataset into training set and test set in the ratio of 7:3

    """
    X = []
    Y = []
    for i in range(len(feature)):
        X.append(feature[i][1])
        Y.append(feature[i][0])
    X = np.array(X)
    Y = np.array(Y).flatten()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=1)
    return x_train, x_test, y_train, y_test

def denoise(feature, n_pos):
    """

    This function is for denoising all features whose postion are detected by function noise_pos
    """
    new_feature = []
    for i in range(len(feature)):
        f_vec = []
        for j in range(1900):
            if j not in n_pos:
                f_vec.append(feature[i][1][j])
        new_feature.append((feature[i][0], f_vec))
    return new_feature

def noise_pos(feature, thres):
    """

    This function is for creating position list
    The list contains those features occurs less than given number

    """
    n_pos = []
    for i in range(len(feature[0][1])):
        max_num = 0
        count = 0
        for j in range(len(feature)):
            max_num = max(feature[j][1][i], max_num)
            count += feature[j][1][i]
        if max_num < thres:
            n_pos.append(i)
    return n_pos

def feat_extract(aa, seq, buffer):
    """

    This function is for integrating features and tags into a list

    """
    full_feature = {}
    feature = []
    for key in seq:
        bi_gram = vec_init(seq, aa)
        full_feature[key] = feat_count(key, seq, bi_gram, buffer)
    for key in full_feature:
        feature.append((tag_extract(key), list(full_feature[key].values())))
    # print(feature)
    return feature

def feat_count(key, seq, table, buffer):
    """

    This function is for counting feature numbers including k-skip-bi-gram and relative position

    """
    acid_list = list(seq[key])
    for i in range(len(acid_list)-(buffer+1)):
        grams = skip_buf(buffer, i, acid_list)
        for gram in grams:
            table[gram] += 1
        pos = pos_cal(seq[key], i, buffer)
        temp = acid_list[i] + str(pos)
        table[temp] += 1
    return table

def pos_cal(seq, curr_pos, buffer):
    """

    Calculating relative position to C-terminus by given current position and buffer number

    """
    dist = len(seq) - curr_pos
    pos = dist + (buffer - dist)%buffer
    return pos

def skip_buf(buffer, pos, acid_list):
    """

    Adding buffer into skip-gram model

    """
    grams = []
    for i in range(1, buffer+2):
        grams.append(acid_list[pos] + acid_list[pos+i])
    return grams

def buf_num(skip_num, para_a):
    """

    This function is for calculating buffer numbers by given parameter

    """
    c = skip_num + (para_a - skip_num) % para_a
    return c

def tag_extract(str):
    """

    This function is for extracting tags from every sequence procedings

    """
    tag = ''
    for char in list(str):
        if char != '|':
            tag += char
        else:
            break
    return tag

def vec_init(seq, amino_acid):
    """

    This function is for initialize feature stats table

    """
    bi_gram = {}
    for acid_1 in amino_acid:
        for acid_2 in amino_acid:
            temp_1 = acid_1 + acid_2
            bi_gram[temp_1] = 0
    pos_list = pos_init(seq, amino_acid)
    for pos_motif in pos_list:
        bi_gram[pos_motif] = 0
    return bi_gram

def pos_init(seq, aa):
    """

    This function is for initializing realative position list

    """
    pos_list = []
    for a in aa:
        for i in range(count_len(seq) + 10):
            temp = a + str(i)
            pos_list.append(temp)
    return pos_list

def count_len(seq):
    """

    Saving the max length of sequence in the dataset

    """
    length = 0
    for key in seq:
        length = max(length, len(seq[key]))
    return length

def read_file(path):
    """

    Reading 'fasta' format file

    """
    fh = open(path)
    prot_seq = {}
    for line in fh:
        if line.startswith('>'):
            name = line.replace('>','').split()[0]
            prot_seq[name] = ''
        else:
            prot_seq[name]  += line.replace('\n','')
    fh.close()
    return prot_seq

def main(file):
    """

    aa provide all amino acids
    All function are called here to extracting features and do denoising
    The accuracy and best parameters are printed in the end

    """
    aa = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R', 'S', 'T', 'C', 'P']
    protseq = read_file(file)
    bi_gram = vec_init(protseq, aa)

    buffer = 1
    un_feature = feat_extract(aa, protseq, buffer)

    no_pos = noise_pos(un_feature, 2)
    feature = denoise(un_feature, no_pos)

    x_train, x_test, y_train, y_test = split_data(feature, 0.2)
    svc = svm.SVC(kernel='poly')
    para_grid = [
                { "C":[0.1, 1, 5, 10, 25, 50],
                  "degree":[1, 2, 3, 4, 5],
                  "gamma":[1, 0.1, 0.01, 0.001]},
                            ]
    grid = GridSearchCV(svc, param_grid=para_grid, cv=10, n_jobs=-1)
    grid.fit(x_train, y_train)
    print("The best accuracy is: " + str(metrics.accuracy_score(y_test, grid.predict(x_test))))
    print(grid.best_estimator_)

main("vir_non.fasta")
