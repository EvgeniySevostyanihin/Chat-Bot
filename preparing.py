from Vocabulary import *
import torch
from itertools import zip_longest


indexesFromSentence = lambda sentence: [Vocabulary.word2index[word] \
                                        for word in sentence.split(' ')] + [EOS_token]

zeroPadding = lambda indexes: list(zip_longest(*indexes, fillvalue=PAD_token))


def binary_matrix(indexes):
    matrix = []

    for ind, seq in enumerate(indexes):
        matrix.append([])
        for token in seq:
            matrix[ind].append(token != PAD_token)

    return matrix


# принимает батч из data и переводит в индексы словаря, при этом допольняет нулями,
# также возвращает список длин всех предложений из батча, это будет нужно позже

def input_var(batch):
    indexes_batch = [indexesFromSentence(sentence) for sentence in batch]

    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zeroPadding(indexes_batch)
    pad_var = torch.LongTensor(pad_list)

    return pad_var, lengths


# принимает батч из data и переводит в индексы словаря, при этом дополняет нулями.
# также возвращает бинарную карту пустых значений и максимальную длину предложений

def output_var(batch):
    indexes_batch = [indexesFromSentence(sentence) for sentence in batch]

    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = zeroPadding(indexes_batch)
    mask = binary_matrix(pad_list)
    mask = torch.BoolTensor(mask)
    pad_var = torch.LongTensor(pad_list)

    return pad_var, mask, max_target_len


def batch2train_data(batch):
    batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)

    inp_batch, out_batch = [], []
    for pair in batch:
        inp_batch.append(pair[0])
        out_batch.append(pair[1])

    inp, lenghts = input_var(inp_batch)
    out, mask, max_target_lenght = output_var(out_batch)

    return inp, lenghts, out, mask, max_target_lenght
