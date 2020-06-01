import torch
from model import Voc
import json
from itertools import zip_longest


PAD_token = 0
SOS_token = 1
EOS_token = 2


# извлечение данных
with open('data/complete', 'r', encoding='utf-8') as data:
    data = json.load(data)


# Создание словаря

Vocabulary = Voc()

for pair in data:
    for sentence in pair:
        Vocabulary.add_sentence(sentence)


# Обрезание словаря

def trim_rare_words(voc, pairs, MIN_COUNT=2):
    voc.trim(MIN_COUNT)

    keep_pairs = []

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]

        keep = True

        for inp, out in zip(input_sentence.split(' '), output_sentence.split(' ')):
            if inp not in voc.word2index or out not in voc.word2index:
                keep = False

        if keep:
            keep_pairs.append(pair)

    print(f'Было {len(pairs)}, стало {len(keep_pairs)}, {len(keep_pairs) / len(pairs)}')

    return keep_pairs


def load_voc(path="data/trimmed_data"):
    import json

    with open(path, 'r', encoding='utf-8') as data:
        data = json.load(data)

    Vocabulary = Voc()

    for pair in data:
        for sentence in pair:
            Vocabulary.add_sentence(sentence)

    return data, Vocabulary


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
# также возвращает список длин всех предложений из батча, это нужно для pack_padded_sequence

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

