import json
import torch
from itertools import zip_longest
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# извлечение данных
with open('data/complete', 'r', encoding='utf-8') as data:
    data = json.load(data)

PAD_token = 0
SOS_token = 1
EOS_token = 2


# Класс Voc это словарь для загрузки и урезки данных, из
# этого класса можно узнать колличество слов,
# и индексы слов


class Voc:
    def __init__(self):
        self.trimmed = False
        self.word2index, self.word2count = {}, {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def trim(self, min_count):

        if self.trimmed:
            return

        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print(f'выбранных слов {len(keep_words)} / {len(self.word2index)}')

        self.word2index, self.word2count = {}, {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.add_word(word)


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


# Encoder
EMBEDDING_SIZE = 128


class Encoder(nn.Module):
    def __init__(self, embedding):
        super(Encoder, self).__init__()

        self.embedding = embedding
        self.gru = nn.GRU(EMBEDDING_SIZE, EMBEDDING_SIZE, num_layers=2,
                          dropout=0.4, bidirectional=True)
        # будет двухнаправленный, тоесть складывает матрицы
        # с конца и с начала, так больше профита даётся

    def forward(self, input_seq, length, hidden=None):
        embed = self.embedding(input_seq)

        packed = pack_padded_sequence(embed, length)  # это запоковаем

        output, hidden = self.gru(packed, hidden)

        output, _ = pad_packed_sequence(output)  # это распаковывает

        # здесь складываются 2 прохода (с конца и с начала)
        output = output[:, :, EMBEDDING_SIZE:] + output[:, :, :EMBEDDING_SIZE]

        return output, hidden


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.dot_score = lambda hidden, encod: torch.sum(hidden * encod, dim=2)

    def forward(self, hidden, encod):
        result = self.dot_score(hidden, encod)

        return nn.functional.softmax(result.t(), dim=1).unsqueeze(1)


class AttentionDecoder(nn.Module):
    def __init__(self, embedding):
        super(AttentionDecoder, self).__init__()

        self.embedding = embedding

        self.dropout = nn.Dropout(0.4)

        self.gru = nn.GRU(EMBEDDING_SIZE, EMBEDDING_SIZE,
                          num_layers=2, dropout=0.4)

        self.attention = Attention()

        # из контакста + выхода с енкодера, получаю вектор размера EMBEDDING_SIZE
        self.concat = nn.Linear(EMBEDDING_SIZE * 2, EMBEDDING_SIZE)
        # из получившегося веткора получаю вектор весов для всех слов,
        # что само по себе предсказание следующего слова
        self.predict = nn.Linear(EMBEDDING_SIZE, embedding.num_embeddings)

    def forward(self, word, hidden, encoder_outputs):
        embed = self.embedding(word)
        embed = self.dropout(embed)

        out, hidden = self.gru(embed, hidden)

        # тут вступает в роль attention
        attention_weight = self.attention(out, encoder_outputs)

        # произвожу матричное умножение, для 3ёх мерной матрицы
        context = attention_weight.bmm(encoder_outputs.transpose(0, 1))

        # делаю одинаковые матрицы
        out, context = out.squeeze(0), context.squeeze(1)

        # конкатенирую
        out = torch.cat((out, context), 1)
        out = self.concat(out)
        out = torch.tanh(out)

        # предсказываю следующее слово
        out = self.predict(out)
        out = nn.functional.softmax(out, dim=1)

        return out, hidden
