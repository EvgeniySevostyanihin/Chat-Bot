import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os


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

        self.epochs = 0

    def forward(self, input_seq, length, hidden=None):
        embed = self.embedding(input_seq)

        packed = pack_padded_sequence(embed, length)  # это запоковаем

        output, hidden = self.gru(packed, hidden)

        output, _ = pad_packed_sequence(output)  # это распаковывает

        # здесь складываются 2 прохода (с конца и с начала)
        output = output[:, :, EMBEDDING_SIZE:] + output[:, :, :EMBEDDING_SIZE]

        return output, hidden


class AttentionDecoder(nn.Module):
    def __init__(self, embedding):
        super(AttentionDecoder, self).__init__()

        self.embedding = embedding

        self.dropout = nn.Dropout(0.4)

        self.gru = nn.GRU(EMBEDDING_SIZE, EMBEDDING_SIZE,
                          num_layers=2, dropout=0.4)

        # из контекста + выхода с енкодера, получаю вектор размера EMBEDDING_SIZE
        self.concat = nn.Linear(EMBEDDING_SIZE * 2, EMBEDDING_SIZE)
        # из получившегося веткора получаю вектор весов для всех слов,
        # что само по себе предсказание следующего слова
        self.predict = nn.Linear(EMBEDDING_SIZE, embedding.num_embeddings)

    def attention(self, hidden, encod):

        result = torch.sum(hidden * encod, dim=2)
        return nn.functional.softmax(result.t(), dim=1).unsqueeze(1)

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


# Загрузка модели, после сохраниния в функции train
def load_models():
    # тк каждые 1000 итираций сохраняются по 3 файла
    name = ((len(os.listdir('models')) // 3) - 1) * 1000

    encoder = torch.load(f'models/encoder{name}')
    decoder = torch.load(f'models/decoder{name}')
    embedding = torch.load(f'models/embedding{name}')

    return encoder, decoder, embedding