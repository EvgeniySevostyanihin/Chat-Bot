import json


# извлечение данных
def extract_data():
    data = open('data/complete')
    data = data.read()
    return [x.split('\t||\t') for x in data.split('\n\n')][:-1]

data = extract_data()

PAD_token = 0
SOS_token = 1
EOS_token = 2

# Класс Voc это словарь для загрузки и урезки данных, из этого класса можно узнать колличество слов,
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


data = trim_rare_words(Vocabulary, data)


with open("data/trimmed_data", "w", encoding="utf-8") as file:
    json.dump(data, file)
