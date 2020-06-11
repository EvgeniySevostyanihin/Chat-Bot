import string

import torch

from prepared import load_voc, indexesFromSentence
from model import load_models

data, Vocabulary = load_voc()
encoder, decoder, embedding = load_models()
device = "cuda:0"


def greedy_search(sequence, length, maximum=40):
    # всё также как на тренировке
    encoder_out, encoder_hidden = encoder(sequence, length)

    decoder_hidden = encoder_hidden[:2]
    decoder_input = torch.ones(1, 1, device=device, dtype=torch.long)
    # здесь складываються ответы по жадному методу
    all_tokens = torch.zeros([0], device=device, dtype=torch.long)
    # всего будет 30 оборотов, при выводе отрежуться ненужные токены, а 30, чтобы больше не было
    for _ in range(maximum):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_out)

        _, decoder_input = torch.max(decoder_output, dim=1)

        all_tokens = torch.cat((all_tokens, decoder_input), dim=0)

        decoder_input = decoder_input.unsqueeze(0)

    return all_tokens


clean = lambda text: "".join(x for x in text.lower() if x not in string.punctuation)

while "Матеша идёт":
    text = input('Я > ')
    print(text)
    # ввод текста и очистка
    text = clean(text)

    # перевод в форму для енкодера
    text = torch.tensor([indexesFromSentence(text)])
    length = torch.tensor([len(text[0])])

    text = text.transpose(0, 1).long().to(device)

    text = greedy_search(text, length)

    text = [Vocabulary.index2word[token.item()] for token in text]

    print(f'Машина > {" ".join(text)}')
