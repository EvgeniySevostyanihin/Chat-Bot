import string
import json


# Ниже код находит диалог, тоесть последовательность строк начинающихся с "-". При чём даже если между ними 2 абзаца,
# без диалога.

def find_dialog(text, start_dialog: str):
    convert = [[]]
    i = 0

    for a in range(len(text)):

        if text[a] in convert[-1]:
            continue

        if text[a][:len(start_dialog)] == start_dialog:
            convert.append([])

            for j in range(a, len(text)):
                if text[j][:len(start_dialog)] == start_dialog:
                    convert[-1].append(text[j])
                    i = 0
                else:
                    try:
                        if (text[j + 1][:len(start_dialog)] == start_dialog) \
                                or (text[j + 2][:len(start_dialog)] == start_dialog) \
                                or (text[j + 3][:len(start_dialog)] == start_dialog):
                            continue
                        break
                    except:
                        pass

    return convert[1:]


complete = []

for i in range(1, 29):
    text = open('data/' + str(i) + '.txt').read()
    text = text.split('\n')
    text = find_dialog(text, '— ')
    complete += text

i = 0
for n in complete:
    i += len(n)

modification = lambda x: x if x not in '.!?][()' else ','

# всего 938 диалгов, в сумме 17138 предложений, в датасете по фильмам (он на английском) побольше.

# тут происходит отделение ненужных слов автора, довольно оригинальным способом.

for index, a in enumerate(complete):
    a2 = []
    for sentence in a:
        a2.append(''.join(modification(x) for x in sentence))
    complete[index] = a2

for i in range(len(complete)):
    for j, n in enumerate(complete[i]):
        complete[i][j] = ''.join([x for i, x in enumerate(n.split(', —')) if not i % 2])

pairs = []

for dialog in complete:
    for n in range(len(dialog)):
        try:
            pair = [dialog[n], dialog[n + 1]]
            pairs.append(pair)
        except Exception:
            pass

old_pairs = pairs.copy()
pairs = []

modification = lambda x: x if x not in string.punctuation + ' — «»' else ' '

for pair in old_pairs:
    inp, out = pair
    inp = inp[2:].lower()
    out = out[2:].lower()
    inp = ''.join(modification(x) for x in inp)
    out = ''.join(modification(x) for x in out)
    pair = [inp, out]
    pairs.append(pair)

# убрал все ненужные знаки припенания и перевёл в нижний регистр

with open("data/complete", "w", encoding="utf-8") as file:
    json.dump(pairs, file)
