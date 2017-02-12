import re
import codecs
import collections
import nltk

word_pattern = re.compile("[A-z0-9]+")
punct_pattern = re.compile("\s|[-.\"',;?:/()&]")

WORD_MODE = "WORD"
PUNCTUATION_MODE = "PUNCTUATION"


def get_mode(c):
    if word_pattern.match(c):
        return WORD_MODE
    if punct_pattern.match(c):
        return PUNCTUATION_MODE
    return None

def custom_parse(data):
    lst = []
    current = ""
    mode = None
    for c in data:
        new_mode = get_mode(c)
        if mode is WORD_MODE and new_mode == WORD_MODE:
            current += c
        else:
            if current:
                lst.append(current)
            current = c
            mode = new_mode
    return lst

if __name__ == "__main__":
    with codecs.open("data/guthrie_words/input.txt", "r", encoding="utf-8") as f:
        data = f.read()
    clean_data = custom_parse(data.lower())
    counter = collections.Counter(clean_data)
    print(counter)


