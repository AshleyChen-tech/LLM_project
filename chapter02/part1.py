from importlib.metadata import version

print("torch version", version("torch"))
print("tiktoken version:", version("tiktoken"))


import os
import urllib.request

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])


import re
# text = "Hello, world. This, is a text."
text = "Hello, world. Is this-- a text?"
# result = re.split(r'(\s)', text)
# result = re.split(r'([,.]|\s)', text)
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)

# Strip whitespace from each item and then filter out any empty strings.
# result = [item for item in result if item.strip()]
result = [item.strip() for item in result if item.strip()]
print(result)


preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip()for item in preprocessed if item.strip()]
print(preprocessed[:30])
print(len(preprocessed))

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
content = tokenizer.decode(ids)
print(content)
en_comtent = tokenizer.decode(tokenizer.encode(text))
print(en_comtent)

