import os
from io import open
import torch


class Dictionary(object):

  def __init__(self):
    self.word2idx = {}
    self.idx2word = []

  def add_word(self, word):
    if word not in self.word2idx:
      self.idx2word.append(word)
      self.word2idx[word] = len(self.idx2word) - 1
    return self.word2idx[word]

  def __len__(self):
    return len(self.idx2word)


class Corpus(object):

  def __init__(self, path, max_token=None):
    self.dictionary = Dictionary()
    self.last = set()
    self.counter = {}
    self.add_dict(os.path.join(path, 'train.txt'))
    self.add_dict(os.path.join(path, 'valid.txt'))
    self.add_dict(os.path.join(path, 'test.txt'))
    words = [(v, k) for k, v in self.counter.items()]
    words.sort(reverse=True)

    if max_token is not None:
      if max_token < len(words):
        max_token -= 1
      for i, item in enumerate(words):
        if (i < max_token):
          self.dictionary.add_word(item[1])
        else:
          self.last.add(item[1])
      self.dictionary.add_word("<ONCE WORD>")
    else:
      for i, item in enumerate(words):
        self.dictionary.add_word(item[1])

    self.train = self.tokenize(os.path.join(path, 'train.txt'))
    self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
    self.test = self.tokenize(os.path.join(path, 'test.txt'))

  def add_dict(self, path):
    assert os.path.exists(path)
    # Add words to the dictionary

    with open(path, 'r', encoding="utf8") as f:
      for line in f:
        words = line.split() + ['<eos>']
        for word in words:
          if word in self.counter:
            self.counter[word] += 1
          else:
            self.counter[word] = 1

  def tokenize(self, path):
    """Tokenizes a text file."""
    assert os.path.exists(path)

    # Tokenize file content
    with open(path, 'r', encoding="utf8") as f:
      idss = []
      for line in f:
        words = line.split() + ['<eos>']
        ids = []
        for word in words:
          if word in self.last:
            word = "<ONCE WORD>"
          ids.append(self.dictionary.word2idx[word])
        idss.append(torch.tensor(ids).type(torch.int64))
      ids = torch.cat(idss)
    return ids
