from __future__ import print_function
import copy
from collections import Counter, defaultdict
from functools import reduce
from sacremoses.util import pairwise


class SubwordTokenizer(object):
    """
    This is a Python port of the Subword NMT from
    https://github.com/rsennrich/subword-nmt
    """
    
    def __init__(self, filename):
        self.vocab = self.get_vocabulary(filename)
        (self.stats, self.indices) = self.get_pair_statistics()
        self.big_stats = copy.deepcopy(self.stats)
    
    def get_vocabulary(self, filename, is_dict=False):
        vocab = Counter()
        with open(filename) as fin:
            if is_dict:
                for line in fin:
                    (word, count) = line.strip().split(' ')
                    vocab[word] += int(count)
            else:
                vocab.update(fin.read().split())
        vocab = Counter({tuple(k[:-1]) + (k[-1] + '\ue000', ): v for (k, v) in vocab.items()})
        return vocab.most_common()
    
    def get_pair_statistics(self):
        """Count frequency of all symbol pairs, and create index"""
        stats = Counter()
        indices = defaultdict(lambda: Counter())
        for (i, (word, freq)) in enumerate(self.vocab):
            for (prev, curr) in pairwise(word):
                stats[(prev, curr)] += freq
                indices[(prev, curr)][i] += 1
        return (stats, indices)
    
    def modify_token(self, token, pair):
        """
        From https://stackoverflow.com/a/40367074/610569
            >>> modify_token(('s', 'h', 'e', 'r', 'l', 'o', 'c', 'k'), ('h', 'e'))
            ('S', 'he', 'r', 'l', 'o', 'c', 'k')
        """
        (first, second) = pair
        pair_str = ''.join(pair).replace('\\', '\\\\')
        f = lambda acc, e: (acc[:-1] + (pair_str, ) if (acc[-1] == first and e == second) else acc + (e, ))
        return reduce(f, token[1:], (token[0], ))
    
    def replace_pair(self, pair):
        """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
        changes = []
        for (j, freq) in self.indices[pair].items():
            if freq < 1:
                continue
            (word, freq) = self.vocab[j]
            new_word = self.modify_token(word, pair)
            self.vocab[j] = (new_word, freq)
            changes.append((j, new_word, word, freq))
        return changes
    
    def update_pair_statistics(self, pair, changed):
        """
        Minimally update the indices and frequency of symbol pairs
        if we merge a pair of symbols, only pairs that overlap with occurrences
        of this pair are affected, and need to be updated.
        """
        self.stats[pair] = 0
        self.indices[pair] = Counter()
        (first, second) = pair
        new_pair = first + second
        for (j, word, old_word, freq) in changed:
            i = 0
            while True:
                try:
                    i = old_word.index(first, i)
                except ValueError:
                    break
                if (i < len(old_word) - 1 and old_word[i + 1] == second):
                    if i:
                        prev = old_word[i - 1:i + 1]
                        self.stats[prev] -= freq
                        self.indices[prev][j] -= 1
                    if i < len(old_word) - 2:
                        if (old_word[i + 2] != first or i >= len(old_word) - 3 or old_word[i + 3] != second):
                            nex = old_word[i + 1:i + 3]
                            self.stats[nex] -= freq
                            self.indices[nex][j] -= 1
                    i += 2
                else:
                    i += 1
            i = 0
            while True:
                try:
                    i = word.index(new_pair, i)
                except ValueError:
                    break
                if i:
                    prev = word[i - 1:i + 1]
                    self.stats[prev] += freq
                    self.indices[prev][j] += 1
                if (i < len(word) - 1 and word[i + 1] != new_pair):
                    nex = word[i:i + 2]
                    self.stats[nex] += freq
                    self.indices[nex][j] += 1
                i += 1
    
    def learn(self, num_symbols, min_freq=2, jump=1, is_dict=None):
        threshold = max(self.stats.values()) / 10
        for i in range(num_symbols):
            most_freq_tokens = self.stats.most_common(jump)
            for (token, count) in most_freq_tokens:
                changes = self.replace_pair(token)
                self.update_pair_statistics(token, changes)
                self.stats[token] = 0
                if self.stats[token] < min_freq:
                    return


