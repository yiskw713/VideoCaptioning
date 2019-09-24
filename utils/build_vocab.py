import argparse
import json
import os
import pickle
import spacy
import tqdm

from collections import Counter

nlp = spacy.load('en_core_web_sm')


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        for word in ['<pad>', '<start>', '<end>', '<unk>']:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_msrvtt_vocab(msrvtt_json_file, threshold=4):
    with open(msrvtt_json_file, "r") as f:
        data = json.load(f)

    sents = data['sentences']

    # count word
    counter = Counter()
    n_sentences = len(sents)

    for i in tqdm.tqdm(range(n_sentences)):
        cap = sents[i]['caption']
        cap = nlp(cap.lower())
        tokens = [token.text for token in cap]
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', type=str,
                        help='path for train annotation file')
    parser.add_argument('--save_dir', type=str, default='./data',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--file_name', type=str, default='vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()

    return args


def main():
    args = get_arguments()

    vocab = build_msrvtt_vocab(
        args.json_file, threshold=args.threshold)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    with open(os.path.join(args.save_dir, args.file_name), 'wb') as f:
        pickle.dump(vocab, f)

    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(
        os.path.join(args.save_dir, args.file_name))
    )


if __name__ == "__main__":
    main()
