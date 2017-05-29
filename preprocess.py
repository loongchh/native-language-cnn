import argparse
import os

import nltk
from nltk.util import ngrams
from string import ascii_lowercase
from itertools import product


def preprocess():
    transcript_dir = os.path.join("data/speech_transcriptions", args.data,
                                  "tokenized")
    file_list = os.listdir(transcript_dir)

    if args.ngram:
        ngram_dir = os.path.join("data/features/speech_transcriptions/ngrams",
                                 args.ngram, args.data)

        ngrams = product(ascii_lowercase + '.,', repeat=args.ngram)
        ngram_dict = {i: ng for (i, ng) in enumerate(ngrams)}
        ngram_rev_dict = {ng: i for (i, ng) in ngram_dict.items()}

    if args.arpabet:
        arpabet_dir = os.path.join("data/features/speech_transcriptions/arpabets/",
                                   args.data)
        arpabets = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', \
                    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', \
                    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', \
                    'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']

        arpabet_dict = {i: a for (i, a) in enumerate(arpabets)}
        arpabet_rev_dict = {a: i for (i, a) in arpabet_dict.items()}
        cmu_dict = nltk.corpus.cmudict.dict()

    for fn in file_list:
        fullpath = os.path.join(transcript_dir, fn)
        with open(fullpath, 'r') as fp:
            if args.ngram:
                f_ngram = open(os.path.join(ngram_dir, fn), 'w')
            if args.arpabet:
                f_arpabet = open(os.path.join(arpabet_dir, fn), 'w')

            for line in fp:
                if args.ngram:
                    ngram = ngrams(''.join(line.split()).lowercase(), n=args.ngram)
                    seq = [ngram_rev_dict[ng] for ng in ngram]
                    f_ngram.writelines(' '.join(str(i) for i in seq))

                if args.arpabet:
                    arpabets = [cmu_dict(word) for word in line.split()
                                if word in cmu_dict]
                    seq = [arpabet_rev_dict(a) for w in arpabets for a in w]
                    f_arpabet.writelines(' '.join(str(i) for i in seq))

            if args.ngram:
                f_ngram.close()
            if args.arpabet:
                f_arpabet.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--data', type=str, default='train',
                        help='dataset division that is to be processed')
    parser.add_argument('--ngram', type=int, default=None,
                        help='generate n-gram features')
    parser.add_argument('--arpabet', action='store_true',
                        help='generate arpabet phoneme features')
    args = parser.parse_args()
    preprocess(args)
