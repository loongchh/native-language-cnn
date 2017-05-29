import argparse
import os

import nltk
from nltk.util import ngrams
from string import ascii_lowercase, punctuation, digits
from itertools import product
from tqdm import tqdm
import pickle


def preprocess(args):
    transcript_dir = os.path.join("data/speech_transcriptions", args.data,
                                  "tokenized")
    file_list = os.listdir(transcript_dir)

    if args.ngram:
        ngram_dir = os.path.join("data/features/speech_transcriptions/ngrams",
                                 str(args.ngram), args.data)
        ngram_list = product(ascii_lowercase + punctuation + digits,
                             repeat=args.ngram)
        ngram_dict = {i: ng for (i, ng) in enumerate(ngram_list)}
        ngram_rev_dict = {ng: i for (i, ng) in ngram_dict.items()}
        pickle_path = os.path.join("data/features/speech_transcriptions/ngrams",
                                   str(args.ngram), 'ngram_dict.pkl')
        with open(pickle_path, 'wb') as fpkl:
            pickle.dump((ngram_dict, ngram_rev_dict), fpkl)

    if args.arpabet:
        cmu_rev_dict = nltk.corpus.cmudict.dict()
        arpabet_dir = os.path.join("data/features/speech_transcriptions/arpabets/",
                                   args.data)
        arpabet_list = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D',
                        'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY',
                        'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R',
                        'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
        arpabet_dict = {i: a for (i, a) in enumerate(arpabet_list)}
        arpabet_rev_dict = {a: i for (i, a) in arpabet_dict.items()}
        pickle_path = os.path.join("data/features/speech_transcriptions/arpabets/",
                                   'arpabet_dict.pkl')
        with open(pickle_path, 'wb') as fpkl:
            pickle.dump((arpabet_dict, arpabet_rev_dict), fpkl)

        def get_arpabet_idx(arpabet):
            if arpabet[-1].isdigit():
                arpabet = arpabet[:-1]
            return arpabet_rev_dict[arpabet]

    for fn in tqdm(file_list):
        fullpath = os.path.join(transcript_dir, fn)
        with open(fullpath, 'r') as fp:
            if args.ngram:
                f_ngram = open(os.path.join(ngram_dir, fn), 'w')
            if args.arpabet:
                f_arpabet = open(os.path.join(arpabet_dir, fn), 'w')

            for line in fp:
                if args.ngram:
                    ngram = ngrams(''.join(line.lower().split()), n=args.ngram)
                    # for ng in ngram:
                    #     if ng not in ngram_rev_dict:
                    #         print(line)
                    #         continue
                    seq = [ngram_rev_dict[ng] for ng in ngram if ng in ngram_rev_dict]
                    f_ngram.writelines(' '.join(str(i) for i in seq) + '\n')

                if args.arpabet:
                    arpabets = [cmu_rev_dict[word][0] for word in line.split()
                                if word in cmu_rev_dict]

                    seq = [get_arpabet_idx(ab) for wd in arpabets for ab in wd]
                    f_arpabet.writelines(' '.join(str(i) for i in seq) + '\n')

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
