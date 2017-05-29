import argparse
import os

import torch.utils.data as data
import pandas as pd


def train(args):
    train_labels = pd.read_csv(args.train_label)
    dev_labels = pd.read_csv(args.dev_label)
    vocab_size =


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--max-length', type=int, default=500,
                        help='maximum feature length for each document')
    parser.add_argument('--embed-dim', type=int, default=500,
                        help='dimension of the feature embeddings')
    parser.add_argument('--channel', type=int, default=500,
                        help='number of channel output for each CNN layer')
    parser.add_argument('--train', type=str, default='data/features/speech_transcriptions/ngrams/2/train',
                        help='directory containing speech transcriptions features for the train set')
    parser.add_argument('--dev', type=str, default='data/features/speech_transcriptions/ngrams/2/dev',
                        help='directory containing speech transcriptions features for the dev set')
    parser.add_argument('--train-label', type=str, default='data/labels/train/labels.train.csv',
                        help='CSV of the train set labels')
    parser.add_argument('--dev-label', type=str, default='data/labels/dev/labels.dev.csv',
                        help='CSV of the dev set labels')
    parser.add_argument('--gpu', action='store_true',
                        help='using GPU-enabled CUDA Variables')
    args = parser.parse_args()
    train(args)
