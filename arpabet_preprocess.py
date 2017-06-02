import argparse
from os import listdir
from os.path import join

import nltk
from nltk.util import ngrams
from string import ascii_lowercase, punctuation, digits
from itertools import product
from tqdm import tqdm
import pickle
import string
import re

from collections import Counter

def add_to_words_to_list(data, vocab, punctuations):
    '''
    Get words from directory of data type
    '''
    transcript_dir = join("data/speech_transcriptions", data,
                                  "tokenized")
    file_list = listdir(transcript_dir)
    
    for fn in tqdm(file_list):
        fullpath = join(transcript_dir, fn)
        with open(fullpath, 'r') as fp:
             for line in fp:
                line = re.sub(r"<.*>", "", line) # Remove unintelligible
                tokens = line.upper().split()                
                for t in tokens:
                    if string.punctuation.find(t) != -1:
                        punctuations.add(t)
                    else:
                        vocab[t] += 1

def generate_word_list():
    '''
    Build list of words to be used with this tool: http://www.speech.cs.cmu.edu/tools/lextool.html
    to generate an arpabet dictionary.
    This enables us to have an arpabet translation for words that aren't present in CMU's
    dictionary
    NOTE: I manually run the tool on the website to generate the dictionary
    '''
    new_arpabet_dir = "data/features/speech_transcriptions/new_arpabets"
    vocab = Counter()
    punctuations = set()
    
    add_to_words_to_list("train", vocab, punctuations)
    add_to_words_to_list("dev", vocab, punctuations)
    
    vocab_list = sorted(vocab, key=vocab.get, reverse=True) 
    with open(join(new_arpabet_dir, "vocab.dat"), "w") as vocab_file:
        for w in vocab_list:
            vocab_file.write("{}\n".format(w))
    with open(join(new_arpabet_dir, "punctuations.dat"), "w") as punc_file:
        for p in punctuations:
            punc_file.write("{}\n".format(p))
            
def load_custom_arpabet_dict():
    new_arpabet_dir = "data/features/speech_transcriptions/new_arpabets"
    arpabet_dict = {':': '<SEMICOLON>', ',' : '<COMMA>', '.' : '<PERIOD>',
                    '-' : '<BAR>', '<UNK>': '<UNKNOWN>', '<SEP>': '<SEP>'}
    
    with open(join(new_arpabet_dir, "arpabet.dict"), "r") as arp_file: 
        for line in arp_file:
            word, translation = line.rstrip('\n').split('\t')
            if '(' in word: # Skip for multiple translations
                continue
            arpabet_dict[word] = translation
    return arpabet_dict    

def arpabet_translation_text(data, arpabet_dict):
    transcript_dir = join("data/speech_transcriptions", data, "tokenized")
    new_arpabet_dir = join("data/features/speech_transcriptions/new_arpabets/translations", data)
    file_list = listdir(transcript_dir)
    
    word_arpabet_dict = load_custom_arpabet_dict()
        
    for fn in tqdm(file_list):
        fullpath = join(transcript_dir, fn)
        
        with open(fullpath, 'r') as fp:
            with open(join(new_arpabet_dir, fn), 'w') as f_arpabet:
                for line in fp:
                    line = re.sub(r"<.*>", "<UNK>", line)
                    tokens = line.upper().split()
                    
                    new_tokens = [word_arpabet_dict[t] for t in tokens if t in word_arpabet_dict]
                    new_line = " <SEP> ".join(new_tokens)
                    
                    f_arpabet.write("{}\n".format(new_line))                   

def generate_arpabet_translation_text():
    arpabet_dict = load_custom_arpabet_dict()
    arpabet_translation_text("train", arpabet_dict)
    arpabet_translation_text("dev", arpabet_dict)

def generate_arpabet_translation_ids(data):
    translation_dir = join("data/features/speech_transcriptions/new_arpabets/translations", data)
    file_list = listdir(translation_dir)

    word_arpabet_dict = load_custom_arpabet_dict()
    translation_ids_dir = join("data/features/speech_transcriptions/new_arpabets/translations_ids", data)
    arpabet_list = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D',
                    'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY',
                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R',
                    'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
                    '<SEMICOLON>', '<COMMA>', '<PERIOD>', '<BAR>', '<UNKNOWN>', '<SEP>']
    arpabet_dict = {i: a for (i, a) in enumerate(arpabet_list)}
    arpabet_rev_dict = {a: i for (i, a) in arpabet_dict.items()}
    pickle_path = "data/features/speech_transcriptions/new_arpabets/dict.pkl"
    with open(pickle_path, 'wb') as fpkl:
        pickle.dump((arpabet_dict, arpabet_rev_dict), fpkl)
    print(arpabet_dict)
    print(arpabet_rev_dict)
    
    for fn in tqdm(file_list):
        fullpath = join(translation_dir, fn)
        with open(fullpath, 'r') as fp:
            with open(join(translation_ids_dir, fn), 'w') as f_arpabet:
                for line in fp:
                    arpabets_ids = [str(arpabet_rev_dict[token]) for token in line.split()]
                    f_arpabet.write('{}\n'.format(' '.join(arpabets_ids)))               