#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
Retrofit
========

This tool is used to post-process word vectors to incorporate knowledge
from semantic lexicons (Faruqui et al, 2014).

Authors
-------
Manaal Faruqui, <mfaruqui@cs.cmu.edu>

Modified by:
Marc Evrard, <marc.evrard@gmail.com>

License
-------
Copyright 2014 Manaal Faruqui

Licensed under the GNU General Public License, version 2 (the "License")
https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html

Example
```````
    ./retrofit.py  -i data/vectors_big_16_no_unk_tag.npy -l lexicons/ppdb-xl.txt -n 10
'''

import argparse
import os
import re

import numpy as np
# from sklearn.preprocessing import normalize as sk_normalize

import embedding_tools as emb
import print_tools as prn


IS_NUMBER = re.compile(r'\d+.*')
IS_NON_WORD = re.compile(r'\W+')


def norm_word(word):
    '''Normalize lexicon words'''
    if IS_NUMBER.search(word.lower()):
        return '<num>'
    elif IS_NON_WORD.sub('', word) == '':
        return '<punc>'
    else:
        return word.lower()


def read_lexicon(fname):
    '''Read the PPDB word relations as a dictionary'''
    lexicon = {}
    with open(fname) as f:
        for line in f:
            words = line.lower().strip().split()
            lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon


def retrofit(id2word, embeds, lexicon, n_iters):
    '''Retrofit word vectors to a lexicon'''
    new_embeds = np.copy(embeds)
    word2id = {word: idx for idx, word in enumerate(id2word)}
    wv_vocab = set(id2word)
    loop_voc = list(wv_vocab & set(lexicon))
    print("Ratio of word presence: {:.0f}% in embs, {:.0f}% in lex."
          "".format(len(loop_voc) / len(id2word) * 100,
                    len(loop_voc) / len(lexicon) * 100))
    # print('len(lexicon)**:', len(lexicon))
    # print('len(id2word)**:', len(id2word))

    for itr_idx in range(n_iters):
        # loop through every node also in ontology (else just use data estimate)
        for word in loop_voc:
            ctxt_words = set(lexicon[word]) & wv_vocab
            n_ctxt = len(ctxt_words)
            # no neighbor, pass - use data estimate
            if n_ctxt == 0:
                continue
            # print('**', idx, word, ctxt_words)
            # the weight of the data estimate is the number of neighbors
            new_vec = n_ctxt * embeds[word2id[word]]
            # loop over neighbors and add to new vector (currently with weight 1)
            for pp_wrd in ctxt_words:
                new_vec += new_embeds[word2id[pp_wrd]]
            new_embeds[word2id[word]] = new_vec / (2 * n_ctxt)

        prn.progress_bar(itr_idx, n_iters)
    print()

    return new_embeds


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_fpath', required=True,
                        help="Input word vecs")
    parser.add_argument('-l', '--lexicon', required=True,
                        help="Lexicon file name")
    parser.add_argument('-o', '--out_fpath',
                        help="Output word vecs")
    parser.add_argument('-n', '--n_iters', type=int, default=10, required=True,
                        help="Num iterations")
    argp = parser.parse_args()

    # wvecs = read_word_vecs(argp.in_fpath)
    id2word, embeds_arr = emb.load_embeds_np(argp.in_fpath)
    # embeds_arr = sk_normalize(embeds_arr, axis=1)   # TODO: remove!?
    lexicon = read_lexicon(argp.lexicon)

    # Enrich the word vectors using ppdb and print the enriched vectors
    embeds_out = retrofit(id2word, embeds_arr, lexicon, argp.n_iters)

    if argp.out_fpath:
        out_fpath = argp.out_fpath
    else:
        out_fpath = os.path.join(
            'embeddings', os.path.splitext(os.path.basename(argp.in_fpath))[0] + '_rtf')

    emb.save_embeds_np(id2word, embeds_out, out_fpath)
    # print_word_vecs(embeds_out, argp.out_fpath)


if __name__ == '__main__':
    main()
