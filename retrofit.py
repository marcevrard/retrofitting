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
    ./retrofit.py   -i embeddings/sample_vec.txt -o out_vec.txt \
                    -l lexicons/ppdb-xl.txt -n 10
'''

import argparse
import re

import numpy as np
from sklearn.preprocessing import normalize as sk_normalize

import embedding_tools as emb


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
    loop_voc = list(set(id2word) & set(lexicon))
    print("Ratio of word presence: {:.0f}% in embs, {:.0f}% in lex."
          "".format(len(loop_voc) / len(id2word) * 100,
                    len(loop_voc) / len(lexicon) * 100))
    # print('len(lexicon)**:', len(lexicon))
    # print('len(id2word)**:', len(id2word))

    for _ in range(n_iters):
        # loop through every node also in ontology (else just use data estimate)
        for word in loop_voc:
            ctxt_words = set(lexicon[word]) & set(id2word)
            n_ctxt = len(ctxt_words)
            # no neighbor, pass - use data estimate
            if n_ctxt == 0:
                continue
            # the weight of the data estimate if the number of neighbors
            new_vec = n_ctxt * emb.emb_lkup(word, id2word, embeds)
            # loop over neighbors and add to new vector (currently with weight 1)
            for pp_wrd in ctxt_words:
                new_vec += emb.emb_lkup(pp_wrd, id2word, new_embeds)
            new_embeds[id2word.index(word)] = new_vec / (2 * n_ctxt)

    return new_embeds


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_fpath', required=True,
                        help="Input word vecs")
    parser.add_argument('-l', '--lexicon', required=True,
                        help="Lexicon file name")
    parser.add_argument('-o', '--out_fpath', required=True,
                        help="Output word vecs")
    parser.add_argument('-n', '--n_iters', type=int, default=10, required=True,
                        help="Num iterations")
    argp = parser.parse_args()

    # wvecs = read_word_vecs(argp.in_fpath)
    id2word, embeds_arr = emb.load_embeds_np(argp.in_fpath)
    embeds_arr = sk_normalize(embeds_arr, axis=1)   # TODO: remove!?
    lexicon = read_lexicon(argp.lexicon)

    # Enrich the word vectors using ppdb and print the enriched vectors
    embeds_out = retrofit(id2word, embeds_arr, lexicon, argp.n_iters)

    emb.save_embeds_np(id2word, embeds_out, argp.out_fpath)
    # print_word_vecs(embeds_out, argp.out_fpath)


if __name__ == '__main__':
    main()
