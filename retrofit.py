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
import gzip
import math
import re
import sys
from copy import deepcopy

import numpy

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


def read_word_vecs(fname):
    '''Read all the word vectors and normalize them'''
    wvecs = {}
    if fname.endswith('.gz'):
        f_open = gzip.open
    else:
        f_open = open

    with f_open(fname) as f:
        for idx, line in enumerate(f):
            row = line.strip().lower().split()
            if idx == 0 and len(row) == 2:
                _, _dim = row
            else:
                word = row[0]
                wvecs[word] = numpy.zeros(len(row) - 1, dtype=float)
                for index, vec_val in enumerate(row[1:]):
                    wvecs[word][index] = float(vec_val)
                # normalize weight vector
                wvecs[word] /= math.sqrt((wvecs[word]**2).sum() + 1e-6)

    print("Vectors read from: " + fname)
    return wvecs


def print_word_vecs(wvecs, out_fname):
    '''Write word vectors to file'''
    print("Writing down the vectors in: " + out_fname)
    with open(out_fname, 'w') as out_f:
        for word in wvecs:
            out_f.write(word + ' ')
            for val in wvecs[word]:
                # out_f.write("{:f} ".format(val))
                out_f.write("{:f} ".format(val))
            out_f.write("\n")


def read_lexicon(fname):
    '''Read the PPDB word relations as a dictionary'''
    lexicon = {}
    with open(fname) as f:
        for line in f:
            words = line.lower().strip().split()
            lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon


def retrofit(wvecs, lexicon, n_iters):
    '''Retrofit word vectors to a lexicon'''
    new_wvecs = deepcopy(wvecs)
    wv_voc = set(new_wvecs.keys())
    loop_voc = wv_voc.intersection(set(lexicon.keys()))
    for _ in range(n_iters):
        # loop through every node also in ontology (else just use data estimate)
        for word in loop_voc:
            ctxt_words = set(lexicon[word]).intersection(wv_voc)
            n_ctxt = len(ctxt_words)
            # no neighbor, pass - use data estimate
            if n_ctxt == 0:
                continue
            # the weight of the data estimate if the number of neighbors
            new_vec = n_ctxt * wvecs[word]
            # loop over neighbors and add to new vector (currently with weight 1)
            for pp_wrd in ctxt_words:
                new_vec += new_wvecs[pp_wrd]
            new_wvecs[word] = new_vec / (2 * n_ctxt)
    return new_wvecs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help="Input word vecs")
    parser.add_argument('-l', '--lexicon', required=True,
                        help="Lexicon file name")
    parser.add_argument('-o', '--output', required=True,
                        help="Output word vecs")
    parser.add_argument('-n', '--n_iters', type=int, default=10, required=True,
                        help="Num iterations")
    args = parser.parse_args()

    wvecs = read_word_vecs(args.input)
    lexicon = read_lexicon(args.lexicon)
    n_iters = int(args.n_iters)
    out_fname = args.output

    # Enrich the word vectors using ppdb and print the enriched vectors
    print_word_vecs(retrofit(wvecs, lexicon, n_iters), out_fname)


if __name__ == '__main__':
    main()
