# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import unicode_literals
from os.path import expanduser, basename
from gensim import models, similarities
from math import sqrt, floor
import networkx as nx
from collections import Counter
import pickle
import sys


home = expanduser("~")
data_dir = home+"/data/wiki/extracted"
code_dir = home+"/workspace/relation-extraction"

from os import chdir
chdir(code_dir+'/src')
import progressbar
import wiki_indexer as wi
reload(wi)


def ochiai_coefficient(x, y):
    x = set(x)
    y = set(y)
    return len(x.intersection(y))/(sqrt(len(x)*len(y)))


def graph_content(content_index, weight_thresh):
    pages = content_index.keys()
    n = len(pages)
    g = nx.Graph()
    
    total_calc = floor(n*(n-1)/2)
    mul_factor = 100.0/total_calc
    count = 0
    for i in xrange(0, n):
        for j in xrange(i+1, n):
            x = set(content_index[pages[i]]+[pages[i]])
            y = set(content_index[pages[j]]+[pages[j]])
            
            #Ochiai coefficient, in this case is equal to Cosine similarity
            if len(x) == 0 or len(y) == 0:
                weight = 0
            else:
                weight = ochiai_coefficient(x, y)
            if weight >= weight_thresh:
                g.add_edge(pages[i], pages[j], weight=weight)
        count += (n-i-1)
        sys.stdout.write("Progress: {0}%\r".format(count*mul_factor))
        sys.stdout.flush()
    return g


def graph_hyperlinks(link_index):
    """
    This function builds a graph with pages as nodes. Only those pages which have
    keyword specific to a given music style are considered. The links refer to the
    hyperlinks in Wikipedia content of the page.
    """
    g = nx.DiGraph()
    rel_pages = link_index.keys()

    for page in rel_pages:
        linked_pages = set(link_index[page]).intersection(rel_pages)
        for linked_page in linked_pages:
            g.add_edge(page, linked_page)

    return g


def graph_entitylinks(files, fileindex, thresh=0.9):
    """
    This function builds a graph with entity linked data. The sum of weights of all out going
    edges for each node is normalized to be 1.

    The threshold parameter defines the minimum confidence value with which an entity is linked.
    """
    g = nx.DiGraph()
    progress = progressbar.ProgressBar(len(files))
    for i in xrange(len(files)):
        f = files[i]
        entity = basename(f)[:-7]
        entity = fileindex[int(entity)]
        res = pickle.load(file(f))
        if 'Resources' not in res.keys():
            continue
        uris = [r['@URI'] for r in res['Resources'] if r['@similarityScore'] >= thresh]
        counter = Counter(uris)
        n = len(uris)
        for uri in counter.keys():
            linked_entity = uri.split('/')[-1].replace('_', ' ').lower()
            if entity == linked_entity:
                #print entity, linked_entity, type(entity), type(linked_entity)
                continue
            g.add_edge(entity, linked_entity, {'weight': counter[uri]/n})
        progress.animate(i)
    return g


def graph_cocitation(hyperlinks_g):
    cocitation_g = nx.Graph()
    nodes = hyperlinks_g.nodes()
    for i in xrange(len(nodes)):
        x = [k[0] for k in hyperlinks_g.in_edges(nodes[i])]
        if len(x) == 0:
            continue
        for j in xrange(i, len(nodes)):
            y = [k[0] for k in hyperlinks_g.in_edges(nodes[j])]
            if len(y) == 0:
                continue
            weight = ochiai_coefficient(x, y)
            if weight > 0:
                cocitation_g.add_edge(nodes[i], nodes[j], {"weight": weight})
    return cocitation_g


def graph_bibcoupling(hyperlinks_g):
    bibcoupling_g = nx.Graph()
    nodes = hyperlinks_g.nodes()
    for i in xrange(len(nodes)):
        x = [k[1] for k in hyperlinks_g.out_edges(nodes[i])]
        if len(x) == 0:
            continue
        for j in xrange(i, len(nodes)):
            y = [k[1] for k in hyperlinks_g.out_edges(nodes[j])]
            if len(y) == 0:
                continue
            weight = ochiai_coefficient(x, y)
            if weight > 0:
                bibcoupling_g.add_edge(nodes[i], nodes[j], {"weight": weight})
    return bibcoupling_g


def graph_lsa(token_index, num_topics=200, num_neighbors=50, sim_thresh=0.25):
    page_titles = token_index.keys()
    dictionary, corpus = wi.build_lsa_index(token_index)

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    index = similarities.MatrixSimilarity(lsi[corpus_tfidf])

    lsa_g = nx.DiGraph()

    for i in xrange(len(corpus)):
        sims = index[lsi[corpus[i]]]
        sims[i] = -1
        sim_index = list(enumerate(sims))
        sim_index = sorted(sim_index, key=lambda x: x[1], reverse=True)
        for j, sim_value in sim_index[:num_neighbors]:
            if sim_value >= sim_thresh:
                lsa_g.add_edge(page_titles[i], page_titles[j], {"weight": float(sim_value)})
    return lsa_g

if __name__ == "__main__":
    pass