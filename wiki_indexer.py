# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division

from glob import glob
import sys
import codecs
import pickle
from os.path import expanduser, basename
from BeautifulSoup import BeautifulSoup
import nltk
import numpy as np
from multiprocessing import Process, Manager
from gensim import corpora

#TODO: Switch to absolute imports and figure out a way to take these variables as arguments
home = expanduser("~")
data_dir = home+"/data/wiki/extracted"
code_dir = home+"/workspace/relation-extraction"

from os import chdir
chdir(code_dir+"/src")


def run(all_args, target_func, func_args=(), process_limit=8):
    #TODO: Should we make this a celery task? Probably not if this code should be executable by default
    """
    This function is a wrapper around multiprocessing to parallelize processes
    """
    count = 0
    mul_factor = 100.0/len(all_args)
    for i in xrange(0, len(all_args), process_limit):
        cur_args = all_args[i:i + process_limit]
        processes = []
        for arg in cur_args:
            cur_func_args = (arg, ) + func_args
            p = Process(target=target_func, args=cur_func_args)
            processes.append(p)
            p.start()

        count += process_limit
        sys.stdout.write("Progress: {0}%\r".format(count*mul_factor))
        sys.stdout.flush()

        for p in processes:
            p.join()


#The following set of functions help you to index and retrieve data from
#the wikipedia dump as extracted by wikipedia_extractor.

def page_indexer_instance(wiki_folder_arg, page_index, lock):
    index = {}
    files = glob(data_dir + "/" + wiki_folder_arg + "/wiki*")

    for f in files:
        print f
        data = codecs.open(f, 'r', 'utf-8').readlines()
        size = len(data)
        step = 100
        for ind in xrange(0, size, step):
            try:
                soup = BeautifulSoup("".join(data[ind:ind + step]))
            except (UnicodeDecodeError, UnicodeEncodeError):
                print 'Some part of', f, 'has some encoding issues and will be skipped'
                continue
            pages = soup.findAll('doc')
            for page in pages:
                page_title = page.attrs[2][1]
                index[page_title.lower()] = wiki_folder_arg + "/" + basename(f)
    with lock:
        page_index.update(index)
    return


def build_page_index(folders, process_limit=8):
    """
    After we run wiki_extractor script on the wiki dump, we get a folder structure.
    Given one folder (eg: AA), this function builds an index of pages. The index's
    keys are page titles, and values are the file path from the given folder, which
    contain that page.

    Eg: {"carnatic music": "AA/wiki_35", "t. m. krishna": "BD/wiki_89" ... }
    """
    manager = Manager()
    page_index = manager.dict()
    lock = manager.Lock()

    run(folders, page_indexer_instance, (page_index, lock), process_limit=process_limit)

    return page_index


def merge_indexes(index_files):
    """
    Give a list of indexes (basically pickled files with dictionaries),
    this handy function merges all of them and returns a single index.
    """
    index = {}
    for f in index_files:
        print f
        part_index = pickle.load(file(f))
        index.update(part_index)

    return index


def get_page_content(page_title, wiki_index, textonly=True):
    """
    Gets the plain text/BeautifulSoup object of a given wikipedia page.

    :param page_title: the title of wikipedia page in lower case with underscores replaced with spaces.
    :param wiki_index: the complete index built using 'build_page_index' function
    :param textonly: If this is set to true, just the plain text is returned, with everything else removed. Otherwise,
    it returns a BeautifulSoup object with the contents of the page as extracted using wikipedia_extractor.py.
    """
    page_title = page_title.lower()
    if page_title not in wiki_index.keys():
        return ""

    file_path = data_dir + "/" + wiki_index[page_title]

    data = codecs.open(file_path, 'r', 'utf-8').read()
    try:
        soup = BeautifulSoup(data)
    except (UnicodeEncodeError, UnicodeDecodeError):
        return ""

    pages = soup.findAll('doc')
    for page in pages:
        if page.attrs[2][1].lower() == page_title:
            if textonly:
                return " ".join(page.findAll(text=True))
            else:
                return page


def group_pages_by_file(pages, wiki_index):
    """
    A convenience function that can be used to avoid opening the same
    file multiple pages which are all in the same file.
    """
    file_page_index = {}

    for page in pages:
        if page in wiki_index.keys():
            if wiki_index[page] in file_page_index.keys():
                file_page_index[wiki_index[page]].append(page)
            else:
                file_page_index[wiki_index[page]] = [page]

    return file_page_index


#The following classes and functions build some basic indexes useful in text-analysis.

class WikiTokenizer():
    def __init__(self, pages, wiki_index):
        self.pages = pages
        self.wiki_index = wiki_index
        self.stemming = True
        self.stopword_removal = True

        self.alphabetic_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.stemmer = nltk.stem.snowball.SnowballStemmer("english")

        self.current_page = ""
        self.content = ""
        self.tokens = []

    def __iter__(self):
        for page in self.pages:
            self.current_page = page
            self.tokenize()
            yield self.current_page, self.tokens

    def tokenize(self, page=None):
        #Tokenize the text to words
        if page:
            self.current_page = page
        if self.content == "":
            self.content = get_page_content(self.current_page, self.wiki_index)

        tokenized_text = [self.alphabetic_tokenizer.tokenize(s) for s in nltk.sent_tokenize(self.content)]
        tokenized_text = np.concatenate(tokenized_text)

        #Do stemming and remove stopwords
        if self.stemming:
            tokenized_text = [self.stemmer.stem(w) for w in tokenized_text if
                              not w in nltk.corpus.stopwords.words('english')]
        elif self.stopword_removal:
            tokenized_text = [w for w in tokenized_text if not w in nltk.corpus.stopwords.words('english')]

        rare_tokens = set(w for w in set(tokenized_text) if tokenized_text.count(w) <= 1)
        short_tokens = set(w for w in set(tokenized_text) if len(w) <= 1)
        #tokenized_text = [w for w in tokenized_text if w not in list(short_tokens)]
        tokenized_text = [w for w in tokenized_text if w not in list(rare_tokens) + list(short_tokens)]

        self.tokens = tokenized_text


def tokenizer_instance(page, wiki_index, token_index, lock):
    tokenizer = WikiTokenizer([page], wiki_index)
    tokenizer.tokenize(page)
    with lock:
        token_index[page] = tokenizer.tokens

    return


def build_token_index(page_titles, wiki_index, process_limit=8):
    manager = Manager()
    token_index = manager.dict()
    lock = manager.Lock()

    page_titles = [i.lower() for i in page_titles]

    run(page_titles, tokenizer_instance, (wiki_index, token_index, lock), process_limit=process_limit)

    return token_index


def link_indexer_instance(page, wiki_index, link_index, lock):
    page = page.lower()
    soup = get_page_content(page, wiki_index, textonly=False)
    if not soup:
        return
    links = soup.findAll("a")
    link_terms = [link.text.lower() for link in links]

    with lock:
        link_index[page] = np.unique(link_terms).tolist()

    return


def build_link_index(page_titles, wiki_index, process_limit=16):
    manager = Manager()
    link_index = manager.dict()
    lock = manager.Lock()

    page_titles = [i.lower() for i in page_titles]

    run(page_titles, link_indexer_instance, (wiki_index, link_index, lock), process_limit=process_limit)

    return link_index


def all_indexer_instance(page, wiki_index, link_index, token_index, lock):
    page = page.lower()
    soup = get_page_content(page, wiki_index, textonly=False)
    if not soup:
        return
    links = soup.findAll("a")
    link_terms = [link.text.lower() for link in links]

    content = " ".join(soup.findAll(text=True))
    tokenizer = WikiTokenizer([page], wiki_index)
    tokenizer.content = content
    tokenizer.tokenize(page)

    with lock:
        link_index[page] = np.unique(link_terms).tolist()
        token_index[page] = tokenizer.tokens
    return


def build_all_indexes(page_titles, wiki_index, process_limit=4):
    """
    This function runs link_indexer and token_indexer together in a single loop,
    instead of making one (costly!) pass for each using seperate functions.
    """
    manager = Manager()
    link_index = manager.dict()
    token_index = manager.dict()
    lock = manager.Lock()

    page_titles = [i.lower() for i in page_titles]

    run(page_titles, all_indexer_instance, (wiki_index, link_index, token_index, lock), process_limit=process_limit)

    return {'link_index': link_index, 'token_index': token_index}


def build_lsa_index(token_index, f_name=None):
    dictionary = corpora.Dictionary(tokens for page, tokens in token_index.items())
    corpus = [dictionary.doc2bow(tokens) for page, tokens in token_index.items()]

    if f_name is None:
        return dictionary, corpus
    else:
        dictionary.save(home+'/workspace/relation-extraction/data/'+f_name+'.dict')
        corpora.MmCorpus.serialize(home+'/workspace/relation-extraction/data/'+f_name+'.mm', corpus)


if __name__ == "__main__":
    pass
    # all_args = sys.argv[1:]

    # Build page index
    # run(all_args, build_page_index, (), process_limit=8)

    # Build Link Indexes
    # run(all_args, build_link_index, (["hindustani", "music"], ), process_limit=8)

    # Merge Indexes
    # folder = sys.argv[1].strip("/")
    # files = glob(code_dir + "/data/" + folder + "/*.pickle")
    # whole_index = merge_indexes(files)
    # pickle.dump(whole_index, file(code_dir + "/data/" + folder + ".pickle", "w"))

    # Build Content Indexes
    # run(all_args, build_content_index, (["jazz", "music"], 30, "bigrams", True), process_limit=8)

