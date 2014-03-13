# -*- coding: utf-8 -*-

import pickle
import codecs
import re
from unidecode import unidecode

#TODO: Switch to absolute imports
import wiki_indexer as wi
reload(wi)


class Page():
    def __init__(self, title):
        self.title = title
        self.content = ""

        #sentence filtering settings
        self.min_words = 2
        self.min_chars = 50

        #Coreference filtering
        self.ref_max_len = 50

        self.resolved_sentences = {}
        self.raw_sentences = []

    def set_content(self, wiki_index):
        self.content = wi.get_page_content(self.title, wiki_index)

    def serialize(self, path):
        pickle.dump(self, file(path, 'w'))

    def serialize_content(self, path):
        codecs.open(path, "w", "utf-8").write(self.content)

    def clean_content(self):
        if self.content == "":
            return
        #Remove braces and quotes
        braces = re.findall('\(.*?\)', self.content)
        braces.extend(re.findall('\[.*?\]', self.content))
        for b in braces:
            self.content = self.content.replace(b, '')
        self.content = self.content.replace('"', '')
        self.content = self.content.replace("'", '')

        #Discard the lines too short
        lines = self.content.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > self.min_chars and len(line.split(' ')) > self.min_words:
                clean_lines.append(line.strip('.'))

        self.content = unidecode('. '.join(clean_lines))