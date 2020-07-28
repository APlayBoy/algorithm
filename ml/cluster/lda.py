import numpy as np
from collections import OrderedDict

class Document:

    def __init__(self):
        self.words = []
        self.length = 0

class DataPreProcessing:

    def __init__(self):
        self.docs_count = 0
        self.words_count = 0
        self.docs = []
        self.word2id = OrderedDict()
        np.random.uniform()

class LDAModel: