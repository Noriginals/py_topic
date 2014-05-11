import logging
import os

from gensim import corpora, models, utils
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def iter_documents(reuters_dir):
    """Iterate over Reuters documents, yielding one document at a time."""
    for fname in os.listdir(reuters_dir):
        # read each document as one big string
        document = open(os.path.join(reuters_dir, fname)).read()
        # parse document into a list of utf8 tokens
        yield utils.simple_preprocess(document)
        
class ReutersCorpus(object):
    def __init__(self, reuters_dir):
        self.reuters_dir = reuters_dir
        self.dictionary = corpora.Dictionary(iter_documents(reuters_dir))
        stop_words = open('/home/evan/Documents/mallot/pov/english_stop.txt','r').read().split()
#       self.dictionary.filter_extremes()  # remove stopwords etc
#        self.dictionary = [word for word in self.dictionary if word not in stop_words]

    def __iter__(self):
        for tokens in iter_documents(self.reuters_dir):
            yield self.dictionary.doc2bow(tokens)
                      

# set up the streamed corpus
corpus=ReutersCorpus('/home/evan/Documents/mallot/pov/syria/')
mallet_path = '/home/evan/Mallet/bin/mallet'
model = models.LdaMallet(mallet_path, corpus, num_topics=10, id2word=corpus.dictionary)
