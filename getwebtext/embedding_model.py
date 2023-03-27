import os
import re

import numpy as np
import yaml
import fasttext
from vae_topic_model import VAETopic
from string_utils import vec2mat_cos_sim
import dill as pickle


def save(model, path):
    if path.endswith('.bin'):
        model.save_model(path)
    filehandler = open(path, 'wb')
    pickle.dump(model, filehandler)
    return

def load(path):
    if path.endswith('.bin'):
        return fasttext.load_model(path)
    filehandler = open(path, 'rb')
    return pickle.load(filehandler)

def read_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


class Embbedding_model():
    def __init__(self, config_file):
        self.global_emb = None
        self.local_emb = {}
        self.config = read_config(config_file)

    def fit_global(self, words_list):
        return

    def fit_local(self, words_list, c):
        return

    def global_embedding_matrix(self):
        return

    def global_vocab(self):
        return

    def local_embedding_matrix(self,c):
        return

    def local_vocab(self,c):
        return

    def dist_func(self,x1,x2):
        return


class fasttext_embedding_model(Embbedding_model):
    def __init__(self, config_file):
        super(fasttext_embedding_model, self).__init__(config_file)

    def fit_global(self, words_list):
        all_words = [word for keywords in words_list for word in keywords]
        f=open('data_model/model/fttrain_global','w')
        str_decode =' '+' '.join(all_words).encode('ascii', 'ignore').decode()
        print(str_decode[:20])
        del all_words
        f.write(str_decode)
        f.close()
        self.global_emb = fasttext.train_unsupervised(
            'data_model/model/fttrain_global',
            dim=self.config['global']['n_topics']
        )
        self.global_emb.save_model(self.config['global']['model_path'])
        self.global_emb = self.config['global']['model_path']
        return

    def fit_local(self, words_list, c):
        all_words = [word for keywords in words_list
                        for word in keywords
                            if len(word)>=self.config['local']['min_word_len']]
        ndoc = len(words_list)
        if ndoc < self.config['local']['min_num_doc_per_topic']:
            return False
        nt = ndoc // self.config['local']['min_num_doc_per_topic'] + 1
        print(ndoc, ' docs - ', nt, ' topics')
        os.makedirs('data_model/model/ftlocal', exist_ok=True)
        data_path ='data_model/model/ftlocal/tr_'+'_'.join(c.split())
        f=open(data_path,'w',encoding='utf-8')
        str_decode =' '+ ' '.join(all_words).encode('ascii', 'ignore').decode()
        str_decode = re.sub(' +', ' ', str_decode)
        print(str_decode[:10])
        f.write(str_decode)
        f.close()
        model = fasttext.train_unsupervised(
            data_path,
            dim= max(nt, 2),
            minCount= self.config['local']['min_word_count']
        )
        model_path = os.path.join( os.path.dirname(data_path),'_'.join(c.split())+'.bin' )
        model.save_model(model_path )
        self.local_emb[c] = model_path
        #model.save_model(self.config['local']['model_path'])
        return True

    def global_embedding_matrix(self):
        model = load(self.config['global']['model_path'])
        return np.stack([model.get_word_vector(w) for w in model.words])

    def global_vocab(self):
        model = load(self.config['global']['model_path'])
        return {w:i for i, w in enumerate(model.words)}

    def local_embedding_matrix(self,c):
        model= load(self.local_emb[c])
        return np.stack([model.get_word_vector(w) for w in model.words])

    def local_vocab(self,c):
        model = load(self.local_emb[c])
        return {w:i for i,w in enumerate(model.words)}

    def dist_func(self,x1,x2=None):
        #return cosine_similarity(x1,x2)
        if x2 is None:
            x2=x1
        return np.vstack([0.5-0.5*vec2mat_cos_sim(x,x2) for x in x1])


class topic_embedding_model(Embbedding_model):
    def __init__(self, config_file):
        super(topic_embedding_model, self).__init__(config_file)

    def fit_global(self, words_list):
        self.global_emb = VAETopic(
            words_list,
            n_topics=self.config['global']['n_topics'],
            min_cnt=self.config['global']['min_word_cnt'])
        self.global_emb.fit(
            num_epoch=self.config['global']['num_epoch'],
            disp=self.config['global']['display'])
        save(self.global_emb, self.config['global']['model_path'])
        return

    def fit_local(self, words_list, c):
        ndoc = len(words_list)
        nt = ndoc // self.config['local']['min_num_doc_per_topic'] + 1
        print(ndoc, ' docs - ', nt, ' topics')
        if ndoc < self.config['local']['min_num_doc_per_topic']:
            return False
        topic_model = VAETopic(
            words_list,
            n_topics=nt,
            min_cnt=self.config['local']['min_word_cnt'])
        if not topic_model.initialized:
            return False
        self.local_emb[c] = topic_model
        self.local_emb[c].fit(
            num_epoch=self.config['local']['num_epoch'],
            disp=self.config['local']['display'])
        #self.save(self.local_emb, self.config['local']['model_path'])
        return True

    def global_embedding_matrix(self):
        return self.global_emb.get_embeding_matrix().T

    def global_vocab(self):
        return self.global_emb.vocab.copy()

    def local_embedding_matrix(self,c):
        return self.local_emb[c].get_embeding_matrix().T

    def local_vocab(self,c):
        return self.local_emb[c].vocab.copy()

    def dist_func(self,x1,x2):
        if x2 is None:
            x2 = x1
        return np.vstack([0.5 - 0.5 * vec2mat_cos_sim(x, x2) for x in x1])
        #return euclidean_distances(x1,x2)
        #return pkl(x1,x2)