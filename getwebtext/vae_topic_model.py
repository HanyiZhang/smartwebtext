# Third-party libraries
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix

associations = {
    'jesus': ['prophet', 'jesus', 'matthew', 'christ', 'worship', 'church'],
    'Computer': ['floppy', 'windows', 'microsoft', 'monitor', 'workstation', 'macintosh',
              'printer', 'programmer', 'colormap', 'scsi', 'jpeg', 'compression'],
    'polit': ['amendment', 'libert', 'regulation', 'president'],
    'crime': ['violent', 'homicide', 'rape'],
    'midea': ['lebanese', 'israel', 'lebanon', 'palest'],
    'Account': ['account','access','login','sign'],
    'App': ['tool','msa','crm', 'azure','csm','ui','gmt','interface','application','version','app','bizapp','ux','eou','gant','cee','csu'],
    'Customer': ['product','customer','partner','use','learning','stakeholder'],
    'Function': ['engagement','enagement','oppty','opportun','email','ticket','export','insight','connect',\
                 'integration','opps','opp','data','excel','info','update','message','meeting','functio','exam'],
    'Time': ['time','timing','slow','clunky','minut','secon','year'],
    'Layout, Design':['link','design','column','tab','page','dashboard','session','tree','status','panel','menu','rule','chart'],
    'gears': ['helmet', 'bike'],
    'nasa ': ['orbit', 'spacecraft'],
    'Sales, Bussiness':['sale','market','sell','competitor','kpi','bussiness','price'],
    'Error': ['test','error'],
    'Action': ['accelerat','action','integrat','visit','training','team','follow'],
}

class config:
    # to be set with vocab size
    num_input=0

    en1_units=100
    en2_units=100
    num_topic=50
    batch_size=256
    optimizer='Adam'
    learning_rate=0.002
    momentum=0.99
    num_epoch=1000
    init_mult=1.0
    variance=0.995
    start=True
    nogpu=True


class ProdLDA(nn.Module):

    def __init__(self, net_arch):
        super(ProdLDA, self).__init__()
        ac = net_arch
        self.net_arch = net_arch
        self.h_dim=ac.num_topic
        # encoder
        self.en1_fc     = nn.Linear(ac.num_input, ac.en1_units)             # 1995 -> 100
        self.en2_fc     = nn.Linear(ac.en1_units, ac.en2_units)             # 100  -> 100
        self.en2_drop   = nn.Dropout(0.2)
        self.mean_fc    = nn.Linear(ac.en2_units, ac.num_topic)             # 100  -> 50
        self.mean_bn    = nn.BatchNorm1d(ac.num_topic)                      # bn for mean
        self.logvar_fc  = nn.Linear(ac.en2_units, ac.num_topic)             # 100  -> 50
        self.logvar_bn  = nn.BatchNorm1d(ac.num_topic)                      # bn for logvar
        # z
        self.p_drop     = nn.Dropout(0.2)
        # decoder
        self.decoder    = nn.Linear(ac.num_topic, ac.num_input)             # 50   -> 1995
        self.decoder_bn = nn.BatchNorm1d(ac.num_input)                      # bn for decoder

        self.a = 1 * np.ones((1, self.h_dim)).astype(np.float32)
        self.prior_mean = torch.from_numpy((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)  # prior_mean  = 0
        self.prior_var = torch.from_numpy(
            (((1.0 / self.a) * (1 - (2.0 / self.h_dim))).T +  # prior_var = 0.99 + 0.005 = 0.995
             (1.0 / (self.h_dim * self.h_dim)) * np.sum(1.0 / self.a, 1)).T)
        self.prior_logvar = self.prior_var.log()

        # initialize decoder weight
        if ac.init_mult != 0:
            self.decoder.weight.data.uniform_(0, ac.init_mult)


    def forward(self, input, compute_loss=False, avg_loss=True):
        # compute posterior
        en1 = F.softplus(self.en1_fc(input))                            # en1_fc   output
        en2 = F.softplus(self.en2_fc(en1))                              # encoder2 output
        en2 = self.en2_drop(en2)
        posterior_mean   = self.mean_bn(self.mean_fc  (en2))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))          # posterior log variance
        posterior_var    = posterior_logvar.exp()
        # take sample
        eps = Variable(input.data.new().resize_as_(posterior_mean.data).normal_()) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
        p0 = F.softmax(z,dim=1)                                                # mixture probability
        p = self.p_drop(p0)
        # do reconstruction
        recon = F.softmax(self.decoder_bn(self.decoder(p)),dim=1)             # reconstructed distribution over vocabulary

        if compute_loss:
            return recon, self.loss(input, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss)
        else:
            return recon,p0

    def loss(self, input, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL  = -(input * (recon+1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017,
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.net_arch.num_topic )
        # loss
        loss = (NL + KLD)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean()
        else:
            return loss


class VAETopic():

    def __init__(self,keywords_list,n_topics=20,min_size=5,topic_def={},min_cnt=10):
        args=config
        args.num_topic=n_topics
        cnt=Counter([word for keywords in keywords_list for word in keywords])
        values = list(cnt.values())
        self.vocab_count_percentiles = {
            i: np.percentile(values,i) for i in range(10,100,10)
        }
        print('word count percentiles: ',self.vocab_count_percentiles)
        del values
        self.vocab={w:i for i,w in enumerate(
            sorted(
                set([word for keywords in keywords_list
                     for word in keywords if cnt[word]>min_cnt ])
            ))}
        self.vocab_size = len(self.vocab)
        self.associations = associations
        self.associations.update(topic_def)

        # --------------convert to one-hot representation------------------
        print('Converting data to one-hot representation')
        irow= 0
        row_ind = []
        col_ind = []
        values =[]
        for keywords in keywords_list:
            doc=[self.vocab[word] for word in keywords if cnt[word]>min_cnt]
            if sum(doc) > 0:
                vec=self.to_onehot(doc, self.vocab_size)
                ids =np.where(vec > 0)
                row_ids, col_ids =list(zip(*[(irow, c) for c in ids[0]]))
                row_ind+= list(row_ids)
                col_ind+= list(col_ids)
                values+=vec[ids].tolist()
                irow+=1
        self.tensor_tr = csr_matrix((values, (row_ind, col_ind)), shape=(irow, len(self.vocab)))
        # torch.sparse_csr_tensor(row_ind, col_ind, values, dtype=torch.int32)
        del row_ind
        del col_ind
        del values
        print('Data Loaded')
        print('Data Dim ', self.tensor_tr.shape)
        if self.tensor_tr.shape[0]<min_size:
            print('Data size too small !')
            self.initialized=False
            return

        # --------------make tensor datasets-------------------------------
        # create model
        args.num_input = self.tensor_tr.shape[1]
        self.model = ProdLDA(args)
        self.initialized=True
        return

    def to_onehot(self,data, min_length):
        return np.bincount(data, minlength=min_length)

    def fit(self,num_epoch=100, disp=False):
        if not self.initialized:
            return False
        if config.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), config.learning_rate, betas=(config.momentum, 0.999))
        elif config.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), config.learning_rate, momentum=config.momentum)
        else:
            print('Unknown optimizer {}'.format(config.optimizer))
            return False
        #nrows=self.tensor_tr.size(0)
        self.model.train()     # switch to training mode
        for epoch in range(num_epoch):
            all_indices = torch.randperm(self.tensor_tr.shape[0]).split(config.batch_size)
            loss_epoch = 0.0
            for i, batch_indices in enumerate(all_indices):
                batch = self.tensor_tr[batch_indices].toarray()
                input = torch.from_numpy(batch).float()
                if input.size(0)==1:
                    input= input.repeat(2,1)
                # optimize
                recon, loss = self.model(input, compute_loss=True)
                optimizer.zero_grad()
                loss.backward()             # backprop
                optimizer.step()            # update parameters
                loss_epoch += loss.data.item()    # add loss to loss_epoch
                #print('batch ', i, '/', len(all_indices))
            if disp:
                print('Epoch {}, loss={}'.format(epoch, loss_epoch / len(all_indices)))
        if disp:
            print('Epoch {}, loss={}'.format(epoch, loss_epoch / len(all_indices)))
        return True

    def get_embeding_matrix(self):
        return self.model.decoder.weight.data.cpu().numpy().T

    def get_doc_topic_prob(self):
        self.model.eval()
        p_all = []
        for ib in range(0,self.tensor_tr.shape[1],config.batch_size):
            batch = self.tensor_tr[ib:ib+config.batch_size].toarray()
            input = torch.from_numpy(batch).float()
            _,p=self.model(input)
            p_all += p.cpu().detach().numpy()
        return np.vstack(p_all)

    def identify_topic_in_line(self,line):
        topics = []
        for topic, keywords in self.associations.items():
            for word in keywords:
                if word.lower() in line.lower():
                    topics.append(topic)
                    break
        return ' '.join(topics)

    def get_top_words_and_topics(self, n_top_words=3,disp=False):
        feature_names=list(self.vocab.keys())
        beta=self.get_embeding_matrix()
        top_words_all=[]
        topic_ids_all=[]
        topics=[]
        if disp: print('---------------Printing the Topics------------------')
        for i in range(len(beta)):
            top_words,topic_ids= zip(*[(feature_names[j],j) for j in beta[i].argsort()[:-n_top_words - 1:-1]])
            top_words_all.append(list(top_words))
            topic_ids_all.append(list( topic_ids))
            line=" ".join(top_words)
            topic = self.identify_topic_in_line(line)
            topics.append(topic+' - '+' '.join(top_words))
            if disp:
                print(topics[-1])
                print('     {}'.format(line))
        if disp: print('---------------End of Topics------------------')
        return top_words_all,topics,topic_ids_all

    def calc_perp(self,disp=False,tensor_te=None):
        if tensor_te is None:
            tensor_te=self.tensor_tr
        self.model.eval()    # switch to testing mode
        loss= np.zeros( len(self.vocab))
        counts = np.zeros( len(self.vocab))
        #input = Variable(tensor_te)
        for ib in range(0,tensor_te.shape[1],config.batch_size):
            tensor_te_batch = tensor_te[ib:ib + config.batch_size].toarray()
            input = torch.from_numpy(tensor_te_batch).float()
            recon, loss_ = self.model(input, compute_loss=True, avg_loss=False)
            loss += loss_.data.cpu().detach().numpy()
            counts += tensor_te_batch.sum(1).detach().numpy()

        avg = (loss / counts).mean()
        perp=np.exp(avg)
        if disp:
            print('The approximated perplexity is: ', perp)
        return perp
