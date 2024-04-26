import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import SequentialSampler
from torch import nn 
import numpy as np
import random
import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, precision_score, recall_score
import pickle 
import copy
from prettytable import PrettyTable
from utils.data_process import data_process_loader, data_predict_process_loader
from model.model import MMTF_CPI
from torch.utils.tensorboard import SummaryWriter


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # dgl.seed(seed)


def model_initialize(cuda_name, result_folder, LR, Decay, Batch_size, Epoch, SEED):
    model = Train_MMTF(cuda_name, result_folder, LR, Decay, Batch_size, Epoch)
    setup_seed(SEED)
    return model


class Train_MMTF:
    def __init__(self, cuda_name, result_folder, LR, Decay, Batch_size, Epoch):
        super(Train_MMTF, self).__init__()
        
        self.device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')        
        
        self.result_folder = result_folder
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)

      
        self.model = MMTF_CPI(self.device)
        
        self.lr = LR
        self.decay = Decay
        self.batch_size = Batch_size
        self.epoch = Epoch

    def dgl_collate_func(self, x):
        import dgl
        MAX_PROTEIN_LEN = 1500
        MAX_DRUG_LEN = 200
        
        d_s,p_s, d_h, p_h, d_t,p_t,y = zip(*x)
        d_s = dgl.batch(d_s)
        
        N = len(x)
        
        proteins_len = 0

        for protein in p_s:
            if protein.shape[0] >= proteins_len:
                proteins_len = protein.shape[0]
        
        if proteins_len>MAX_PROTEIN_LEN: proteins_len = MAX_PROTEIN_LEN
        proteins_new = torch.zeros((N, proteins_len, 100))
        
        i = 0
        for protein in p_s:
            a_len = protein.shape[0]
            if a_len>proteins_len: a_len = proteins_len
            proteins_new[i, :a_len, :] = torch.tensor(protein[:a_len, :])
            i += 1
        
        return d_s,torch.as_tensor(proteins_new, dtype=torch.float32),\
            torch.as_tensor(d_h, dtype=torch.float32),torch.as_tensor(p_h, dtype=torch.float32),\
            torch.as_tensor(d_t, dtype=torch.float32),torch.as_tensor(p_t, dtype=torch.float32),\
            torch.as_tensor(y, dtype=torch.float32)
    

    def predict_dgl_collate_func(self, x):
        import dgl
        MAX_PROTEIN_LEN = 1500
        MAX_DRUG_LEN = 200
        
        d_s,p_s, d_h, p_h, d_t,p_t, y, info = zip(*x)
        
        d_s = dgl.batch(d_s) 
        
        N = len(x)
        
        proteins_len = 0

        for protein in p_s:
            if protein.shape[0] >= proteins_len:
                proteins_len = protein.shape[0]
        
        if proteins_len>MAX_PROTEIN_LEN: proteins_len = MAX_PROTEIN_LEN
        proteins_new = torch.zeros((N, proteins_len, 100))
        
        i = 0
        for protein in p_s:
            a_len = protein.shape[0]
            if a_len>proteins_len: a_len = proteins_len
            proteins_new[i, :a_len, :] = torch.tensor(protein[:a_len, :])
            i += 1
        
        return d_s,torch.as_tensor(proteins_new, dtype=torch.float32),\
            torch.as_tensor(d_h, dtype=torch.float32),torch.as_tensor(p_h, dtype=torch.float32),\
            torch.as_tensor(d_t, dtype=torch.float32),torch.as_tensor(p_t, dtype=torch.float32),\
            torch.as_tensor(y, dtype=torch.float32), pd.DataFrame(list(info))
        

    def cal_score(self,feat, predict, model=None):
        
        d_s = feat[0].to(self.device)
        p_s = feat[1].to(self.device)

        d_h = feat[2].to(torch.float32).to(self.device)
        p_h = feat[3].to(torch.float32).to(self.device)

        d_t = feat[4].to(torch.float32).to(self.device)
        p_t = feat[5].to(torch.float32).to(self.device)
        label = feat[6].to(self.device)

        if predict:
            score = model((d_s,p_s,d_h, p_h, d_t,p_t))
            
        else:
            score = self.model((d_s,p_s,d_h, p_h, d_t,p_t))
            
        return score, label
        

    def test_(self, data_generator, model):
        y_pred = []
        y_label = []
        model.eval()
        for i, feat in enumerate(data_generator):
            score,label = self.cal_score(feat, predict=False)
            loss_fct = torch.nn.BCELoss()

            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score), 1)
            loss = loss_fct(n, label)
            
            logits = torch.squeeze(m(score)).detach().cpu().numpy()
            
            label_ids = Variable(label)
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

            confusion = confusion_matrix(y_label, outputs)
            TP = confusion[1, 1]
            TN = confusion[0, 0]
            FP = confusion[0, 1]
            FN = confusion[1, 0]
        
        return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), y_pred, y_label, loss
    

    def predict(self, data_generator, model):
        y_pred = []
        test_info = []
        model.eval()
        for i, feat in enumerate(data_generator):
            score,label = self.cal_score(feat[:-1], predict=True, model=model)

            m = torch.nn.Sigmoid()
            logits = torch.squeeze(m(score)).detach().cpu().numpy()
            
            label_ids = Variable(label)
            y_pred = y_pred + logits.flatten().tolist()
            test_info.append(feat[-1])
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

        return y_pred, test_info
    
    

    def train(self, fold, train, val = None, test = None, verbose = True):
        lr = self.lr
        decay = self.decay
        BATCH_SIZE = self.batch_size
        train_epoch = self.epoch
        
        loss_history = []
        
        opt = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = decay)

        params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'drop_last': True}
        params['collate_fn'] = self.dgl_collate_func
        
        training_generator = data.DataLoader(data_process_loader(train), **params)
        if val is not None:
            validation_generator = data.DataLoader(data_process_loader(val), **params)
        if test is not None:
            testing_generator = data.DataLoader(data_process_loader(test), **params)
        
        # early stopping
        
        min_loss = 20
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []
        valid_metric_header = ["# epoch"] 

        valid_metric_header.extend(["AUROC", "AUPRC"])
        table = PrettyTable(valid_metric_header)
        
        float2str = lambda x:'%0.4f'%x

        if verbose:
            print('Go for Training ...')
        
        self.model = self.model.to(self.device)
        self.model.train()
        writer = SummaryWriter()
        t_start = time() 
        iteration_loss = 0
        
        for epo in range(train_epoch):
            for i, feat in enumerate(training_generator):
                score,label = self.cal_score(feat, predict=False)

                loss_fct = torch.nn.BCELoss()
                m = torch.nn.Sigmoid()
                n = torch.squeeze(m(score), 1)
                loss = loss_fct(n, label)
                
                loss_history.append(loss.item())
                writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()

                if verbose:
                    if (i % 500 == 0):
                        t_now = time()
                        print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + \
                            ' with loss ' + str(loss.cpu().detach().numpy())[:7] +\
                            ". Total time " + str(int(t_now - t_start)/3600)[:7] + " hours") 

            if val is not None:
                with torch.set_grad_enabled(False):
                    auc, auprc, _, _,loss = self.test_(validation_generator, self.model)
                    lst = ["epoch " + str(epo)] + list(map(float2str,[auc, auprc]))
                    valid_metric_record.append(lst)
                    
                    if verbose:
                        print('Validation at Epoch '+ str(epo + 1) + ', AUROC: ' + str(auc)[:7] + \
                        ' , AUPRC: ' + str(auprc)[:7] + ' , Cross-entropy Loss: ' + \
                        str(loss)[:7])

                    if loss < min_loss:
                        model_max = copy.deepcopy(self.model)
                        min_loss = loss
                        es = 0
                    else:
                        es += 1
                        print("Counter {} of 5".format(es))
                        if es > 4:
                            print("Early stopping with best_auc: ", "auc for this epoch: ", str(auc)[:7], "...")
                            break

                table.add_row(lst)
            else:
                model_max = copy.deepcopy(self.model)

        self.model = model_max
        
        if val is not None:
            prettytable_file = os.path.join(self.result_folder, "valid_markdowntable.txt")
            with open(prettytable_file, 'w') as fp:
                fp.write(table.get_string())

        if test is not None:
            if verbose:
                print('Go for Testing ...')
                
            auc, auprc,logits, test_label, _ = self.test_(testing_generator, self.model)
            test_table = PrettyTable(["AUROC", "AUPRC"])
            test_table.add_row(list(map(float2str, [auc, auprc])))
            
            if verbose:
                print('Test at Epoch '+ str(epo + 1) + ' , AUROC: ' + str(auc)[:7] + \
                    ' , AUPRC: ' + str(auprc)[:7])    
            
            preds = np.concatenate((np.array(logits).reshape(1,-1),np.array(test_label).reshape(1,-1)),axis=0)
            np.save(os.path.join(self.result_folder, str(fold)  + 'fold.npy'), preds)                
   
            prettytable_file = os.path.join(self.result_folder, "test_markdowntable.txt")
            with open(prettytable_file, 'w') as fp:
                fp.write(test_table.get_string())

        if verbose:
            print('Training Finished!')
            writer.flush()
            writer.close()

    def Predicting(self, test, model, predict_result_path):
        lr = self.lr
        decay = self.decay
        BATCH_SIZE = self.batch_size
    
        opt = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = decay)

        params = {'batch_size': BATCH_SIZE,
                'shuffle': False,
                'drop_last': False}
        params['collate_fn'] = self.predict_dgl_collate_func
        
        testing_generator = data.DataLoader(data_predict_process_loader(test), **params)
        
        logits, test_info = self.predict(testing_generator, model)

        predict_result = pd.concat(test_info, axis=0)
        predict_result['predict_score'] = logits

        return predict_result


    

    def save_model(self, result_folder):
        torch.save(self.model.state_dict(), result_folder + '/model.pt')

    





