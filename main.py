import os
import pandas as pd
import pickle
from utils.Utils import *
from utils.data_process import *
from utils import word2vec
from sklearn.utils import shuffle
import train.train_model as models
from sklearn.model_selection import KFold
import numpy as np

KFolds = 10
cuda_name = 'cuda:1'

cell_id = 'A375'
model_type ='fusion'
lr = 1e-3
decay = 0
batch_size = 128
Kfolds=10
epoch = 50
seed = 1234
rpath = './data/'
embedding_path = './data/KG_embedding/embeddings/metapath_embeddings.pkl'


for i in range(Kfolds):
    print('==================== fold ' + str(i) + ' ======================')
    print('data preparing ...')
    
    result_folder = './result/'+model_type+'/' + cell_id + '/fold'+str(i)+'/'   
    train = pd.read_csv(rpath + 'Cross_validation_10fold/' + str(i) + 'fold/train.csv')
    val = pd.read_csv(rpath + 'Cross_validation_10fold/' +str(i) + 'fold/val.csv')
    test = pd.read_csv(rpath + 'Cross_validation_10fold/' +str(i) + 'fold/test.csv')
    
    train, val, test = process(train, val, test, rpath, cell_id, embedding_path)
    
    print('model training ...')
    train_MMTF = models.model_initialize(cuda_name, result_folder, lr, decay, batch_size, epoch, seed)
    train_MMTF.train(i,train,val,test)
    train_MMTF.save_model(result_folder)
    
    print('==================== fold' + str(i) + 'finished ======================')
    print('     ')
