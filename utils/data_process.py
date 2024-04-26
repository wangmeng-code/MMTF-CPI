import pandas as pd
import numpy as np
import pickle
from torch.utils import data
from time import time
import torch
import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer,CanonicalBondFeaturizer
from utils.word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec
from dgllife.utils import smiles_to_bigraph, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from functools import partial

def get_d_structure(drug,drug_info):
    drug_node_featurizer = AttentiveFPAtomFeaturizer()
    drug_bond_featurizer = AttentiveFPBondFeaturizer(self_loop=True)
    
    fc = partial(smiles_to_bigraph, add_self_loop=True)
    drug = drug.split('::')[1]
    if drug in list(drug_info.keys()):
        D_data = drug_info[drug]

        try:
            D_data['structure'] = fc(smiles = D_data['SMILES'], node_featurizer = drug_node_featurizer, edge_featurizer = drug_bond_featurizer)
        except:
            D_data['structure'] = ''
    else:
        D_data = {}
        D_data['structure'] = ''
    return D_data['structure']


def get_p_structure(protein,prot_info, prot_model):
    protein = protein.split('::')[1]
    if protein in list(prot_info.keys()):
        P_data = prot_info[protein]
        P_data['Structure'] = get_protein_embedding(prot_model, seq_to_kmers(P_data['Sequence']))
        
    else:
        P_data = {}
        P_data['Structure'] = ''
    return P_data['Structure']


def get_heterogeneous_network_embs(embedding_path):
    with open(embedding_path,'rb') as f:
        entity_representation = pickle.load(f)
    return entity_representation


def get_d_h(entity_representation, drug):
    if drug in list(entity_representation.keys()):
        d_h = np.array(entity_representation[drug])
    else:
        d_h = np.zeros((64,))
    return d_h


def get_p_h(entity_representation, protein): 
    if protein in list(entity_representation.keys()):
        p_h_embs = np.array(entity_representation[protein])
    else:
        p_h_embs = np.zeros((64,))
    return p_h_embs

def get_d_profiles(cell_id):
    expr = pd.read_csv('./data/Cell_line_data/' + cell_id + '/drugbank.csv',index_col=0)
    drug_expr_info = expr.index.values
    return expr,drug_expr_info

def get_d_t(drug,expr,drug_expr_info):
    drug = drug.split('::')[1]
    n_genes = expr.shape[1]
    if drug in drug_expr_info:
        d_t = np.array(expr.loc[drug])
    else:
        d_t =  np.zeros((n_genes, ))
    return d_t


def get_p_profiles(cell_id):
    xpr_sig = pd.read_csv('./data/Cell_line_data/' + cell_id + '/knockdown.csv',index_col=0)
    oe_sig = pd.read_csv('./data/Cell_line_data/' + cell_id + '/overexpression.csv',index_col=0)
    
    xpr_sig.index = [str(id) for id in xpr_sig.index.values]
    xpr_info = xpr_sig.index.values
    
    oe_sig.index = [str(id) for id in oe_sig.index.values]
    oe_info = oe_sig.index.values
    return xpr_sig,oe_sig, xpr_info,oe_info


def get_p_t(protein,xpr_sig,oe_sig, xpr_info,oe_info,UP):
    protein = protein.split('::')[1]
    n_genes = xpr_sig.shape[1]
    if UP == 1:
        if protein in oe_info:
            p_t = np.array(oe_sig.loc[protein])
        else:
            p_t =  np.zeros((n_genes, ))
    
    if UP == 0:
        if protein in xpr_info:
            p_t = np.array(xpr_sig.loc[protein])
        else:
            p_t =  np.zeros((n_genes, ))

    return p_t


def process(train, val, test, info_path, cell_id, embedding_path):
    train_pro = {}
    val_pro = {}
    test_pro = {}
    t_start = time() 
    df = pd.concat([train,val,test])
    
    with open(info_path + 'protein_info.pkl', 'rb') as fp:
        prot_info = pickle.load(fp)

    with open(info_path + 'drug_info.pkl', 'rb') as fp:
        drug_info = pickle.load(fp)
    
    prot_model = Word2Vec.load(info_path + "word2vec_30.model")
    
    ## Structure modality
    # drug structures
    print('Prepare drug structures ...')
    unique = []
    D_str_item = []
    
    for item in df['head'].unique():
        d_graph = get_d_structure(item,drug_info)
        unique.append(d_graph)

        if d_graph:
            if d_graph.edata:
                D_str_item.append(item)
    
    unique_dict_d = dict(zip(df['head'].unique(), unique))

    # protein structures
    print('Prepare protein structures ...')
    unique = []
    P_str_item = []

    for item in df['tail'].unique():
        p_graph = get_p_structure(item,prot_info, prot_model)
        unique.append(p_graph)

        if p_graph.any():
            P_str_item.append(item)

    unique_dict_p = dict(zip(df['tail'].unique(), unique))
    
    ## Transcriptional profiling modality
    # drug profilings
    print('Prepare drug profilings ...')
    expr,drug_expr_info = get_d_profiles(cell_id)
    D_Expr = []
    D_expr_item = []
    for item in df['head'].unique():
        d_t = get_d_t(item,expr,drug_expr_info)
        D_Expr.append(d_t)
        if d_t.any():
            D_expr_item.append(item)
    D_Expr_dic = dict(zip(df['head'].unique(), D_Expr))
    
    # protein profilings
    print('Prepare protein profilings ...')
    xpr_sig,oe_sig, xpr_info,oe_info = get_p_profiles(cell_id)

    P_expr_item = []
    for i in range(0,len(df)):
        tail = df['tail'].iloc[i]
        UP = df['UP'].iloc[i]

        p_t = get_p_t(tail,xpr_sig,oe_sig, xpr_info,oe_info,UP)
        
        if p_t.any():
            P_expr_item.append(tail)
    
    keep_D_item = D_expr_item
    keep_P_item = list(set(P_expr_item))
    
    keep_D_item = pd.DataFrame(keep_D_item,columns=['head'])
    keep_P_item = pd.DataFrame(keep_P_item,columns=['tail'])
    
    df = pd.merge(df,keep_D_item,on='head')
    df = pd.merge(df,keep_P_item,on='tail')
    
    train = pd.merge(train,keep_D_item,on='head')
    train_pro['df'] = pd.merge(train,keep_P_item,on='tail')
    val = pd.merge(val,keep_D_item,on='head')
    val_pro['df'] = pd.merge(val,keep_P_item,on='tail')
    test = pd.merge(test,keep_D_item,on='head')
    test_pro['df'] = pd.merge(test,keep_P_item,on='tail')

    train_pro['D_Structure'] = [unique_dict_d[i] for i in train_pro['df']['head']]
    val_pro['D_Structure'] = [unique_dict_d[i] for i in val_pro['df']['head']]
    test_pro['D_Structure'] = [unique_dict_d[i] for i in test_pro['df']['head']]

    train_pro['P_Structure'] = [unique_dict_p[i] for i in train_pro['df']['tail']]
    val_pro['P_Structure'] = [unique_dict_p[i] for i in val_pro['df']['tail']]
    test_pro['P_Structure'] = [unique_dict_p[i] for i in test_pro['df']['tail']]

    train_pro['D_t'] = [D_Expr_dic[i] for i in train_pro['df']['head']]
    val_pro['D_t'] = [D_Expr_dic[i] for i in val_pro['df']['head']]
    test_pro['D_t'] = [D_Expr_dic[i] for i in test_pro['df']['head']]

    # get protein expr
    xpr_sig,oe_sig, xpr_info,oe_info = get_p_profiles(cell_id)
    
    def get_p_exprs(df,xpr_sig,oe_sig, xpr_info,oe_info):
        P_expr = []
        for i in range(0,len(df)):
            tail = df['tail'].iloc[i]
            UP = df['UP'].iloc[i]
            P_expr.append(get_p_t(tail,xpr_sig,oe_sig, xpr_info,oe_info,UP))
        return P_expr
    
    train_pro['P_t'] = get_p_exprs(train_pro['df'],xpr_sig,oe_sig, xpr_info,oe_info)
    val_pro['P_t'] = get_p_exprs(val_pro['df'],xpr_sig,oe_sig, xpr_info,oe_info)
    test_pro['P_t'] = get_p_exprs(test_pro['df'],xpr_sig,oe_sig, xpr_info,oe_info)
    
    ## Heterogeneous network modality
    # Drug embeddings
    print('Prepare drug embeddings from heterogeneous network modality ...')
    entity_representation = get_heterogeneous_network_embs(embedding_path)
    unique = []
    
    for item in df['head'].unique():
        unique.append(get_d_h(entity_representation,item))
    unique_dict = dict(zip(df['head'].unique(), unique))
    train_pro['D_h'] = [unique_dict[i] for i in train_pro['df']['head']]
    val_pro['D_h'] = [unique_dict[i] for i in val_pro['df']['head']]
    test_pro['D_h'] = [unique_dict[i] for i in test_pro['df']['head']]

    # protein embeddings
    print('Prepare protein embeddings from heterogeneous network modality ...')
    unique = []
    for item in df['tail'].unique():
        unique.append(get_p_h(entity_representation,item))

    unique_dict = dict(zip(df['tail'].unique(), unique))
    train_pro['P_h'] = [unique_dict[i] for i in train_pro['df']['tail']]
    val_pro['P_h'] = [unique_dict[i] for i in val_pro['df']['tail']]
    test_pro['P_h'] = [unique_dict[i] for i in test_pro['df']['tail']]   

    train_pro['Label'] = torch.tensor(train_pro['df']['Label'].values).to(torch.float32)
    val_pro['Label'] = torch.tensor(val_pro['df']['Label'].values).to(torch.float32)
    test_pro['Label'] = torch.tensor(test_pro['df']['Label'].values).to(torch.float32)
    
    return train_pro, val_pro, test_pro


def predict_process(test, info_path, cell_id, embedding_path):
    test_pro = {}
    t_start = time() 
    df = test
    
    with open(info_path + 'protein_info.pkl', 'rb') as fp:
        prot_info = pickle.load(fp)

    with open(info_path + 'drug_info.pkl', 'rb') as fp:
        drug_info = pickle.load(fp)
    
    prot_model = Word2Vec.load(info_path + "word2vec_30.model")
    
    ## Structures
    # drug structures
    print('prepare drug structures ...')
    unique = []
    D_str_item = []
    
    for item in df['head'].unique():
        d_graph = get_d_structure(item,drug_info)
        unique.append(d_graph)

        if d_graph:
            if d_graph.edata:
                D_str_item.append(item)
    
    unique_dict_d = dict(zip(df['head'].unique(), unique))

    # protein structures
    print('prepare protein structures ...')
    unique = []
    P_str_item = []

    for item in df['tail'].unique():
        p_graph = get_p_structure(item,prot_info, prot_model)
        unique.append(p_graph)

        if p_graph.any():
            P_str_item.append(item)

    unique_dict_p = dict(zip(df['tail'].unique(), unique))
    
    ## Transcriptional profilings
    # drug transcriptional profilingns
    print('prepare drug transcriptional profilings ...')
    expr,drug_expr_info = get_d_profiles(cell_id)
    D_Expr = []
    D_expr_item = []
    for item in df['head'].unique():
        d_t = get_d_t(item,expr,drug_expr_info)
        D_Expr.append(d_t)
        if d_t.any():
            D_expr_item.append(item)
    D_Expr_dic = dict(zip(df['head'].unique(), D_Expr))
    
    # protein transcriptional profilings
    print('prepare protein transcriptional profilings ...')
    xpr_sig,oe_sig, xpr_info,oe_info = get_p_profiles(cell_id)

    P_expr_item = []
    for i in range(0,len(df)):
        tail = df['tail'].iloc[i]
        UP = df['UP'].iloc[i]

        p_t = get_p_t(tail,xpr_sig,oe_sig, xpr_info,oe_info,UP)
        
        if p_t.any():
            P_expr_item.append(tail)
    
    keep_D_item = D_expr_item
    keep_P_item = list(set(P_expr_item))
    
    keep_D_item = pd.DataFrame(keep_D_item,columns=['head'])
    keep_P_item = pd.DataFrame(keep_P_item,columns=['tail'])
    
    df = pd.merge(df,keep_D_item,on='head')
    df = pd.merge(df,keep_P_item,on='tail')

    test = pd.merge(test,keep_D_item,on='head')
    test_pro['df'] = pd.merge(test,keep_P_item,on='tail')
    
    test_pro['D_Structure'] = [unique_dict_d[i] for i in test_pro['df']['head']]

    test_pro['P_Structure'] = [unique_dict_p[i] for i in test_pro['df']['tail']]

    test_pro['D_t'] = [D_Expr_dic[i] for i in test_pro['df']['head']]

    # get protein expr
    xpr_sig,oe_sig, xpr_info,oe_info = get_p_profiles(cell_id)
    
    def get_p_exprs(df,xpr_sig,oe_sig, xpr_info,oe_info):
        P_expr = []
        for i in range(0,len(df)):
            tail = df['tail'].iloc[i]
            UP = df['UP'].iloc[i]
            P_expr.append(get_p_t(tail,xpr_sig,oe_sig, xpr_info,oe_info,UP))
        return P_expr
    
    test_pro['P_t'] = get_p_exprs(test_pro['df'],xpr_sig,oe_sig, xpr_info,oe_info)
    
    ## Heterogeneous network modality
    # Drug embeddings
    print('Prepare drug embeddings from heterogeneous network modality ...')
    entity_representation = get_heterogeneous_network_embs(embedding_path)
    unique = []
    
    for item in df['head'].unique():
        unique.append(get_d_h(entity_representation,item))
    unique_dict = dict(zip(df['head'].unique(), unique))

    test_pro['D_h'] = [unique_dict[i] for i in test_pro['df']['head']]

    # protein embeddings
    print('Prepare protein embeddings from heterogeneous network modality ...')
    unique = []
    for item in df['tail'].unique():
        unique.append(get_p_h(entity_representation,item))

    unique_dict = dict(zip(df['tail'].unique(), unique))

    test_pro['P_h'] = [unique_dict[i] for i in test_pro['df']['tail']]   

    test_pro['Label'] = torch.tensor(test_pro['df']['Label'].values).to(torch.float32)
    
    return test_pro


class data_process_loader(data.Dataset):
    
    def __init__(self, df):
        self.df = df
        self.list_IDs = np.array(range(0,len(self.df['df'])))

    def __len__(self):
        return len(self.df['df'])

    def __getitem__(self, index):
        index = self.list_IDs[index]
            
        d_s = self.df['D_Structure'][index]
        p_s = self.df['P_Structure'][index]
        
        d_h = self.df['D_h'][index]
        p_h = self.df['P_h'][index]

        d_t = self.df['D_t'][index]
        p_t = self.df['P_t'][index]
        y = self.df['Label'][index]

        return d_s, p_s, d_h, p_h, d_t, p_t, y


class data_predict_process_loader(data.Dataset):
    
    def __init__(self, df):
        self.df = df
        self.list_IDs = np.array(range(0,len(self.df['df'])))

    def __len__(self):
        return len(self.df['df'])

    def __getitem__(self, index):
        index = self.list_IDs[index]
        

        d_s = self.df['D_Structure'][index]
        p_s = self.df['P_Structure'][index]
        
        d_h = self.df['D_h'][index]
        p_h = self.df['P_h'][index]

        d_t = self.df['D_t'][index]
        p_t = self.df['P_t'][index]
        y = self.df['Label'][index]

        info = self.df['df'].iloc[index,:3]

        return d_s, p_s, d_h, p_h, d_t, p_t, y, info