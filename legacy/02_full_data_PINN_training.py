# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:05:14 2022

@author: LAI Chenyang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import scipy.io as scio
from torch import optim
from sklearn.model_selection import KFold # import KFold
from pinn_resp import PhysicsInformedNN, EarlyStopping, rul_metrics
import os
import itertools
import random #, optuna

#%% define random
# set random seeds
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     # random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.enabled = False
setup_seed(22)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% load data
##### load simulation data 
cell_B1 = './trajectory_cell_B1.mat'
cell_B1 = scipy.io.loadmat(cell_B1)['trajectory_cell_B1'].tolist()

cell_B2 = './trajectory_cell_B2.mat'
cell_B2 = scipy.io.loadmat(cell_B2)['trajectory_cell_B2'].tolist()

cell_B3 = './trajectory_cell_B3.mat'
cell_B3 = scipy.io.loadmat(cell_B3)['trajectory_cell_B3'].tolist()

cell = cell_B1 + cell_B2 + cell_B3
# cell = cell_B2

# feature index: 0-13; 14-24
features = [i[0][:,0:24] for i in cell]
features = [np.delete(i, 13, axis=1) for i in features]
rul = [i[0][:,-1][:,None] for i in cell]
capacity = [i[0][:,-2][:,None] for i in cell]
t = [np.arange(len(i[0]))[:,None] for i in cell]

data_all = [features, t, capacity, rul]

#%% cross-validation: prepare fold-based data
shuffle_ix = np.random.permutation(np.arange(len(features))) # random index

Kfold = 5
fold_ind = np.around(np.linspace(0,len(data_all[0]),Kfold+1)) # number of each fold

data_folds = [] # fold_num - (feats, t, cap, rul)
for i in range(Kfold):
    start_ind = fold_ind[i]
    end_ind = fold_ind[i+1]
    ind_set = shuffle_ix[int(start_ind):int(end_ind)]
    data_fold = []
    for data in data_all:
        fold = [j for ind,j in enumerate(data) if ind in ind_set]
        data_fold.append(fold)
    data_folds.append(data_fold)

#%% normalization
fold_test_ind = 0
# fold_val_ind = 1
# fold_train_ind = [2,3,4]
best_w_w2 = np.load("optimal_models/best_w_w2.npy")

for num_, fold_val_ind in enumerate([1,2,3,4]):
    
    fold_train_ind = [1,2,3,4]
    # fold_train_ind = [i for i in [1,2,3,4] if i!=fold_val_ind]

    #### generate data scaler
    fold_train_data = [i for ind, i in enumerate(data_folds) if ind in fold_train_ind]
    # connect training data
    train_data_x = []
    train_data_t = []
    train_data_cap = []
    train_data_rul = []
    for i in fold_train_data:
       train_data_x = train_data_x + i[0]
       train_data_t = train_data_t + i[1]
       train_data_cap = train_data_cap + i[2]
       train_data_rul = train_data_rul + i[3]
    train_data = [train_data_x, \
                          train_data_t, \
                          train_data_cap, \
                          train_data_rul]
    
    train_data_size_seq = [len(i) for i in train_data[0]]
        
    train_data_feats_con = np.concatenate(train_data[0])
    train_data_t_con = np.concatenate(train_data[1])
    train_data_cap_con = np.concatenate(train_data[2])
    
    # normalization scaler
    x_scaler = MinMaxScaler()
    x_scaler.fit(train_data_feats_con)
    t_scaler = MinMaxScaler()
    t_scaler.fit(train_data_t_con)
    cap_scaler = MinMaxScaler()
    cap_scaler.fit(train_data_cap_con)
    
    #### training input data normalization
    train_x_con_norm = x_scaler.transform(train_data_feats_con)
    train_t_con_norm = t_scaler.transform(train_data_t_con)
    train_cap_con_norm = cap_scaler.transform(train_data_cap_con)
    
    size_ind_train = np.cumsum(train_data_size_seq)
    size_ind_train = np.insert(size_ind_train, 0, 0)
    train_x_norm = [train_x_con_norm[i:j,:] for i,j in zip(size_ind_train[:-1], size_ind_train[1:])]
    train_t_norm = [train_t_con_norm[i:j,:] for i,j in zip(size_ind_train[:-1], size_ind_train[1:])]
    train_cap_norm = [train_cap_con_norm[i:j,:] for i,j in zip(size_ind_train[:-1], size_ind_train[1:])]
    
    train_x_norm = [train_x_norm, train_t_norm, train_cap_norm]
    
    #### valdiation input data normalization
    fold_valid_data = data_folds[fold_val_ind]
    valid_x_con_norm = x_scaler.transform(np.concatenate(fold_valid_data[0]))
    valid_t_con_norm = t_scaler.transform(np.concatenate(fold_valid_data[1]))
    valid_cap_con_norm = cap_scaler.transform(np.concatenate(fold_valid_data[2]))
    
    size_ind_valid = np.cumsum([len(i) for i in fold_valid_data[0]])
    size_ind_valid = np.insert(size_ind_valid, 0, 0)
    valid_x_norm = [valid_x_con_norm[i:j,:] for i,j in zip(size_ind_valid[:-1], size_ind_valid[1:])]
    valid_t_norm = [valid_t_con_norm[i:j,:] for i,j in zip(size_ind_valid[:-1], size_ind_valid[1:])]
    valid_cap_norm = [valid_cap_con_norm[i:j,:] for i,j in zip(size_ind_valid[:-1], size_ind_valid[1:])]
    
    valid_x_norm = [valid_x_norm, valid_t_norm, valid_cap_norm]
    
    #### test input data normalization
    fold_test_data = data_folds[fold_test_ind]
    test_x_con_norm = x_scaler.transform(np.concatenate(fold_test_data[0]))
    test_t_con_norm = t_scaler.transform(np.concatenate(fold_test_data[1]))
    test_cap_con_norm = cap_scaler.transform(np.concatenate(fold_test_data[2]))
    
    size_ind_test = np.cumsum([len(i) for i in fold_test_data[0]])
    size_ind_test = np.insert(size_ind_test, 0, 0)
    test_x_norm = [test_x_con_norm[i:j,:] for i,j in zip(size_ind_test[:-1], size_ind_test[1:])]
    test_t_norm = [test_t_con_norm[i:j,:] for i,j in zip(size_ind_test[:-1], size_ind_test[1:])]
    test_cap_norm = [test_cap_con_norm[i:j,:] for i,j in zip(size_ind_test[:-1], size_ind_test[1:])]
    
    test_x_norm = [test_x_norm, test_t_norm, test_cap_norm]
    
    #### rul data normalization (individually) 
    train_data_rul_con = np.concatenate(train_data[3])
    valid_data_rul_con = np.concatenate(fold_valid_data[3])
    test_data_rul_con = np.concatenate(fold_test_data[3])
    
    train_data_rul = [train_data_rul_con[i:j,:] for i,j in zip(size_ind_train[:-1], size_ind_train[1:])]
    valid_data_rul = [valid_data_rul_con[i:j,:] for i,j in zip(size_ind_valid[:-1], size_ind_valid[1:])]
    test_data_rul = [test_data_rul_con[i:j,:] for i,j in zip(size_ind_test[:-1], size_ind_test[1:])]
    
    train_data_rul_norm = [np.linspace(1,0,len(i)).reshape(-1,1) for i in train_data_rul]
    valid_data_rul_norm = [np.linspace(1,0,len(i)).reshape(-1,1) for i in valid_data_rul]
    test_data_rul_norm = [np.linspace(1,0,len(i)).reshape(-1,1) for i in test_data_rul]
    
    #### slope data
    train_data_slope = [np.zeros((len(i),1))-j[0]/i[-1] for i, j in zip(train_t_norm,train_data_rul_norm)]
    valid_data_slope = [np.zeros((len(i),1))-j[0]/i[-1] for i, j in zip(valid_t_norm,valid_data_rul_norm)]
    test_data_slope = [np.zeros((len(i),1))-j[0]/i[-1] for i, j in zip(test_t_norm,test_data_rul_norm)]
    
    #%% zero-padding of training data
    def zero_padding(data):
        size_seq = [len(i) for i in data]
        max_len = max(size_seq)
        data_padding = [np.pad(k, [(0, max_len-len(k)), (0, 0)], mode='constant', \
                               constant_values=0) for k in data]
        return data_padding, np.array(size_seq)
    
    train_data_slope_padding, train_len_seq = zero_padding(train_data_slope)
    train_data_rul_norm_padding = zero_padding(train_data_rul_norm)[0]
    train_x_norm_padding = [zero_padding(i)[0] for i in train_x_norm] # x,t,cap
    
    valid_data_slope_padding, valid_len_seq = zero_padding(valid_data_slope)
    valid_data_rul_norm_padding = zero_padding(valid_data_rul_norm)[0]
    valid_x_norm_padding = [zero_padding(i)[0] for i in valid_x_norm] # x,t,cap
    
    #%% PINN definition LSTM architecture
    input_size = 23
    hidden_size = 16
    layer_LSTM = 2
    layers_FC=[hidden_size+1, int(np.around((hidden_size+1)/2)),1]
    
    # training parms
    learning_rate = 0.001
    batch_size = 16
    n_epoch = 100
    seq_criterion = nn.MSELoss().to(device)
    
    # PINN hyper
    w = 0
    w2 = 0
    tol = 0
    
    #%% PINN TRAINING
    
    w_seq = [best_w_w2[num_][0]]
    w2_seq = [best_w_w2[num_][1]]
    
    X_train = [np.array(train_x_norm_padding[0]), train_len_seq]
    t_train = np.array(train_x_norm_padding[1])
    u_train = np.array(train_data_rul_norm_padding)
    s_train = np.array(train_data_slope_padding)
    
    X_valid = [np.array(valid_x_norm_padding[0]), valid_len_seq]
    t_valid = np.array(valid_x_norm_padding[1])
    u_valid = np.array(valid_data_rul_norm_padding)
    s_valid = np.array(valid_data_slope_padding)
    
    setup_seed(20)
    for j,w in enumerate(w_seq): 
        
        for k,w2 in enumerate(w2_seq): 
            # w = 0.1
            # CRA, score, rmse, ASI = obj(w)
            
            print('fold num:', fold_val_ind)
            print('w num:', w)
            print('w2 num:', w2)
        
            setup_seed(22)
            # pinn
            model = PhysicsInformedNN(X_train, t_train,  u_train\
                                       ,s_train, input_size, hidden_size, layers_FC, layer_LSTM, w, w2, learning_rate, fold_val_ind)
            # early stopping
            early_stopping = EarlyStopping(fold_val_ind,w,layer_LSTM, hidden_size, patience=50, verbose=True, delta=0)
            # training
            losses_train, losses_mse_train, losses_phy_train, losses_phy_2_train,\
            losses_val, losses_mse_val, losses_phy_val, losses_phy_2_val \
            = model.train(n_epoch, X_valid, t_valid, u_valid, \
                          s_valid, tol, early_stopping)
            # model.train(n_epoch, fold_early[0], fold_early[2], fold_early[3], fold_early[4], tol, early_stopping)
            
            # save model
            # creat dir path
            path = 'optimal_models'
            os.makedirs(path, exist_ok = True) 
            # 这里会存储迄今最优模型的参数
            torch.save(model.LSTM.state_dict(), 'optimal_models/model_parms_full_{0}_{1}.pt'.format(w, w2))    
            # 这里会存储迄今最优的模型
            torch.save(model.LSTM, 'optimal_models/model_cv_full_{0}_{1}.pkl'.format(w, w2))
            
            # path_loss = 'grid_search_loss'
            # os.makedirs(path_loss, exist_ok = True) 
            # np.save('grid_search_loss/losses_mse_train_{0}_{1}_{2}'.format(fold_val_ind,w, tol), losses_mse_train)  
            # np.save('grid_search_loss/losses_phy_train_{0}_{1}_{2}'.format(fold_val_ind,w, tol), losses_phy_train)  
            # np.save('grid_search_loss/losses_mse_val_{0}_{1}_{2}'.format(fold_val_ind,w, tol), losses_mse_val)  
            # np.save('grid_search_loss/losses_phy_val_{0}_{1}_{2}'.format(fold_val_ind,w, tol), losses_phy_val)  
            
            # print('end')
            
            # plot loss
            # plt.figure()
            # plt.plot(losses_train)
            # plt.plot(losses_val)
            # plt.show()
            
            # plt.figure()
            # plt.plot(losses_mse_train)
            # plt.plot(losses_mse_val)
            # plt.show()
            
            # plt.figure()
            # plt.plot(losses_phy_train)
            # plt.plot(losses_phy_val)
            # plt.show()
            
            # store loss decreasing
            # loss_store = [losses_train, losses_val]
            # np.save('grid_search_results/loss_{0}'.format(w), loss_store)  
        
            ####### loading optimal model and test
            # opt_model_pinn = PhysicsInformedNN(fold_train[0], fold_train[2], fold_train[3] \
            #                       ,fold_train[4], input_size, hidden_size, layers_FC, layer_LSTM, w, w2, learning_rate)
            # opt_model_pinn.LSTM.load_state_dict(torch.load('grid_search_results/model_parms_{0}.pt'.format(w)))
            # validation
            # model.LSTM.eval()
            # y_pred, f_pred = model.predict(fold_early[0][0], fold_early[2], fold_early[0][1])
        
            # # predictions
            # y_pred_set = [i[:j] for i,j in zip(y_pred, fold_early[0][1])]
            # # y_pred_all_pinn = np.concatenate(y_pred_set)
            
            # # predictions inverse normalization
            # y_pred_innor = [j*(len(i)-1) for i, j  in zip(y_actual,y_pred_set)]
            # y_pred_innor_all = np.concatenate(y_pred_innor)
            
            # plt.figure()
            # plt.plot(y_actual_innor_all)
            # plt.plot(y_pred_innor_all)
            # plt.show()
            
            # val_size = fold_early[0][1]
            # # 4 metrics
            # CRA, score, rmse, ASI = rul_metrics(y_actual_innor_all, y_pred_innor_all, val_size)


































