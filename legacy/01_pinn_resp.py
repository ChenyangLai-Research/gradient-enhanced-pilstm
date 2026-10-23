# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:49:33 2022

@author: LAI Chenyang
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import random
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error as mse
from torch.optim import lr_scheduler

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

# the deep neural network
class myLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_dim, layers, n_layers_LSTM):
        
        setup_seed(22)
        
        super(myLSTM, self).__init__()

        # add LSTM layers
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers_LSTM
        self.lstm = torch.nn.LSTM(input_size, hidden_dim, self.n_layers, batch_first=True)  
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = torch.nn.Tanh
        
        layer_list = list()
        
        # add full-connectted layers
        layer_list.append(('activation_%d' % 0, self.activation()))
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % (i+1), self.activation()))
        
        layer_list.append(
            ('layer_%d' % (self.depth-1), torch.nn.Dropout(0.1))
        )  
        
        layer_list.append(
            ('layer_%d' % (self.depth), torch.nn.Linear(layers[-2], layers[-1]))
        )
        

        
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x, t, len_seq):
        
        packed_input = pack_padded_sequence(input=x, lengths=len_seq, batch_first=True, enforce_sorted=False)
        
        out_lstm, h_c_n = self.lstm(packed_input)
        
        out_lstm, _ = pad_packed_sequence(out_lstm, batch_first=True)
        
        t_adapt = t[:,:out_lstm.shape[1],:]
        
        out_lstm_t = torch.cat([out_lstm, t_adapt], dim=2)
        # out = self.layers(h_c_n[0][-1,:,:])
        out = self.layers(out_lstm_t)
        
        return out
    
# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X, t, u, s, input_size, hidden_size, layers, n_layers_LSTM, w, w2, learning_rate,fold_val_ind):
        
        setup_seed(22)
        
        # data
        self.x_u = torch.tensor(X[0], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(t, requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.s = torch.tensor(s).float().to(device)
        
        self.len_seq = torch.tensor(X[1]).float()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.w = w
        self.w2 = w2
        self.fold_val_ind = fold_val_ind
        
        # deep neural networks
        self.LSTM = myLSTM(input_size, hidden_size, layers, n_layers_LSTM).to(device)

        self.optimizer_Adam = torch.optim.Adam(self.LSTM.parameters(), lr=learning_rate)
        
    def net_u(self, x, t, len_traj):  
        u = self.LSTM(x, t, len_traj)
        return u
    
    def net_f(self, x, t, len_traj):
        """ The pytorch autograd version of calculating residual """
        # prediction
        u = self.net_u(x, t, len_traj)
        
        # t_adapt = t[:,:u.shape[1],:]
        
        # all gradients btw u and x
        u_diff = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_diff2 = torch.autograd.grad(
            u_diff, t, 
            grad_outputs=torch.ones_like(u_diff),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_diff = u_diff[:,:u.shape[1],:]
        u_diff2 = u_diff2[:,:u.shape[1],:]
        # u_diff2 = 0
        
        return u_diff , u_diff2
    
    def train(self, nIter, X_val, t_val, u_val, s_val, tolerance, early_stopping):
        
        self.LSTM.train()
        x_val = torch.tensor(X_val[0], requires_grad=True).float().to(device)
        t_val = torch.tensor(t_val, requires_grad=True).float().to(device)
        u_val = torch.tensor(u_val).float().to(device)
        s_val = torch.tensor(s_val).float().to(device)
        
        len_seq_val = torch.tensor(X_val[1]).float()
        
        # nn training
        losses_train = torch.zeros(nIter).to(device)
        losses_mse_train = torch.zeros(nIter).to(device)
        losses_phy_train = torch.zeros(nIter).to(device)
        losses_phy_2_train = torch.zeros(nIter).to(device)
    
        losses_val = torch.zeros(nIter).to(device)
        losses_mse_val = torch.zeros(nIter).to(device)
        losses_phy_val = torch.zeros(nIter).to(device)
        losses_phy_2_val = torch.zeros(nIter).to(device)
        
        num_trajs_train = len(self.len_seq)
        
        batch_size = 16
        batch_num = int(np.ceil(num_trajs_train/batch_size))
        
        relu = torch.nn.ReLU()
        
        scheduler = lr_scheduler.CyclicLR(self.optimizer_Adam, base_lr=0.0005,max_lr=0.008, step_size_up=20, step_size_down=10, cycle_momentum=False)
        
        for epoch in range(nIter):
            
            self.optimizer_Adam.step()
            scheduler.step()
                        
            for i in range(batch_num):
                
                # print(scheduler.get_lr()[0])
                
                batch_mse_loss_train = 0.
                batch_phy_loss_train = 0.
                batch_phy_2_loss_train = 0.
                batch_loss_train = 0.
                
                # select data
                if i<(batch_num-1):
                    x_traj = self.x_u[i*batch_size:(i+1)*batch_size,:,:]
                    t_traj = self.t_u[i*batch_size:(i+1)*batch_size,:,:]
                    u_traj = self.u[i*batch_size:(i+1)*batch_size,:,:]
                    s_traj = self.s[i*batch_size:(i+1)*batch_size,:,:]
                    len_traj =  self.len_seq[i*batch_size:(i+1)*batch_size]
                else:
                    x_traj = self.x_u[i*batch_size:,:,:]
                    t_traj = self.t_u[i*batch_size:,:,:]
                    u_traj = self.u[i*batch_size:,:,:]
                    s_traj = self.s[i*batch_size:,:,:]
                    len_traj =  self.len_seq[i*batch_size:]                               
                
                # zero the gradient
                self.optimizer_Adam.zero_grad()
                
                # prediction
                with torch.backends.cudnn.flags(enabled=False):
                    u_pred = self.net_u(x_traj,t_traj, len_traj)
                    f_pred, f_pred_2 = self.net_f(x_traj,t_traj,len_traj)
                
                # mixed loss pure PINN
                mse_loss = torch.mean((u_traj[:,:u_pred.shape[1],:] - u_pred) ** 2)
                phy_loss = torch.mean((s_traj[:,:u_pred.shape[1],:] - f_pred) ** 2) 
                # tolerance-based
                # phy_loss = torch.mean(relu(torch.abs(s_traj[:,:u_pred.shape[1],:] - f_pred)**2 - tolerance * (-s_traj[:,:u_pred.shape[1],:])**2))
                # gradien-enhanced term
                phy_loss_2 = torch.mean((f_pred_2[:,:u_pred.shape[1],:]) ** 2)
                
                # plt.plot(u_traj[0,:,:].cpu().detach().numpy())
                # plt.plot(t_traj[0,:,:].cpu().detach().numpy())
                # plt.plot(s_traj[0,:,:].cpu().detach().numpy())
                # plt.plot(u_pred[0,:,:].cpu().detach().numpy())
                
                loss = (1-self.w) *  mse_loss + self.w *  phy_loss + self.w2 * phy_loss_2

                
                # Backward and optimize
                loss.backward()
                self.optimizer_Adam.step()
                
                batch_mse_loss_train += mse_loss.item()
                batch_phy_loss_train += phy_loss.item()
                batch_phy_2_loss_train += phy_loss_2.item()
                batch_loss_train += loss.item()
                
            # training loss per epoch 
            batch_mse_loss_train /= batch_num
            batch_phy_loss_train /= batch_num
            batch_phy_2_loss_train /= batch_num
            batch_loss_train /= batch_num
            losses_mse_train[epoch] = batch_mse_loss_train
            losses_phy_train[epoch] = batch_phy_loss_train
            losses_phy_2_train[epoch] = batch_phy_2_loss_train
            losses_train[epoch] = batch_loss_train
                
            # print(epoch)
            
            ######################
            # validate the model #
            ######################
            self.LSTM.eval() # prep model for evaluation
            
            num_trajs_valid = len(len_seq_val)
        
            batch_num_valid = int(np.ceil(num_trajs_valid/batch_size))
            
            for i in range(batch_num_valid):
                
                batch_mse_loss_valid = 0.
                batch_phy_loss_valid = 0.
                batch_phy_2_loss_valid = 0.
                batch_loss_valid = 0.
                
                # select data
                if i<(batch_num-1):
                    x_traj = x_val[i*batch_size:(i+1)*batch_size,:,:]
                    t_traj = t_val[i*batch_size:(i+1)*batch_size,:,:]
                    u_traj = u_val[i*batch_size:(i+1)*batch_size,:,:]
                    s_traj = s_val[i*batch_size:(i+1)*batch_size,:,:]
                    len_traj =  len_seq_val[i*batch_size:(i+1)*batch_size]
                else:
                    x_traj = x_val[i*batch_size:,:,:]
                    t_traj = t_val[i*batch_size:,:,:]
                    u_traj = u_val[i*batch_size:,:,:]
                    s_traj = s_val[i*batch_size:,:,:]
                    len_traj =  len_seq_val[i*batch_size:] 
                
                # prediction
                with torch.backends.cudnn.flags(enabled=False):
                    u_pred = self.net_u(x_traj,t_traj, len_traj)
                    f_pred, f_pred_2 = self.net_f(x_traj,t_traj,len_traj)
                    
                # mixed loss
                mse_loss_val = torch.mean((u_traj[:,:u_pred.shape[1],:] - u_pred) ** 2)
                phy_loss_val = torch.mean((s_traj[:,:u_pred.shape[1],:] - f_pred) ** 2) 
                # tolerance-based
                # phy_loss_val = torch.mean(relu(torch.abs(s_traj[:,:u_pred.shape[1],:] - f_pred)**2 - tolerance * (-s_traj[:,:u_pred.shape[1],:])**2))
                # gradien-enhanced term
                phy_loss_2_val = torch.mean((f_pred_2) ** 2)
                loss_val = (1-self.w) * mse_loss_val + self.w * phy_loss_val + self.w2 * phy_loss_2_val     
                    
                batch_mse_loss_valid += mse_loss_val.item()
                batch_phy_loss_valid += phy_loss_val.item()
                batch_phy_2_loss_valid += phy_loss_2_val.item()
                batch_loss_valid += loss_val.item()
                    
            # val loss per epoch 
            batch_mse_loss_valid /= batch_num_valid
            batch_phy_loss_valid /= batch_num_valid
            batch_phy_2_loss_valid /= batch_num_valid
            batch_loss_valid /= batch_num_valid
            losses_mse_val[epoch] = batch_mse_loss_valid
            losses_phy_val[epoch] = batch_phy_loss_valid
            losses_phy_2_val[epoch] = batch_phy_2_loss_valid
            losses_val[epoch] = batch_loss_valid
            
            # print loss
            print(epoch, ':', batch_loss_train, ',', batch_loss_valid)
            # print(epoch, ':', loss_train.item())
            
            # # batch_loss_ = loss_test
            # batch_loss_ = batch_mse_loss_valid
            # early_stopping(batch_loss_, self.LSTM)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
        
            # if epoch>99:
            #     early_stopping(batch_loss_, self.LSTM)
            #     if early_stopping.early_stop:
            #         print("Early stopping")
            #         break
                    
        # creat dir path
        # path = 'grid_search_model'
        # os.makedirs(path, exist_ok = True) 

        # # 这里会存储迄今最优模型的参数
        # torch.save(self.LSTM.state_dict(), 'grid_search_model/model_parms_{0}_{1}.pt'.format(self.fold_val_ind,self.w))    
        # # 这里会存储迄今最优的模型
        # torch.save(self.LSTM, 'grid_search_model/model_cv_{0}_{1}.pkl'.format(self.fold_val_ind,self.w))
        
        #return np.array(losses_train.cpu()[:epoch]), np.array(losses_mse_train.cpu()[:epoch]), np.array(losses_phy_train.cpu()[:epoch]), np.array(losses_phy_2_train.cpu()[:epoch]), np.array(losses_val.cpu()[:epoch]), np.array(losses_mse_val.cpu()[:epoch]), np.array(losses_phy_val.cpu()[:epoch]), np.array(losses_phy_2_val.cpu()[:epoch])
        return np.array(losses_train.cpu()[:epoch]), np.array(losses_mse_train.cpu()[:epoch]),\
               np.array(losses_phy_train.cpu()[:epoch]), np.array(losses_phy_2_train.cpu()[:epoch]),\
                np.array(losses_val.cpu()[:epoch]),  np.array(losses_mse_val.cpu()[:epoch]), \
                np.array(losses_phy_val.cpu()[:epoch]), np.array(losses_phy_2_val.cpu()[:epoch])
               

    def predict(self, X, t,len_seq_test):
        
        x = torch.tensor(X, requires_grad=True).float().to(device)
        t = torch.tensor(t, requires_grad=True).float().to(device)
        len_seq = torch.tensor(len_seq_test).float()

        self.LSTM.eval()

        u = self.net_u(x, t, len_seq)
        f, f2 = self.net_f(x, t, len_seq)
            
        return u.cpu().detach().numpy(), f.cpu().detach().numpy()




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, fold_val_ind, w, layer_LSTM, hidden_size, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.fold_val_ind = fold_val_ind
        self.w = w
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.layer_LSTM = layer_LSTM
        self.hidden_size = hidden_size

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # creat dir path
        path = 'grid_search_model'
        os.makedirs(path, exist_ok = True) 

        # 这里会存储迄今最优模型的参数
        torch.save(model.state_dict(), 'grid_search_model/model_parms_{0}_{1}.pt'.format(self.fold_val_ind,self.w))    
        # 这里会存储迄今最优的模型
        torch.save(model, 'grid_search_model/model_cv_{0}_{1}.pkl'.format(self.fold_val_ind,self.w))
                 
        self.val_loss_min = val_loss
        
def rul_metrics(rul_actual, rul_est, test_size_seq):
    
    # RMSE
    rmse=np.sqrt(mse(rul_actual, rul_est))
    # CRA: cumulative relative accuracy
    
    y_ind = np.where(rul_actual != 0)[0]
    rul_est_no=rul_est[y_ind]
    rul_actual_no=rul_actual[y_ind]
    
    ra=1-np.abs(rul_est_no-rul_actual_no)/rul_actual_no
    CRA=np.sum(ra)/np.size(ra)
    # Alpha-Lambda
    alpha=0.2
    score=0
    for k in range(np.size(rul_est)):
        if (rul_actual[k]*(1-alpha))<=rul_est[k] and (rul_actual[k]*(1+alpha))>=rul_est[k]:
            score=score+1
    score=score/np.size(rul_est)
    # ASI: Average Steadiness Index
    seq = np.cumsum(test_size_seq)
    seq = np.insert(seq, 0, 0)
    seq = [int(i) for i in seq]
    var_all = []
    for ind_start, ind_end in zip(seq[:-1], seq[1:]):
        real = rul_actual[ind_start:ind_end]
        est = rul_est[ind_start:ind_end]
        var_trag = []
        tw = 5 # time-window
        for i in range(len(real)-tw+1):
            var = np.std(real[i:i+tw]-est[i:i+tw])
            var_trag.append(var)
        var_all.append(np.mean(var_trag))
    ASI = np.mean(var_all)
    
    return CRA, score, rmse, ASI 
        
        
        
        
        
        
        
        
        
        
        
        
        