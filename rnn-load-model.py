import os, sys, re, subprocess, time, pickle, json

from itertools import groupby
from collections import Counter
sys.path.append("../../utils")
from dep_lib import *


import torch
import torch.nn as nn
from torch.autograd import Variable

import h5py
import numpy as np

torch.manual_seed(1)

n_epochs=5

device_num=1

if device_num==0: device_name="cuda:0"
elif device_num==1: device_name="cuda:1"
else: device_name="cpu"
device = torch.device(device_name if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)      
        #self.lstm.cuda(cuda0)  
        self.hidden2out=nn.Linear(hidden_size,output_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()
        self.softmax = nn.LogSoftmax(dim=1) #new
        
    def forward(self, x):
        x=x.to(device)
        #self.hidden
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        outs = self.out(lstm_out)
        #outs = self.softmax(outs)
        return outs#, h_state


    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size).to(device),
                torch.zeros(1, 1, self.hidden_size).to(device))


def v2t(vec,integer=False): #build a tensor form a 2-d list
    if integer: tensor = torch.zeros(len(vec), 1, len(vec[0]), dtype=torch.long)
    else: tensor = torch.zeros(len(vec), 1, len(vec[0]))

    for vi, v1 in enumerate(vec):
        tensor[vi][0]=torch.tensor(v1)
        #tensor[vi][0]=torch.tensor(v1, dtype=torch.long)        
    return tensor

def inflate_one_hot(list_2d,cur_n):
    new_list=[]
    for l2 in list_2d:
        cur_row=[0]*cur_n
        for cell_i,cell_val in l2:
            cur_row[cell_i]=cell_val
        new_list.append(cur_row)
    cur_tensor=v2t(new_list)
    #cur_tensor=new_list
    return cur_tensor


def cpk(var1,file1):
    fopen=open(file1,'wb')
    pickle.dump(var1,fopen,protocol=2)
    fopen.close()

# UnPickle
def lpk(file1):
    fopen=open(file1,'rb')
    output=pickle.load(fopen)
    fopen.close()
    return output


def run_testing(cur_set): #to run for both devset and testset
    pred_items=[]
    print("Now testing", len(cur_set))

    for i_,item in enumerate(cur_set):
        if i_%100==0: print(i_)
        #if i_==10: brea
        rnn.hidden = rnn.init_hidden()
        rnn.zero_grad()

        item_data=cur_set[item]
        line_tensor=torch.tensor(item_data)#.to(device)
        #print(line_tensor)
        parser_uas=cur_uas=item_data.attrs["uas"]
        sent_size=item_data.attrs["sent_size"]
        category_tensor=torch.tensor([cur_uas]).view([1,1,1])
        sf_id="-".join(item.split("-")[:-1]) 
        parser_name=item.split("-")[-1]
        for i in range(line_tensor.size()[0]):
            cur_tensor=line_tensor[i]#.view([(1, 1, 3)])
            #cur_tensor=torch.randn(1, 1, n_input)
            cur_tensor=cur_tensor.view([1,1,n_input])

            output = rnn(cur_tensor)
        predicted=output[0][0].item()

        pred_items.append((sf_id, sent_size, parser_name, parser_uas,predicted))


    grouped_items=[(key,[v[1:] for v in list(group)]) for key,group in groupby(pred_items,lambda x:x[0])]
    cum_ensemble_pred=0
    cum_oracle_pred=0
    cum_total=0
    for k, grp in grouped_items:
        #print(k,grp)
        grp.sort(key=lambda x:-x[-1]) #get the prediction
        top=grp[0]
        cur_size=top[0]
        cur_uas=top[2]
        cur_ensemble=cur_uas*cur_size
        cum_ensemble_pred+=cur_ensemble

        grp.sort(key=lambda x:-x[2]) #get the oracle
        top=grp[0]
        cur_size=top[0]
        cur_uas=top[2]
        cur_oracle=cur_uas*cur_size
        cum_oracle_pred+=cur_oracle

        cum_total+=cur_size
        #print(grp)
        #print("-----")
    ensemble_ratio=float(cum_ensemble_pred)/cum_total
    oracle_ratio=float(cum_oracle_pred)/cum_total
    print("==========")
    print("training size", train_size, "n_epochs", n_epochs, "LR",LR)
    print(exp_parameters)
    print("predicted ratio:", ensemble_ratio)
    print("oracle ratio:", oracle_ratio)
    return pred_items




if __name__=="__main__":
    exp_name="exp10"
    model_name="model-sd-20.pth"

    #RNN setup
    #n_epochs=5
    n_hidden = 128
    LR = 0.005           # learning rate

    #directories and files speification
    exp_dir=os.path.join(os.getcwd(),exp_name)
    train_size=30000
    train_fname="train.pickle"
    dev_fname="dev.pickle"
    test_fname="test.pickle"
    parameters_fname="parameters.json"
    train_fpath=os.path.join(exp_dir,train_fname)
    dev_fpath=os.path.join(exp_dir,dev_fname)
    test_fpath=os.path.join(exp_dir,test_fname)

    #get the parameters data
    parameters_fpath=os.path.join(exp_dir,parameters_fname)
    parameters_fopen=open(parameters_fpath)
    content=parameters_fopen.read()
    parameters_fopen.close()
    parameters_dict=json.loads(content)
    print(parameters_dict)
    n_input=parameters_dict["n_input"]
    exp_parameters=parameters_dict

    hdf5_fpath=os.path.join(exp_dir,"data.hdf5")
    hdf5_file=h5py.File(hdf5_fpath, 'r')    
    hdf5_train=hdf5_file["train"]
    hdf5_dev=hdf5_file["dev"]
    hdf5_test=hdf5_file["test"]

    print("train", len(hdf5_train), "dev", len(hdf5_dev),"test",len(hdf5_test))

    #now loading the trained model
    model_path=os.path.join(exp_dir,model_name)
    rnn = RNN(n_input, n_hidden, 1)
    rnn.to(device)
    rnn.load_state_dict(torch.load(model_path))
    rnn.eval()
    #print(rnn)



    #get the actual train and dev data
    #train_items_list=lpk(train_fpath)
    #dev_items_list=lpk(dev_fpath)
    #test_items_list=lpk(test_fpath)

    
    
    #rnn = torch.load(model_path)
    
    

    print("evaluating devset")
    #run_testing(dev_items_list)
    dev_pred_list=run_testing(hdf5_dev)
    
    print("evaluating testset")
    dev_pred_list=run_testing(hdf5_test)

    hdf5_file.close()



