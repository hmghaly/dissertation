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
#cuda0 = torch.device('cuda:0') 

n_epochs=101

device_num=1

if device_num==0: device_name="cuda:0"
elif device_num==1: device_name="cuda:1"
else: device_name="cpu"


device = torch.device(device_name if torch.cuda.is_available() else "cpu")



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=1):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size

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
        return (torch.zeros(1, self.batch_size, self.hidden_size).to(device),
                torch.zeros(1, self.batch_size, self.hidden_size).to(device))



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



# def train(category_tensor, line_tensor):
#     hidden = rnn.initHidden()

#     rnn.zero_grad()
#     for i in range(line_tensor.size()[0]):
#         output, hidden = rnn(line_tensor[i], hidden)
    
#     loss = criterion(output, category_tensor)
#     loss.backward()

#     # Add parameters' gradients to their values, multiplied by learning rate
#     for p in rnn.parameters():
#         p.data.add_(-learning_rate, p.grad.data) #this one has problems with one word entries

#     return output, loss.item()


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
        line_tensor=torch.tensor(item_data)
        parser_uas=cur_uas=item_data.attrs["uas"]
        sent_size=item_data.attrs["sent_size"]
        category_tensor=torch.tensor([cur_uas]).view([1,1,1])
        sf_id="-".join(item.split("-")[:-1]) 
        parser_name=item.split("-")[-1]


    # for d_, di in enumerate(cur_set):
    #     if d_%100==0: print("testing, item", d_)
        #compressed_matrix,parser_uas,sf_id, sent_size, parser_name=di

        #line_tensor=inflate_one_hot(compressed_matrix, n_input)
        #category_tensor=torch.tensor([parser_uas]).view([1,1,1])


        # line_tensor=make_one_hot(cur_ft_dict,ft_size_dict,index_dict)
        # #print(line_tensor)
        # category_tensor=torch.tensor([parser_uas]).view([1,1,1])


        for i in range(line_tensor.size()[0]):
            cur_tensor=line_tensor[i]#.view([(1, 1, 3)])
            #cur_tensor=torch.randn(1, 1, n_input)
            cur_tensor=cur_tensor.view([1,1,n_input])

            output = rnn(cur_tensor)
        predicted=output[0][0].item()

        pred_items.append((sf_id, sent_size, parser_name, parser_uas,predicted))

        # print(sf_id, parser_name, parser_uas)
        # print(output, predicted)
        # print("-----")

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
    exp_name="exp11"

    #RNN setup
    #n_epochs=5
    n_hidden = 128
    n_hidden = 64
    n_hidden = 32
    n_hidden = 16
    LR = 0.0001           # learning rate
    #LR = 0.0005           # learning rate
    #LR = 0.0002           # learning rate
    #LR = 0.00005           # learning rate

    #directories and files speification
    exp_dir=os.path.join(os.getcwd(),exp_name)
    train_size=30000



    # train_fname="train.pickle"
    # dev_fname="dev.pickle"
    # test_fname="test.pickle"
    # parameters_fname="parameters.json"
    # train_fpath=os.path.join(exp_dir,train_fname)
    # dev_fpath=os.path.join(exp_dir,dev_fname)
    # test_fpath=os.path.join(exp_dir,test_fname)

    #get the parameters data
    parameters_fname="parameters.json"
    parameters_fpath=os.path.join(exp_dir,parameters_fname)
    parameters_fopen=open(parameters_fpath)
    content=parameters_fopen.read()
    parameters_fopen.close()
    exp_parameters=json.loads(content)
    #exp_parameters=parameters_dict
    print(exp_parameters)
    n_input=exp_parameters["n_input"]


    #get the actual train and dev data
    # train_items_list=lpk(train_fpath)
    # dev_items_list=lpk(dev_fpath)
    hdf5_fpath=os.path.join(exp_dir,"data.hdf5")
    hdf5_file=h5py.File(hdf5_fpath, 'r')    
    hdf5_train=hdf5_file["train"]
    hdf5_dev=hdf5_file["dev"]

    # train_items_list=[]

    # start_loading=time.time()
    # for i_,item in enumerate(hdf5_train):
    #     if i_%100==0: print(i_)
    #     item_data=hdf5_train[item]
    #     cur_input_tensor=torch.tensor(item_data)
    #     cur_uas=item_data.attrs["uas"]
    #     cur_sent_size=item_data.attrs["sent_size"]
    #     cur_category_tensor=torch.tensor([cur_uas]).view([1,1,1])
    #     train_items_list.append((item,cur_uas,cur_sent_size,cur_input_tensor,cur_category_tensor))

    # elapsed=time.time()-start_loading
    # print("loaded %s training items in %s seconds"%(len(train_items_list),elapsed))
        #print(item, item_data.shape, cur_uas, cur_sent_size)

    #now we start the network
    rnn = RNN(n_input, n_hidden, 1)
    rnn.to(device)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters

    current_loss = 0
    dev_loss=0
    all_losses = []

    #starting training
    print("starting training", len(hdf5_train))
    start_time=time.time()


    prev_time=time.time()
    for e0 in range(n_epochs):
        for i_,item in enumerate(hdf5_train):
            #if i_%100==0: print(i_)
            #if i_==50: break
            item_data=hdf5_train[item]
            # print(type(item_data))
            # print(len(item_data) )
            # print(item_data.file)
            # print(dir(item_data) )

            line_tensor=torch.tensor(item_data)
            #line_tensor=item_data
            cur_uas=item_data.attrs["uas"]
            cur_sent_size=item_data.attrs["sent_size"]
            category_tensor=torch.tensor([cur_uas]).view([1,1,1])
            #train_items_list.append((item,cur_uas,cur_sent_size,cur_input_tensor,cur_category_tensor))


        # for t_,ti in enumerate(train_items_list):
        #     if t_>100: break 
        #     #if t_%100==0: print(t_)
        #     compressed_matrix,parser_uas,sf_id, sent_size, parser_name=ti
        #     #print(parser_uas,sf_id, sent_size, parser_name)
        #     #print(compressed_matrix)

        #     line_tensor=inflate_one_hot(compressed_matrix, n_input)
        #     category_tensor=torch.tensor([parser_uas]).view([1,1,1])

            #the RNN part in the training
            rnn.hidden = rnn.init_hidden()
            rnn.zero_grad()
            for i in range(line_tensor.size()[0]):
                cur_tensor=line_tensor[i]#.view([(1, 1, 3)])
                cur_tensor=cur_tensor.view([1,1,n_input])
                output = rnn(cur_tensor)
            
            #loss = criterion(output, category_tensor)
            output=output.to(device)
            category_tensor=category_tensor.to(device)
            loss = loss_func(output, category_tensor)
            current_loss += loss  
            #print(current_loss)
            
            loss.backward()
            optimizer.step()
        # elapsed=time.time()-prev_time
        # prev_time=time.time()
        #print("training, epoch",e0, "item", t_, "elapsed", elapsed)


        
        for i_,item in enumerate(hdf5_dev):
            #if i_%100==0: print(i_)
            #if i_==100: break
            rnn.hidden = rnn.init_hidden()
            rnn.zero_grad()

            item_data=hdf5_dev[item]
            line_tensor=torch.tensor(item_data)
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
            category_tensor=category_tensor.to(device)
            cur_dev_loss = loss_func(output, category_tensor)
            dev_loss += cur_dev_loss

        elapsed=time.time()-prev_time
        prev_time=time.time()
        #print("%s\t%s\t%s\t%s"%(e0,current_loss.item(),dev_loss.item(),elapsed))
        avg_train_loss=float(current_loss.item())/len(hdf5_train)
        avg_dev_loss=float(dev_loss.item())/len(hdf5_dev)
        print("%s\t%s\t%s\t%s\t%s\t%s"%(e0,current_loss.item(),dev_loss.item(),avg_train_loss,avg_dev_loss,elapsed))
        current_loss = 0
        dev_loss=0
        if True:
        #if e0>0 and e0%10==0: #save model every 10 epochs
            #model_path=os.path.join(exp_dir,"model-%s.pth"%e0)
            model_sd_path=os.path.join(exp_dir,"model-sd-%s.pth"%e0)#state dict
            torch.save(rnn.state_dict(), model_sd_path)
            #torch.save(rnn, model_path)







        #print()
        #print("%s\t%s\t%s\t%s"%(e0,current_loss.item(),dev_loss.item(),elapsed))




    training_time=time.time()-start_time
    print("training completed in %s seconds"%training_time)
    model_path=os.path.join(exp_dir,"model-final.pth")

    # print(rnn)
    # pytorch_total_params = sum(p.numel() for p in rnn.parameters())
    # print("sum the number of elements for every parameter group", pytorch_total_params)

    # pytorch_total_params = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
    # print("only the trainable parameters", pytorch_total_params)

    torch.save(rnn, model_path)
    print("now predicting")
    dev_pred_list=run_testing(hdf5_dev)
    hdf5_file.close()



