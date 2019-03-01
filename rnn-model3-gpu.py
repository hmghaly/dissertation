import os, sys, re, subprocess, time

from itertools import groupby
from collections import Counter
sys.path.append("../../utils")
from dep_lib import *


import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(1)
#cuda0 = torch.device('cuda:0') 

n_epochs=100

device_num=1

if device_num==0: device_name="cuda:0"
elif device_num==1: device_name="cuda:1"
else: device_name="cpu"

# device0_name="cuda:0"
# device1_name="cuda:1"
#device_name=device0_name #or device 0


device = torch.device(device_name if torch.cuda.is_available() else "cpu")

# device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")




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


def make_one_hot(input_ft_dict,ft_size_dict,ft_index_dict):
    one_hot_matrix=[]
    keys=list(sorted(cur_ft_dict.keys(),reverse=True))
    for i in range(sent_size):
        cur_row=[]
        for k in keys:
            cur_val=cur_ft_dict[k][i]
            #print(i,k,cur_val)
            one_hot_size=ft_size_dict.get(k)
            if one_hot_size==None: cur_row.append(cur_val)
            else:
                cur_zeros=[0]*one_hot_size
                #(ft_name,val0)
                cur_index=index_dict[(k,cur_val)]
                cur_zeros[cur_index]=1
                cur_row.extend(cur_zeros)
        one_hot_matrix.append(cur_row)
    one_hot_tensor=v2t(one_hot_matrix)
    return one_hot_tensor



def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()
    # print(rnn)
    # return

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    #print(output,category_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        #print(p.size(),p.grad)
        p.data.add_(-learning_rate, p.grad.data) #this one has problems with one word entries

    return output, loss.item()

def printable_2d(list_2d):
    return "\n".join(["\t".join([str(k) for k in v]) for v in list_2d])

def file2dict(fpath,do_split_chunk=False): #to read the text files and convert them into dictionaries where the key is the file/sentence ID
    cur_dict={}
    fopen=open(fpath)
    chunk=""
    for f in fopen:
        if len(f)<2:
            chunk_split=chunk.split("\n")
            chunk=""
            chunk_header=chunk_split[0]
            chunk_conll="\n".join(chunk_split[1:])
            chunk_header_split=chunk_header.strip("*").split("\t")
            sf_id=chunk_header_split[0]
            f_id=int(sf_id.split(".")[0][2:])
            if do_split_chunk:
                #indexes=[0]+[vi for vi,v in enumerate(chunk_split) if v and v[0]=="*"]+[len(chunk_split)]
                indexes=[vi for vi,v in enumerate(chunk_split) if v and v[0]=="*"]+[len(chunk_split)]
                #print indexes
                indexes=sorted(list(set(indexes)))
                #print indexes
                temp_dict={}
                sent_size=indexes[1]-1 
                for i0,i1 in zip(indexes,indexes[1:]):
                    cur_parser_header=chunk_split[i0]
                    cur_parser_header_split=cur_parser_header.split("\t")
                    parser_name,parser_uas=cur_parser_header_split[-2],cur_parser_header_split[-1]
                    cur_parser_conll="\n".join(chunk_split[i0+1:i1])
                    temp_dict[parser_name]=(parser_uas,sent_size,cur_parser_conll)
                cur_dict[sf_id]=temp_dict
            else:
                cur_dict[sf_id]=chunk_conll
        else:
            chunk+=f
    fopen.close()
    return cur_dict

def strip_tobi(word_with_tobi):
    word=word_with_tobi.strip("-")
    word_split=word.split("-")
    if word_split[-1] and word_split[-1][0].isdigit(): word="-".join(word_split[:-1])
    return word

def extract_features(input_conll,input_acoustics,parameters={}):
    cur_ft_dict={}
    conll_obj=conll(input_conll)
    
    #lengthening_list=[str(float(v[2])>=1) for v in input_acoustics]
    #dur_str_list=[str(round10(float(v[2]) )) for v in input_acoustics]
    #pause_str_list=[str(float(v[-1])>0) for v in input_acoustics]
    if parameters.get("ft-word-word",False):
        cur_word_pairs=conll_obj.word_dep_pairs
        cur_ft_dict["words"]=[strip_tobi(v[0]) for v in cur_word_pairs]
        cur_ft_dict["heads"]=[strip_tobi(v[1]) for v in cur_word_pairs]
    if parameters.get("ft-tobi",False):
        cur_ft_dict["tobi"]=[v[1] for v in input_acoustics]
        #cur_ft_dict["tobi_num"]=tobi_list=[float(v[1][0]) for v in input_acoustics]
        #cur_ft_dict["tobi_p"]=[1 if v[1][-1].lower()=="p" else 0 for v in input_acoustics]
        #cur_ft_dict["tobi_p"]=tobi_list=[v[1][-1].lower()=="p" for v in input_acoustics]
    if parameters.get("ft-dur",False):
        cur_ft_dict["dur"]=[float(v[2]) for v in input_acoustics]
    if parameters.get("ft-dur-diff",False):
        tmp_dur_diff_list=[1]
        for i,a in enumerate(input_acoustics[:-1]):
            cur_dur=float(a[2])
            next_dur=float(input_acoustics[i+1][2])
            if next_dur>cur_dur: cur_diff=1
            else: cur_diff=-1
            tmp_dur_diff_list.append(cur_diff)
        cur_ft_dict["dur-diff"]=tmp_dur_diff_list
        # print([float(v[2]) for v in input_acoustics])
        # print(tmp_dur_diff_list)
        # print("------")



    if parameters.get("ft-pause",False):
        cur_ft_dict["pause"]=[float(v[-1]) for v in input_acoustics]
    if parameters.get("ft-configs",False):
        dep_configs=conll_obj.list_dep_configs
        if dep_configs[-1]==(): dep_configs[-1]=(-2,2)
        cur_ft_dict["configs"]=dep_configs
    if parameters.get("ft-links",False):
        dep_links=conll_obj.list_link_configs
        if dep_links[-1]==(): dep_links[-1]=(-1,-1,-1)
        cur_ft_dict["links"]=dep_links
    if parameters.get("ft-brackets",False):
        num_brackets=[v[-1] for v in conll_obj.num_brackets]
        #if dep_links[-1]==(): dep_links[-1]=(-1,-1,-1)
        cur_ft_dict["brackets"]=num_brackets
        #print(num_brackets)

    #self.num_brackets


        #ft_items=["%s-%s"%(strip_tobi(v[0]),strip_tobi(v[1]) ) for v in cur_word_pairs]
    return cur_ft_dict

def get_ft_index(ft_name,ft_val,ft_index_dict):
    cur_val_list=ft_index_dict.get(ft_name,[])
    if ft_val in cur_val_list:  cur_index=cur_val_list.index(ft_val)
    else: 
        cur_index=len(cur_val_list)
        cur_val_list.append(ft_val)
        ft_index_dict[ft_name]=cur_val_list
    return cur_index, ft_index_dict




if __name__=="__main__":
    
    train_size=30000
    
    
    # Keep track of losses for plotting
    list_parameters=[]
    exp_parameters={}
    exp_parameters["ft-word-word"]=True
    exp_parameters["ft-configs"]=True
    exp_parameters["ft-links"]=False
    exp_parameters["ft-brackets"]=False
    #cur_exp_params=dict(exp_parameters)
    #list_parameters.append(cur_exp_params)

    #prosodic featues
    exp_parameters["ft-tobi"]=True
    exp_parameters["ft-dur"]=False
    exp_parameters["ft-pause"]=False
    exp_parameters["ft-dur-diff"]=False
    #list_parameters.append(exp_parameters)
    #list_parameters.reverse()

    category_features=["words","heads","tobi","configs","links"] #we convert their values to indexes, use one-hot
    #for exp_parameters in list_parameters:

    gold_fpath="gold-dep-tobi.txt"
    gold_dict=file2dict(gold_fpath)

    acoustics_fpath="acoustics-new.txt"
    acoustics_dict=file2dict(acoustics_fpath)

    parsers_output_fpath="parser-output-tobi.txt"
    parser_output_dict=file2dict(parsers_output_fpath,do_split_chunk=True)
    print("finished processing")
    train_set,dev_set,test_set,held_set=[],[],[],[]

    for s_i,sf_id in enumerate(parser_output_dict):
        f_id=int(sf_id.split(".")[0][2:])
        if f_id>=4519 and f_id<=4936: dev_set.append(sf_id)#set_name="dev"
        elif f_id>=4004 and f_id<=4153: test_set.append(sf_id)#set_name="test"             
        elif f_id>=4154 and f_id<=4483: held_set.append(sf_id)#set_name="held" #dev/held           
        else: train_set.append(sf_id)# set_name="train"

    #print(len(train_set),train_set[:10])

    sent_size_dict={}
    ft_dict={}
    index_dict={}

    train_items=[]
    dev_items=[]
    

    print("extracting features from sentences")
    for si,sf_id in enumerate(train_set+dev_set):
        if si%500==0: print("current sentence", si)
        #if si>200: break

        
        gold_conll=gold_dict.get(sf_id)

        cur_acoustics=acoustics_dict[sf_id]
        cur_acoustics_2d=[v.split("\t") for v in cur_acoustics.split("\n")[1:] if v]
        cur_acoustics_2d_headers=[v.split("\t") for v in cur_acoustics.split("\n") if v]


        sent_size=len(cur_acoustics_2d)
        sent_size_dict[sf_id]=sent_size


        parsers_out=parser_output_dict[sf_id]
        parsers_out_list=[]
        parser_matrixes=[]
        all_uas_vals=[]
        for parser_name in parsers_out:
            items=parsers_out[parser_name]
            parser_uas,sent_size,cur_parser_conll=items
            parser_uas=float(parser_uas)
            # if not parser_uas in all_uas_vals: all_uas_vals.append(parser_uas)
            # else: continue

            # cur_ft_dict={}
            # cur_ft_dict["uas"]=[parser_uas]* sent_size

            #print(sf_id, parser_name, parser_uas)
            cur_ft_dict=extract_features(cur_parser_conll, cur_acoustics_2d, exp_parameters)
            #print(cur_ft_dict)
            #print("-----")
            #print(">>>>>>", cur_ft_dict)
            for ft_name in cur_ft_dict:
                vals=cur_ft_dict[ft_name]
                #print(">>>>", ft_name, vals)
                if ft_name in category_features: 
                    for val0 in vals:
                        val_index, ft_dict = get_ft_index(ft_name,val0,ft_dict)
                        index_dict[(ft_name,val0)]=val_index #easy access to the index of each feature-value pair

            #print("--------")

            cur_item=(cur_ft_dict,sf_id, sent_size, parser_name, parser_uas)
            if si<len(train_set): train_items.append(cur_item)
            else: dev_items.append(cur_item)

            # if si<train_size: train_items.append(cur_item)
            # else: dev_items.append(cur_item)

            
    print(len(train_items),len(dev_items))
    #sys.exit()
    
    ft_size_dict={} #this is to know the number of catgories in categorical features
    for k,v in ft_dict.items():
       #print(k, len(v))
       ft_size_dict[k]=len(v)
    #print(ft_size_dict)

    n_input=0 #now we identify the number of inputs to the RNN based on the size allocated to each feature
    first_ft_dict=train_items[0][0]
    for ft0 in first_ft_dict:
        size0=ft_size_dict.get(ft0,1) #categorical feature are allocated the size of the number of categories, others size 1
        #print(ft0,size0)
        n_input+=size0 #the final input size is the sum of the size allocated to each feature


    #============== NN definition and implementation =============
    #now starting RNN
    #uas_threshold=0.8 #to determine if this is a good or a bad parse
    
    n_hidden = 128

    current_loss = 0
    all_losses = []
    LR = 0.02           # learning rate
    #LR = 0.005           # learning rate
    #LR = 0.05           # learning rate


    rnn = RNN(n_input, n_hidden, 1)
    rnn.to(device)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
    
    #loss_func = nn.NLLLoss()
    #criterion = nn.NLLLoss()
    #learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn



    

    #print(list(ft_dict.keys())[0])
    prev_time=time.time()
    for e0 in range(n_epochs):
        for t_, ti in enumerate(train_items):
            if t_%500==0: 
                pass
                # elapsed=time.time()-prev_time
                # prev_time=time.time()
                # print("training, epoch",e0, "item", t_, "elapsed", elapsed)
            if t_>train_size: break
            #hidden = rnn.initHidden()
            #hidden = None
            rnn.hidden = rnn.init_hidden()
            #print("HIDDEN >>>", t_, hidden.size())
            rnn.zero_grad()

            #
            cur_ft_dict,sf_id, sent_size, parser_name, parser_uas=ti


            # cur_ft_dict={}
            # cur_ft_dict["uas"]=[parser_uas]* sent_size
            # print("out pred", cur_ft_dict)

            #if sent_size<2: continue
            #line_tensor=make_one_hot(cur_ft_dict,ft_size_dict,index_dict)
            line_tensor=make_one_hot(cur_ft_dict,ft_size_dict,index_dict)#.cuda(cuda0)
            #line_tensor.to(device)
            
            #print("line_tensor", line_tensor)
            
            #print(input_tensor.size())
            category_tensor=torch.tensor([parser_uas]).view([1,1,1])
            #category_tensor.to(device)
            #print(category_tensor)

            # good_parse=0
            # if parser_uas>=uas_threshold: good_parse=1
            # category_tensor=torch.tensor([good_parse])
            #print(line_tensor.size())
            #line_tensor=line_tensor.view([(1, 1, 3)])

            


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
        elapsed=time.time()-prev_time
        prev_time=time.time()
        #print("training, epoch",e0, "item", t_, "elapsed", elapsed)

        print("%s\t%s\t%s"%(e0,current_loss.item(),elapsed))
        current_loss = 0


    pred_items=[]
    print("Now testing", len(dev_items))
    for d_, di in enumerate(dev_items):
        if d_%100==0: print("testing, item", d_)
        rnn.hidden = rnn.init_hidden()
        rnn.zero_grad()
        #print(d_)

        #if d_>100: break
        cur_ft_dict,sf_id, sent_size, parser_name, parser_uas=di

        # cur_ft_dict={}
        # cur_ft_dict["uas"]=[parser_uas]* sent_size
        #print("out pred", cur_ft_dict)


        line_tensor=make_one_hot(cur_ft_dict,ft_size_dict,index_dict)
        #print(line_tensor)
        category_tensor=torch.tensor([parser_uas]).view([1,1,1])


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




