import os, sys, re, subprocess, time, json, pickle

if sys.version[0]=="3": 
    import _pickle as cPickle
else: 
    import cPickle


from itertools import groupby
from collections import Counter
sys.path.append("../../utils")
from dep_lib import *


import torch
import torch.nn as nn
from torch.autograd import Variable


# CPickle
def cpk2(var1,file1):
    fopen=open(file1,'wb')
    cPickle.dump(var1,fopen,protocol=2)
    fopen.close()

def cpk(var1,file1):
    fopen=open(file1,'wb')
    pickle.dump(var1,fopen,protocol=2)
    fopen.close()

# UnPickle
def lpk2(file1):
    fopen=open(file1,'rb')
    output=cPickle.load(fopen)
    fopen.close()
    return output


# UnPickle
def lpk(file1):
    fopen=open(file1,'rb')
    output=pickle.load(fopen)
    fopen.close()
    return output


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
    #one_hot_tensor=v2t(one_hot_matrix)
    return one_hot_matrix#one_hot_tensor

def make_one_hot_compressed(input_ft_dict,ft_size_dict,ft_index_dict):
    one_hot_matrix=[]
    compressed_matrix=[]
    new_matrix=[]
    keys=list(sorted(cur_ft_dict.keys(),reverse=True))
    for i in range(sent_size):
        row_size=0
        cell_i=0
        cur_row=[]
        row_val_index_list=[]
        for k in keys:
            cur_val=cur_ft_dict[k][i]
            #print(i,k,cur_val)
            one_hot_size=ft_size_dict.get(k)
            if one_hot_size==None: 
                cur_row.append(cur_val)
                row_val_index_list.append((cell_i,cur_val))
                cell_i+=1
            else:
                cur_index=index_dict[(k,cur_val)]+cell_i
                row_val_index_list.append((cur_index,1))

                cell_i+=one_hot_size

                cur_zeros=[0]*one_hot_size
                #(ft_name,val0)
                cur_index=index_dict[(k,cur_val)]
                cur_zeros[cur_index]=1
                cur_row.extend(cur_zeros)
        one_hot_matrix.append(cur_row)
        compressed_matrix.append(row_val_index_list)
    return compressed_matrix#one_hot_matrix,compressed_matrix#one_hot_tensor


if __name__=="__main__":
    exp_name="exp0"
    exp_dir=os.path.join(os.getcwd(),exp_name)
    if not os.path.exists(exp_dir): os.makedirs(exp_dir)
    train_size=30000
    # Keep track of losses for plotting
    list_parameters=[]
    exp_parameters={}
    exp_parameters["ft-word-word"]=True
    exp_parameters["ft-configs"]=False
    exp_parameters["ft-links"]=False
    exp_parameters["ft-brackets"]=False
    #cur_exp_params=dict(exp_parameters)
    #list_parameters.append(cur_exp_params)

    #prosodic featues
    exp_parameters["ft-tobi"]=False
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
        elif f_id>=3900 and f_id<=4003: held_set.append(sf_id)#set_name="held" #New portion of the training goes into dev/held           
        else: train_set.append(sf_id)# set_name="train"

    #print(len(train_set),train_set[:10])
    #print(sorted(train_set))
    print(len(train_set),len(dev_set),len(test_set),len(held_set))
    set_dict={}
    set_dict["train"]=train_set
    set_dict["test"]=test_set
    set_dict["dev"]=dev_set+held_set




    sent_size_dict={}
    ft_dict={}
    index_dict={}

    train_items=[]
    dev_items=[]
    test_items=[]
    print("extracting features from sentences")
    

    #print("extracting features from sentences")
    for sd_name,sd_ids in set_dict.items():
        print(sd_name, len(sd_ids))
        #continue
        for si,sf_id in enumerate(sd_ids):
            if si%500==0: print(sd_name, "current sentence", si)
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
                cur_ft_dict=extract_features(cur_parser_conll, cur_acoustics_2d, exp_parameters)

                for ft_name in cur_ft_dict:
                    vals=cur_ft_dict[ft_name]
                    if ft_name in category_features: 
                        for val0 in vals:
                            val_index, ft_dict = get_ft_index(ft_name,val0,ft_dict)
                            index_dict[(ft_name,val0)]=val_index #easy access to the index of each feature-value pair

                cur_item=(cur_ft_dict,sf_id, sent_size, parser_name, parser_uas)
                if sd_name=="train": train_items.append(cur_item)
                if sd_name=="dev": dev_items.append(cur_item)
                if sd_name=="test": test_items.append(cur_item)

    # if si<len(train_set): train_items.append(cur_item)
    # else: dev_items.append(cur_item)

    # if si<train_size: train_items.append(cur_item)
    # else: dev_items.append(cur_item)
    items_dict={}
    items_dict["train"]=train_items
    items_dict["test"]=test_items
    items_dict["dev"]=dev_items




    print(len(train_items),len(dev_items),len(test_items))

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

    exp_parameters["n_input"]=n_input #size of the input feature matrix
    parameters_json=json.dumps(exp_parameters)
    parameters_fpath=os.path.join(exp_dir,"parameters.json")
    parameters_file=open(parameters_fpath,"w")
    parameters_file.write(parameters_json)
    parameters_file.close()


    print("converting to tensors")
    for it,cur_items in items_dict.items():
        pickle_fname=it+".pickle"
        pickle_fpath=os.path.join(exp_dir,pickle_fname)
        print(it, len(cur_items))
        cur_tensor_list=[]
        for i_,ti in enumerate(cur_items):
            if i_%500==0: print(i_)
            cur_ft_dict,sf_id, sent_size, parser_name, parser_uas=ti
            #one_hot_matrix=make_one_hot(cur_ft_dict,ft_size_dict,index_dict)#.cuda(cuda0)
            compressed_matrix=make_one_hot_compressed(cur_ft_dict,ft_size_dict,index_dict)
            #print(compressed_matrix)
            # for cm in compressed_matrix:
            #     print(cm)
            # print("---------")
            #line_tensor=make_one_hot(cur_ft_dict,ft_size_dict,index_dict)#.cuda(cuda0)
            #category_tensor=torch.tensor([parser_uas]).view([1,1,1])
            #cur_tensor_list.append((line_tensor,category_tensor))
            #cur_tensor_list.append((one_hot_matrix,parser_uas))
            cur_tensor_list.append((compressed_matrix,parser_uas,sf_id, sent_size, parser_name))
        cpk(cur_tensor_list,pickle_fpath)




