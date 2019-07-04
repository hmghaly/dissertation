import os, sys, re, subprocess, time, json, pickle, math, shelve

uas_method=False #in our ensemble, do we predict the UAS, so we will need to train the model on the output of the three parsers and the features, or just on the offsets of the gold standard and the features

if sys.version[0]=="3": 
    import _pickle as cPickle
else: 
    import cPickle

import random

random.seed(0)

from itertools import groupby
from collections import Counter
sys.path.append("../../utils")
from dep_lib import *
from gensim.models import Word2Vec #installed for python2 only

import torch
import torch.nn as nn
from torch.autograd import Variable

import h5py
import numpy as np

import nltk
nltk.download('universal_tagset')

from nltk.tag import pos_tag, map_tag

UNIVERSAL_TAGS = (
    'VERB',
    'NOUN',
    'PRON',
    'ADJ',
    'ADV',
    'ADP',
    'CONJ',
    'DET',
    'NUM',
    'PRT',
    'X',
    '.',
)

def simplify_tag(tag):
    if tag.upper() in UNIVERSAL_TAGS: return tag.upper()
    if 'CONJ' in tag.upper(): return 'CONJ'
    if 'PART' in tag.upper(): return 'PRT'
    return map_tag('en-ptb', 'universal', tag)

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
    if parameters.get("ft-word-word-embeddings",False):
        cur_word_pairs=conll_obj.word_dep_pairs
        cur_ft_dict["words-embeddings"]=[strip_tobi(v[0]) for v in cur_word_pairs]
        cur_ft_dict["heads-embeddings"]=[strip_tobi(v[1]) for v in cur_word_pairs]

    if parameters.get("ft-pos",False):
        #words=[v[0] for v in conll_obj.word_pos_pairs]
        cur_pos_raw=[v[1] for v in conll_obj.word_pos_pairs]
        cur_pos_simplified=[simplify_tag(v) for v in cur_pos_raw]
        cur_ft_dict["pos"]=cur_pos_simplified
        # print("words", words)
        # print("cur_pos_raw", cur_pos_raw)
        # print("cur_pos_simplified", cur_pos_simplified)
        # print("----------")
    if parameters.get("ft-offset",False):
        cur_offsets=conll_obj.all_head_offsets
        cur_ft_dict["offsets"]=cur_offsets

    if parameters.get("ft-depth",False):
        cur_depth_list=conll_obj.list_depths
        cur_ft_dict["depth"]=cur_depth_list
        #print(cur_depth_list)



    if parameters.get("ft-position",False):
    
        #cur_offsets=conll_obj.all_head_offsets
        cur_size=len(conll_obj.all_head_offsets)
        cur_ft_dict["position"]=[float(i+1)/cur_size for i in range(cur_size)]
        #print(cur_ft_dict["position"])

    


    if parameters.get("ft-tobi",False):
        cur_ft_dict["tobi"]=[v[1] for v in input_acoustics]
        #cur_ft_dict["tobi_num"]=tobi_list=[float(v[1][0]) for v in input_acoustics]
        #cur_ft_dict["tobi_p"]=[1 if v[1][-1].lower()=="p" else 0 for v in input_acoustics]
        #cur_ft_dict["tobi_p"]=tobi_list=[v[1][-1].lower()=="p" for v in input_acoustics]
    if parameters.get("ft-tobi-split",False):
        cur_ft_dict["tobi-number"]=[float(v[1][0]) if v[1][0].isdigit() else 1 for v in input_acoustics]
        cur_ft_dict["tobi-r"]=[1 if v[1][-1].lower()=="p" else 0 for v in input_acoustics]

    if parameters.get("ft-dur",False):
        cur_ft_dict["dur"]=[float(v[2]) for v in input_acoustics]
        #print([float(v[2]) for v in input_acoustics])

    if parameters.get("ft-dur-advanced",False):
        cur_ft_dict["dur"]=[float(v[3]) for v in input_acoustics]
        
        
    if parameters.get("ft-dur-log",False):
        dur_vals=[float(v[2]) for v in input_acoustics]
        dur_vals=[v if v>0 else 0.001 for v in dur_vals]

        #cur_ft_dict["dur-log"]=[math.log(float(v[2])) for v in input_acoustics]
        cur_ft_dict["dur-log"]=[math.log(v) for v in dur_vals]
        #print(cur_ft_dict["dur-log"])

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
        cur_ft_dict["pause"]=[float(v[-2]) for v in input_acoustics]

    if parameters.get("ft-pause-binary",False):
        pause_bin_listn_list=[]
        for ia in input_acoustics:
            if float(ia[-1])>0: pause_bin_list.append(1)
            else: pause_bin_list.append(0)
        cur_ft_dict["pause-binary"]=pause_bin_list
        #cur_ft_dict["pause"]=[float(v[-1]) for v in input_acoustics]
                
        
    if parameters.get("ft-configs",False):
        dep_configs=conll_obj.list_dep_configs
        if dep_configs[-1]==(): dep_configs[-1]=(-2,2)
        cur_ft_dict["configs"]=dep_configs
    if parameters.get("ft-links",False):
        dep_links=conll_obj.list_link_configs
        if dep_links[-1]==(): dep_links[-1]=(-1,-1,-1)
        cur_ft_dict["links"]=dep_links
        #ur_ft_dict["links1"]=[v[0] for v in dep_links]
        #cur_ft_dict["links2"]=[v[1] for v in dep_links]
        #cur_ft_dict["links3"]=[v[2] for v in dep_links]
        #
        
        
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
        #tensor[vi][0]=torch.tensor(v1)
        #print(len(v1), v1)
        tensor[vi][0]=torch.as_tensor(v1)
    return tensor



def make_tensor_embedding(input_ft_dict,ft_size_dict,ft_index_dict,embedding_model):
    matrix=[]
    keys=list(sorted(cur_ft_dict.keys(),reverse=True))
    sent_size=len(cur_ft_dict[keys[0]])
    for i in range(sent_size):
        cur_row=[]
        for k in keys:
            cur_val=cur_ft_dict[k][i]
            one_hot_size=ft_size_dict.get(k)
            #print(i,k,cur_val)
            if "embedding" in k:
                cur_word=cur_ft_dict[k][i]
                try: cur_embedding=list(embedding_model[cur_word])
                except: cur_embedding=[0.0]*100
                #print(cur_word, cur_embedding[:10])
                cur_row.extend(cur_embedding)
            elif one_hot_size==None: cur_row.append(cur_val)
            else:
                cur_zeros=[0.0]*one_hot_size
                #(ft_name,val0)
                cur_index=index_dict[(k,cur_val)]
                cur_zeros[cur_index]=1.0
                cur_row.extend(cur_zeros)



        matrix.append(cur_row)
    #tensor1 = torch.zeros(sent_size, 1, len(cur_row))

    #cur_tensor=v2t(matrix)
    cur_tensor=torch.FloatTensor(matrix)
    cur_tensor=cur_tensor.view(sent_size,1,len(cur_row))
    #print(cur_tensor.shape)
    return cur_tensor


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
    #return one_hot_matrix#one_hot_tensor
    return one_hot_tensor


if __name__=="__main__":
    t0=time.time()
    exp_name="exp01_correct_heads"
    if len(sys.argv)>1: 
        exp_name=sys.argv[1]
    print(exp_name)

    exp_dir=os.path.join(os.getcwd(),exp_name)
    if not os.path.exists(exp_dir): os.makedirs(exp_dir)
    train_size=30000
    # Keep track of losses for plottingpause_bin_list=[]
    list_parameters=[]
    exp_parameters={}
    
    exp_parameters["ft-word-word-embeddings"]=True
    exp_parameters["ft-pos"]=True
    exp_parameters["ft-configs"]=True
    exp_parameters["ft-depth"]=True
    
    exp_parameters["ft-word-word"]=False
    exp_parameters["ft-offset"]=True
    exp_parameters["ft-position"]=True
    exp_parameters["ft-links"]=True
    exp_parameters["ft-brackets"]=True
    
    #cur_exp_params=dict(exp_parameters)
    #list_parameters.append(cur_exp_params)

    #prosodic featues
    exp_parameters["ft-tobi"]=True
    exp_parameters["ft-dur"]=True
    exp_parameters["ft-dur-adv"]=False
    exp_parameters["ft-pause"]=True
    exp_parameters["ft-pause-binary"]=False
    exp_parameters["pause-norm"]=False
    exp_parameters["ft-dur-diff"]=True
    exp_parameters["ft-dur-log"]=True
    
    print(exp_parameters)
    gensim_model = Word2Vec.load('gensim-model.bin')


    #list_parameters.append(exp_parameters)
    #list_parameters.reverse()

    category_features=["words","heads","tobi","configs","links","pos"] #we convert their values to indexes, use one-hot
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
            #if si==100: break
            if si%500==0: print(sd_name, "current sentence", si)
            gold_conll=gold_dict.get(sf_id)
            gold_conll_obj=conll(gold_conll)
            
            
            gold_offsets=gold_conll_obj.all_head_offsets

            

            cur_acoustics=acoustics_dict[sf_id]
            cur_acoustics_2d=[v.split("\t") for v in cur_acoustics.split("\n")[1:] if v]
            cur_acoustics_2d_headers=[v.split("\t") for v in cur_acoustics.split("\n") if v]
            
            #if sd_name=="train": continue

            sent_size=len(cur_acoustics_2d)

            parsers_out=parser_output_dict[sf_id]
            #if uas_method==False and sd_name=="train":
            #    parsers_out={}
            #    parsers_out["gold"]=(1,sent_size,gold_conll)
            #print(sd_name, sf_id, len(parsers_out))


            parsers_out_list=[]
            parser_matrixes=[]
            all_uas_vals=[]
            for parser_name in parsers_out:
                items=parsers_out[parser_name]
                parser_uas,sent_size,cur_parser_conll=items
                #print(sd_name, sf_id,parser_name,parser_uas,sent_size)
                parser_uas=float(parser_uas)
                cur_ft_dict=extract_features(cur_parser_conll, cur_acoustics_2d, exp_parameters)
                
                cur_conll_obj=conll(cur_parser_conll)
                offsets=cur_conll_obj.all_head_offsets

                #print(cur_ft_dict)
                for ft_name in cur_ft_dict:
                    vals=cur_ft_dict[ft_name]
                    if ft_name in category_features: 
                        for val0 in vals:
                            val_index, ft_dict = get_ft_index(ft_name,val0,ft_dict)
                            index_dict[(ft_name,val0)]=val_index #easy access to the index of each feature-value pair
                
                correct_heads=[]
                for hi in range(len(offsets)):
                    cur_parser_offset=offsets[hi]
                    cur_gold_offset=gold_offsets[hi]
                    if cur_parser_offset==cur_gold_offset: correct_heads.append(1.0)
                    else: correct_heads.append(0.0)
                
                #print(sf_id, parser_name, "UAS", parser_uas)
                #print("gold offsets", gold_offsets)
                #print("offsets", offsets)
                #print("correct_heads",correct_heads)
                #print("--------")

                cur_item=(cur_ft_dict,sf_id, sent_size, correct_heads, parser_name, parser_uas)
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



    print("converting to tensors")
    hdf5_fpath=os.path.join(exp_dir,"data.hdf5")
    hdf5_file=h5py.File(hdf5_fpath, 'w')
    for it,cur_items in items_dict.items():
        #pickle_fname=it+".pickle"
        #pickle_fpath=os.path.join(exp_dir,pickle_fname)
        print(it, len(cur_items))
        cur_tensor_list=[]
        grp = hdf5_file.create_group(it) #creating a group for train/test/dev
        
        set_shelve_fpath=os.path.join(exp_dir,it+".shelve")
        set_shelve_fopen=shelve.open(set_shelve_fpath)
        for i_,ti in enumerate(cur_items):
                    #if i_>10: break
            if i_%500==0: print(i_)
            #cur_ft_dict,sf_id, sent_size, parser_name, parser_uas=ti
            cur_ft_dict,sf_id, sent_size, correct_heads, parser_name, parser_uas=ti
            new_tensor=make_tensor_embedding(cur_ft_dict,ft_size_dict,index_dict,gensim_model)#.cuda(cuda0)
            one_hot_tensor_numpy=new_tensor.numpy()
            #continue
            #print(cur_ft_dict)
            #one_hot_tensor=make_one_hot(cur_ft_dict,ft_size_dict,index_dict)#.cuda(cuda0)
            #one_hot_tensor=[]
            

            #one_hot_tensor_numpy=one_hot_tensor.numpy()
            one_hot_tensor_torch=torch.tensor(one_hot_tensor_numpy)
            #print(list(one_hot_tensor_numpy.shape) )
            n_input=list(one_hot_tensor_numpy.shape)[-1]
            example_id="%s-%s"%(sf_id,parser_name)
            dset = grp.create_dataset(example_id, data=one_hot_tensor_numpy, compression="gzip")
            #dset = grp.create_dataset(example_id, data=one_hot_tensor_torch, compression="gzip")
            dset.attrs["uas"]=parser_uas
            dset.attrs["sent_size"]=sent_size
            dset.attrs["correct_heads"]=correct_heads

            #compressed_matrix=make_one_hot_compresse

            cur_obj={}            
            cur_obj["uas"]=parser_uas
            cur_obj["sent_size"]=sent_size
            cur_obj["correct_heads"]=correct_heads
            cur_obj["features"]=one_hot_tensor_numpy
            set_shelve_fopen[example_id]=one_hot_tensor_numpy
            # print("---------")
            #line_tensor=make_one_hot(cur_ft_dict,ft_size_dict,index_dict)#.cuda(cuda0)
            #category_tensor=torch.tensor([parser_uas]).view([1,1,1])
            #cur_tensor_list.append((line_tensor,category_tensor))
            #cur_tensor_list.append((one_hot_matrix,parser_uas))
        #     cur_tensor_list.append((compressed_matrix,parser_uas,sf_id, sent_size, parser_name))
        # cpk(cur_tensor_list,pickle_fpath)

    #for item in hdf5_file:
        set_shelve_fopen.close()    #    obj=hdf5_file[item]
    #    for ds in obj:
    #        cur_data=obj[ds]
    #        print(item, ds, cur_data.shape, cur_data.attrs["uas"], cur_data.attrs["offsets"])
    #    #print(item)
    hdf5_file.close()
    elapsed=time.time()-t0
    print("finished pre-processing in %s seconds"%elapsed)


    exp_parameters["n_input"]=n_input #size of the input feature matrix
    parameters_json=json.dumps(exp_parameters)
    parameters_fpath=os.path.join(exp_dir,"parameters.json")
    parameters_file=open(parameters_fpath,"w")
    parameters_file.write(parameters_json)
    parameters_file.close()


