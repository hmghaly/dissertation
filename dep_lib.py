import os, re, random
from itertools import groupby
import itertools

random.seed(0)


def consctv(list1): #to split a group of items into subgroups of contiguous/consecutive items
    grouped=[[v[1] for v in group] for key,group in groupby(enumerate(list1),lambda x:x[1]-x[0])]
    return grouped


def printable_conll(conll_chunk):
	conll_2d=[v.split("\t") for v in conll_chunk.split("\n") if v]
	new_conll_str=""
	for row in conll_2d:
		items=[] 
		for cell in row:
			#if "|" in cell or cell=="_": items.append("-") #get rid of the pipes
			if "|" in cell: items.append("_") #get rid of the pipes, for visualization
			else: items.append(cell)
		new_conll_str+="\t".join(items)+"\n"
	return new_conll_str

def get_conll_heads(conll_chunk):
	conll_2d=[v.split("\t") for v in conll_chunk.split("\n") if v]
	words_heads=[(v[1],v[6]) for v in conll_2d]
	return words_heads



def recursive_dep(node,dep_dict,child_list=[],depth=0):
	if node!=0: child_list.append(node)
	child_list.extend(dep_dict.get(node,[]))
	for cc in dep_dict.get(node,[]):
		#print "node", node, "child node", cc, "depth", depth
		child_list=recursive_dep(cc,dep_dict,child_list,depth+1)
	return child_list

def conll2ptb(input_conll_str,bracket="(",input_pos_index=3,input_dep_index=6):
	conll_obj=conll_new(input_conll_str,pos_index=input_pos_index,dep_index=input_dep_index)
	cur_ptb=conll_obj.get_ptb()
	if bracket=="[": cur_ptb=cur_ptb.replace("(","[").replace(")","]")
	return cur_ptb

def get_recursive_dep(head_index,cur_dep_dict,level=0):
	parent_head=cur_dep_dict.get(head_index)
	if parent_head!=None: level=get_recursive_dep(parent_head,cur_dep_dict,level+1)
	return level

def get_recursive_ancestors(head_index,cur_dep_dict,ancestors=[],level=0):
	parent_head=cur_dep_dict.get(head_index)
	if level>10:
		ancestors.append("MAX_REACHED")
		return ancestors
	if parent_head!=None: 
		ancestors.append(parent_head)
		ancestors=get_recursive_ancestors(parent_head,cur_dep_dict,ancestors, level+1)
	return ancestors

def get_recursive_all_heads(cur_dep_dict,level=0): #get recursively all the heads and ancestors to each node
	cur_list=[]
	for ind,head in cur_dep_dict.items():
		ancestors=get_recursive_ancestors(head,cur_dep_dict,ancestors=[],level=0)
		all_cur_heads=[head]+ancestors
		cur_list.append((ind,all_cur_heads))
		#print(ind,head,ancestors)
	return cur_list





def get_dep_levels(conll_str,pos_index=3,dep_index=6,drel_index=7): #get depth of each word in the dependency structure
	conll_2d=[v.split("\t") for v in conll_str.split("\n") if v] #splitting the conll string into 2-d array
	dep_dict={}
	word_dict={}
	all_heads=[]
	all_indexes=[]
	for c2 in conll_2d:
		cur_index,depends_on,cur_dep_rel=int(c2[0]),int(c2[dep_index]),c2[drel_index]
		word,pos=c2[1],c2[pos_index]
		dep_dict[cur_index]=depends_on
		word_dict[cur_index]=word
		all_heads.append(depends_on)
		all_indexes.append(cur_index)
	all_heads=list(set(all_heads))
	head_level_dict={}
	max_level=0
	for ah in all_heads:
		cur_level=get_recursive_dep(ah,dep_dict,level=0)
		#print ah, cur_level 
		head_level_dict[ah]=cur_level
		if cur_level>max_level: max_level=cur_level

	index_level_dict={}
	for cur_index in all_indexes:
		cur_word=word_dict[cur_index]
		cur_head=dep_dict[cur_index]
		cur_head_level=head_level_dict[cur_head]
		#cur_head_level_norm=2.0*cur_head_level/max_level

		#print "%s\t%s"%(cur_word,cur_head_level_norm)
		#print cur_index, cur_word, cur_head,  cur_head_level
		index_level_dict[(cur_index,cur_head_level)]=cur_word
	for i in range(max_level+1):
		row_items=[]
		for ind in all_indexes:
			cur_item=index_level_dict.get((ind,i),"")
			row_items.append(cur_item)
		#print "\t".join(row_items)




def analyze_dep_link_configs(conll_chunk): #get the dependency configuration betweeen each two consecutive words, as well as their link configuration
	conll_2d=[v.split("\t") for v in conll_chunk.split("\n") if v]
	new_2d=[(int(v[0]),v[1].strip("-"),int(v[6]),v[7]) for v in conll_2d]
	all_deps=[(v[0],v[2]) for v in new_2d if v[2]!=0]
	dep_configs=[]
	#print all_deps
	for di in range(len(new_2d)-1):
		cur_item=new_2d[di]
		next_item=new_2d[di+1]
		cur_index,cur_word,cur_head,cur_rel=cur_item
		next_index,next_word,next_head,next_rel=next_item
		cur_tobi="1"
		cur_word_split=cur_word.split("-")
		if len(cur_word_split)>1 and cur_word_split[1][0].isdigit(): cur_tobi=cur_word_split[1]
		cur_head_offset=0
		next_head_offset=0
		if cur_head!=0: cur_head_offset=cur_head-cur_index
		if next_head!=0: next_head_offset=next_head-next_index

		net_offset=next_head_offset-cur_head_offset

		cur_head_offset_norm=cur_head_offset
		if cur_head_offset<=-1: cur_head_offset_norm=-1
		if cur_head_offset>1: cur_head_offset_norm=2
		next_head_offset_norm=next_head_offset
		if next_head_offset<-1: next_head_offset_norm=-2
		if next_head_offset>=1: next_head_offset_norm=1



		rightward_links=[v for v in all_deps if v[1]<v[0] and v[0]>next_index and v[1]==cur_index]
		leftward_links=[v for v in all_deps if v[1]>v[0] and v[0]<cur_index and v[1]==next_index]
		crossing_links=[v for v in all_deps if (v[0]<cur_index and v[1]>next_index) or (v[0]>next_index and v[1]<cur_index)]
		#print cur_item, next_item, "ToBI", cur_tobi, "cur_head_offset", cur_head_offset_norm, "next_head_offset", next_head_offset_norm
		dep_configs.append((cur_tobi,(cur_head_offset_norm,next_head_offset_norm),net_offset,len(rightward_links),len(leftward_links),len(crossing_links)))
		# print cur_item, next_item
		# print "cur_tobi", cur_tobi, "cur_head_offset", cur_head_offset, "next_head_offset", next_head_offset
		# print "rightward_links", rightward_links, "leftward_links", leftward_links, "crossing_links", crossing_links
		# print "------"
	return dep_configs

def check_crossing(pairs):
	ordered_deps=[(min(v),max(v)) for v in pairs if v[1]!=0]
	for in0,in1 in pairs:
		if in1==0: continue
		x0,x1=min(in0,in1),max(in0,in1)
		if ordered_deps.count((x0,x1))>1: return True 
		check1=[(y0,y1) for y0,y1 in ordered_deps if y0>x0 and y0<x1 and y1>x1]
		check2=[(y0,y1) for y0,y1 in ordered_deps if y1>x0 and y1<x1 and y0<x0]
		if check1 or check2: return True #yes there is crossing
	return False

def check_cyclic(pairs):
	out=False
	cur_dep_dict=dict(iter(pairs))
	cur_list=get_recursive_all_heads(cur_dep_dict)
	for ind,heads in cur_list:
		#print("??????>>>",ind,heads)
		if ind in heads: out=True #there is cyclic dependence
		if "MAX_REACHED" in heads: out=True
	return out


class conll: #we will use this one, so we can convert to PTB at some point
	def __init__(self,conll_str,pos_index=3,dep_index=6,drel_index=7,full_analysis=True): #full analysis to include syntactic constituents, phonological/prosodic chunks, and also configurations
		self.dep_index=dep_index
		self.drel_index=drel_index
		self.dep_dict={} #depedency dict; which indexes depend on a given head
		self.index_head_dict={} #what is the head of each given index
		self.word_dict={} #this index corresponds to this word
		self.pos_dict={} #this index corresponds to this POS
		self.deprel_dict={} #the dependency relation of this index
		self.head_offset_dict={} #the distance between each word and its head
		self.index_list=[] #the list of indexes and their corresponding words
		self.full_list=[] #the list of indexes and their corresponding words, POS and dependencies
		self.conll_2d=[v.split("\t") for v in conll_str.split("\n") if v] #splitting the conll string into 2-d array
		self.dep_index_pairs=[]
		self.size=len(self.conll_2d)
		self.all_heads=[]
		self.all_deps=[] #index-head_index pairs
		self.all_words=[]
		self.all_indexes=[]
		self.all_head_offsets=[]
		self.basic_dep_dict={}
		
		

		for c2 in self.conll_2d:
			cur_index,depends_on,cur_dep_rel=int(c2[0]),int(c2[dep_index]),c2[drel_index]
			word,pos=c2[1],c2[pos_index]
			self.index_list.append((cur_index,word))
			self.full_list.append((cur_index,word,pos,depends_on)) #depends on should be head_index
			self.word_dict[cur_index]=word
			self.pos_dict[cur_index]=pos
			self.deprel_dict[cur_index]=cur_dep_rel
			self.dep_dict[depends_on]=self.dep_dict.get(depends_on,[])+[cur_index]
			self.dep_index_pairs.append((cur_index,depends_on)) #this is the one we need: pairs of indexes, heads
			if depends_on==0: head_offset=0
			else: head_offset=cur_index-depends_on
			self.head_offset_dict[pos_index]=head_offset
			self.all_heads.append(depends_on)
			if depends_on!=0: self.all_deps.append((cur_index,depends_on)) #we will need to adjust the naming so all_deps doesn't exclude roots
			self.index_head_dict[cur_index]=depends_on
			self.all_words.append(word)
			self.all_indexes.append(cur_index)
			self.all_head_offsets.append(head_offset)
			self.basic_dep_dict[cur_index]=depends_on
			


		self.word_dep_pairs=[(self.word_dict.get(v[0],""),self.word_dict.get(v[1],"")) for v in self.dep_index_pairs]
		self.word_deprel_pairs=[(self.word_dict.get(v[0],""),self.deprel_dict.get(v[0],"")) for v in self.dep_index_pairs]
		self.word_deprel_word_triplets=[(self.word_dict.get(v[0],""),self.deprel_dict.get(v[0],""),self.word_dict.get(v[1],"")) for v in self.dep_index_pairs]
		self.word_pos_pairs=[(self.word_dict.get(v[0],""),self.pos_dict.get(v[0],"")) for v in self.dep_index_pairs]

		#advanced processing of conll to get further info from the structure

		#syntactic phrase geometry information
		self.phrase_list=[]
		self.number_phrases_end_index_list=[0]*self.size #number of phrases ending at a certain index
		self.number_phrases_start_index_list=[0]*self.size #number of phrases starting *immediately after* the current index
		self.size_phrases_end_index_list=[0]*self.size #size of the largest phrase ending at each index
		self.size_phrases_start_index_list=[0]*self.size #size of the largest phrase starting *after* the current index

		#depth of each word within the structure (key: word index, value, depth)
		self.depth_dict={}
		self.list_depths=[]

		#phonological/prosodic dependency chunks, obtained by grouping each head with its adjacent non-head children 
		self.dep_chunks=[] #the chunks of words themselves
		self.chunk_boundaries=[0]*self.size #list of indexes, indicating whether each index correspond to chunk boundary or not

		#dependency configuration
		self.list_dep_configs_raw=[] #list of dependency configurations between each two consecutive words, with the original head offsets -6,+2
		self.list_dep_configs=[] #same list but normalized -1,+1
		self.list_link_configs=[] #number of leftward links, rightward links, and overhead links between each two consecutive words


		if not full_analysis: return 

		
		for dd in self.dep_dict:
			all_children=recursive_dep(dd,self.dep_dict,[])
			all_children=sorted(list(set(all_children)))
			start_i,end_i=all_children[0]-1,all_children[-1]-1
			full_phrase=" ".join([self.word_dict[v] for v in all_children]) 
			#print all_children, full_phrase, start_i,end_i

			#print dd, self.word_dict.get(dd,""), self.pos_dict.get(dd,"root"), self.dep_dict[dd], full_phrase
			self.phrase_list.append((dd, self.word_dict.get(dd,""), self.pos_dict.get(dd,"root"), all_children, full_phrase))

		self.phrase_list.sort(key=lambda x:(x[-2][-1],-x[-2][0]))
		#print "phrases"
		endings=[]
		for fl in self.phrase_list:
			#print fl
			endings.append(fl[3][-1])
		#print "brackets"#, endings
		self.num_brackets=[]
		for ind,wd in self.index_list:
		        num_brackets=endings.count(ind)
		        #print wd, num_brackets
		        self.num_brackets.append((ind,wd,num_brackets))
		#print ">>>", [v[-1] for v in self.num_brackets]
		#print ">>>", self.num_brackets
		self.number_phrases_end_index_list=[v[-1] for v in self.num_brackets]

		all_phon_phrases=[]
		for head in self.dep_dict:
			if int(head)==0:  continue
			dependents=self.dep_dict[head]
			#print(head,dependents)
			cur_phrase=[head]
			for d in dependents:
				if not self.dep_dict.get(d): cur_phrase.append(d)
			cur_phrase.sort()
			non_consecutive_phrases=consctv(cur_phrase)
			all_phon_phrases.extend(non_consecutive_phrases)
		all_phon_phrases.sort()
		for ap in all_phon_phrases:
			#print ap, [index_word_dict[v] for v in ap]
			#print ap, " ".join([self.word_dict[v] for v in ap])
			phrase_words=" ".join([self.word_dict[v] for v in ap])
			self.dep_chunks.append(phrase_words)
			last_index=ap[-1]-1
			self.chunk_boundaries[last_index]=1
			#print ap, phrase_words
		#print self.chunk_boundaries

		#now getting the depth =========================
		temp_heads=list(set(self.all_heads))
		depth_dict={}
		#print self.dep_dict
		max_level=0
		for ah in temp_heads:
			cur_level=get_recursive_dep(ah,self.index_head_dict,level=0)
			#print ah, cur_level 
			depth_dict[ah]=cur_level
			if cur_level>max_level: max_level=cur_level
		#print depth_dict
		index_level_dict={}
		for cur_index in self.all_indexes:
			cur_word=self.word_dict[cur_index]
			cur_head=self.index_head_dict[cur_index]
			cur_head_level=depth_dict[cur_head]
			self.list_depths.append(cur_head_level)
			#cur_head_level_norm=2.0*cur_head_level/max_level

			#print "%s\t%s"%(cur_word,cur_head_level_norm)
			#print cur_index, cur_word, cur_head,  cur_head_level
			index_level_dict[(cur_index,cur_head_level)]=cur_word
		for i in range(max_level+1):
			row_items=[]
			for ind in self.all_indexes:
				cur_item=index_level_dict.get((ind,i),"")
				row_items.append(cur_item)
			#print "\t".join(row_items)
		tmp_depths=self.list_depths#+[-1]
		self.depth_diff=[z1-z0 for z0,z1 in zip(tmp_depths,tmp_depths[1:])] #pairs of depths of consecutive words
		self.depth_diff.append(-100)

		new_2d=self.conll_2d
		for di in range(len(new_2d)-1):
			cur_index,next_index=di+1,di+2
			cur_head=self.all_heads[di]
			next_head=self.all_heads[di+1]
			cur_head_offset_raw=0
			next_head_offset_raw=0
			if cur_head!=0: cur_head_offset_raw=cur_head-cur_index
			if next_head!=0: next_head_offset_raw=next_head-next_index

			#net_offset=next_head_offset-cur_head_offset

			cur_head_offset=cur_head_offset_raw
			if cur_head_offset_raw<=-1: cur_head_offset=-1
			if cur_head_offset_raw>1: cur_head_offset=2
			next_head_offset=next_head_offset_raw
			if next_head_offset_raw<-1: next_head_offset=-2
			if next_head_offset_raw>=1: next_head_offset=1
			cur_config=(cur_head_offset,next_head_offset)
			cur_raw_config=(cur_head_offset_raw,next_head_offset_raw)

			self.list_dep_configs_raw.append(cur_raw_config)
			self.list_dep_configs.append(cur_config)



			rightward_links=[v for v in self.all_deps if v[1]<v[0] and v[0]>next_index and v[1]==cur_index]
			leftward_links=[v for v in self.all_deps if v[1]>v[0] and v[0]<cur_index and v[1]==next_index]
			crossing_links=[v for v in self.all_deps if (v[0]<cur_index and v[1]>next_index) or (v[0]>next_index and v[1]<cur_index)]
			self.list_link_configs.append((len(rightward_links),len(leftward_links),len(crossing_links)))
			#print cur_index,next_index, cur_config
		self.list_dep_configs_raw.append(())
		self.list_dep_configs.append(())
		self.list_link_configs.append(())

		self.closing_brackets_list=self.number_phrases_end_index_list
		self.feature_headers=["all_words", "word_dep_pairs", "all_heads","all_head_offsets","list_depths","link_configs","dep_configs", "syntactic_closing_brackets", "chunk_boundaries", "depth_diff"]
		self.features_zipped=zip(self.all_words, self.word_dep_pairs, self.all_heads,self.all_head_offsets,self.list_depths,self.list_link_configs,self.list_dep_configs, self.number_phrases_end_index_list, self.chunk_boundaries, self.depth_diff)
		self.features_matrix=[self.feature_headers]+list(self.features_zipped)
		self.features_str=""
		self.features_str+="\t".join(self.feature_headers)+"\n"
		for z1 in self.features_zipped:
			self.features_str+="\t".join([str(v) for v in z1])+"\n"



	def get_ptb(self):		       
		projection_dict={}
		for cl in self.full_list: #we create projections for each word, depending on POS
			cur_index,word,pos,depends_on=cl
			depends_on_pos=self.pos_dict.get(depends_on,"")
			dependents=self.dep_dict.get(cur_index,[])
			projections=[pos] #the first projection for any word is its POS tag
			if dependents:
				if pos[0].lower()=="n": projections+=["NP"]
				elif pos[0].lower()=="v": 
					prev=[v for v in dependents if v<cur_index]
					if prev: projections+=["VP","S"]
					else: projections=["VP"]
				elif pos[:2].lower()=="in": 
					projections+=["PP"]
				elif pos[:2].lower()=="jj": 
					projections+=["ADJP"]
				elif pos[:2].lower()=="rb": 
					projections+=["ADVP"]
				elif pos[:2].lower()=="cd": 
					projections+=["NP"]
				elif pos[:2].lower()=="uh": 
					projections+=["INTJ"]

				else:
					projections=[pos,pos+"P"] #an ugly hack to differenciate between the first projection (POS tag), and a further projection if there are dependents on it
			else:
				if pos.lower()=="prp": 
					projections+=["NP"]

			projection_dict[cur_index]=projections

		child_parent_list=[]
		root_node=None

		for cl in self.full_list:
			cur_index,word,pos,depends_on=cl
			depends_on_pos=self.pos_dict.get(depends_on,"")
			cur_projections=projection_dict[cur_index]
			cur_projections=[word]+cur_projections
			if depends_on==0: root_node=(cur_projections[-1],cur_index)
			for i in range(len(cur_projections)-1): #get single word/head projections
				child,parent=cur_projections[i],cur_projections[i+1]
				#parent_dict[(child,cur_index)]=(parent,cur_index)
				tmp_child,tmp_parent=(child,cur_index), (parent,cur_index)
				child_parent_list.append((tmp_child,tmp_parent))
			#print cl, depends_on_pos, cur_projections
			cur_node=(cur_projections[-1],cur_index)
			parent_node=None
			if depends_on_pos and depends_on_pos and depends_on_pos[0].lower()=="v" and cur_index>depends_on:
				parent_node=("VP",depends_on)
			elif pos.lower()=="md" and depends_on_pos and depends_on_pos[0].lower()=="v":
				parent_node=("VP",depends_on)

			else: #attach the current maximal projection to the maximal project of the word that the current word depends on
				depends_on_projections=projection_dict.get(depends_on)
				if depends_on_projections: parent_node=(depends_on_projections[-1],depends_on)
			if parent_node: child_parent_list.append((cur_node,parent_node)) #parent_dict[cur_node]=parent_node
			#print "child node", cur_node, "parent node", parent_node
		#for a in child_parent_list:
		#	print ">>>", a

		child_parent_list.sort(key=lambda x:x[1])
		grouped=[(key,[v[0] for v in list(group)]) for key,group in groupby(child_parent_list,lambda x:x[1])]
		self.child_dict={}
		self.root_node=root_node
		for k,grp in grouped: self.child_dict[k]=grp
		self.cur_ptb=gen_ptb(self.child_dict,root_node,"")
		#print self.cur_ptb
		return self.cur_ptb
	def augment(self,seed_uas=0.5,max_n=20):
		all_conll=[]
		all_conll_uas=[]

		#print(self.all_deps)
		ordered_deps=[(min(v),max(v)) for v in self.all_deps]
		#print(ordered_deps)
		uas_vals=[0.1,0.25,0.5,0.75,0.9] #scattered seed values to make sure we have a wide diversity of UAS values
		#uas_vals=[0.1,0.25,0.5]
		for seed_uas in uas_vals:

		#print(self.basic_dep_dict)
		#print(self.all_indexes)
			self.possible_heads_dict={}
			self.possible_heads_list=[]
			gold_list=[]
			for in0 in self.all_indexes:
				actual_head=self.basic_dep_dict[in0]
				gold_list.append((in0,actual_head))
				#print(">>>>", in0,actual_head)
				cur_possibilities=[actual_head]
				if actual_head==0: 
					self.possible_heads_dict[in0]=cur_possibilities
					self.possible_heads_list.append([(in0,actual_head)])
					continue
				for in1 in self.all_indexes:
					if in1==in0: continue
					if in1==actual_head: continue
					#print(">>>> !!!!", in0,in1,actual_head)
					x0,x1=min(in0,in1),max(in0,in1)
					check1=[(y0,y1) for y0,y1 in ordered_deps if y0>x0 and y0<x1 and y1>x1]
					check2=[(y0,y1) for y0,y1 in ordered_deps if y1>x0 and y1<x1 and y0<x0]

					if check1 or check2: continue

					cur_possibilities.append(in1)
				self.possible_heads_dict[in0]=cur_possibilities
				cur_items=[(in0,v) for v in cur_possibilities]
				self.possible_heads_list.append(cur_items)

			final_list=[]
			
			for _ in range(200):
				final_list=[]
				rel_list=[]
				if len(all_conll_uas)>max_n: break
				for a in self.possible_heads_list:
					#print(a)
					a1=a[1:]
					if not a1: 
						final_list.append(a[0])
						rel_list.append("")
					else:
						rand_val=random.random()
						if rand_val>seed_uas:
							random.shuffle(a1)
							final_list.append(a1[0])
							rel_list.append("dep")
						else:
							final_list.append(a[0])
							rel_list.append("")

				correct=list(set(gold_list).intersection(final_list))
				new_uas=float(len(correct))/len(gold_list)
				new_conll_2d=[]
				new_gen_conll_str=""
				for ci,c2 in enumerate(self.conll_2d):
					ind,head=final_list[ci]
					new_row=list(c2)
					cur_rel=rel_list[ci]
					if cur_rel!="": new_row[self.drel_index]=cur_rel
					new_row[self.dep_index]=head
					new_conll_2d.append(new_row)
					new_gen_conll_str+="\t".join([str(v) for v in new_row])+"\n"
				info_row=["_"]*len(new_row)
				info_row[0]=len(self.conll_2d)+1
				info_row[self.dep_index]=0
				info_row[self.drel_index]="-"
				info_row[3]="UAS"
				info_row[1]=round(new_uas,2)
				new_conll_2d.append(info_row)
				#new_gen_conll_str+="\t".join([str(v) for v in info_row])+"\n"
				if new_gen_conll_str in all_conll: continue
				all_conll.append(new_gen_conll_str)
				
				
				#conll_str="\n".join(["\t".join()])
				crossing_exists=check_crossing(final_list)
				cyclic_exists=check_cyclic(final_list)
				#print(crossing_exists,cyclic_exists)

				#print("--------")
				if not cyclic_exists and not crossing_exists:
					all_conll_uas.append((new_gen_conll_str,new_uas))
					#print(new_gen_conll_str)
					#print
		#print(len(all_conll))
		return all_conll_uas

			#if crossing_exists==False: break











			
			#print(a1)
			
			#print(rand_val)
			#print("-----")

		# combinations=list(itertools.product(*self.possible_heads_list))
		# print(len(combinations))
		# for cmb in combinations[:10]:
		# 	print(cmb)
		# print("-----")
		# print(gold_list)
		# print("-----")
		# print(final_list)
		
		# print(check_crossing(final_list))
		# print(len(correct),)
		# print("=====")





def merge_conll_acoustics(conll_st,acoustics_list): #to add the acoustics info into the conll
	pass

def visualize_dep(conll_str):
	pass

# def dep2const(conll_str,dep_index=6,drel_index=7):
# 	dep_dict={}
# 	pos_dict={}
# 	conll_2d=[v.split("\t") for v in conll_str.split("\n") if v]
# 	our_list=[]
# 	for c2 in conll_2d: #extracting info from the conll 2-d array
# 		cur_index,depends_on,cur_dep_rel=int(c2[0]),int(c2[dep_index]),c2[drel_index]
# 		word,pos=c2[1],c2[3]
# 		our_list.append((cur_index,word,pos,depends_on))
# 		dep_dict[depends_on]=dep_dict.get(depends_on,[])+[cur_index]
# 		pos_dict[cur_index]=pos

# 	projection_dict={}
# 	for cl in our_list: #we create projections for each word, depending on POS
# 		cur_index,word,pos,depends_on=cl
# 		depends_on_pos=pos_dict.get(depends_on,"")
# 		dependents=dep_dict.get(cur_index,[])
# 		projections=[pos]
# 		if dependents:
# 			if pos[0].lower()=="n": projections+=["NP"]
# 			elif pos[0].lower()=="v": 
# 				prev=[v for v in dependents if v<cur_index]
# 				if prev: projections+=["VP","S"]
# 				else: projections=["VP"]
# 			elif pos[:2].lower()=="in" or pos.lower()=="p": 
# 				projections+=["PP"]
# 			elif pos[:2].lower()=="jj": 
# 				projections+=["ADJP"]
# 			elif pos[:2].lower()=="rb": 
# 				projections+=["ADVP"]
# 			elif pos[:2].lower()=="cd": 
# 				projections+=["NP"]

# 			else:
# 				projections=[pos,pos+"P"] #an ugly hack to differenciate between the first projection (POS tag), and a further projection if there are dependents on it
# 		else:
# 			if pos.lower()=="prp": 
# 				projections+=["NP"]

# 		projection_dict[cur_index]=projections

# 	child_parent_list=[]
# 	root_node=None

# 	for cl in our_list:
# 		cur_index,word,pos,depends_on=cl
# 		depends_on_pos=pos_dict.get(depends_on,"")
# 		cur_projections=projection_dict[cur_index]
# 		cur_projections=[word]+cur_projections
# 		if depends_on==0: root_node=(cur_projections[-1],cur_index)
# 		for i in range(len(cur_projections)-1): #get single word/head projections
# 			child,parent=cur_projections[i],cur_projections[i+1]
# 			#parent_dict[(child,cur_index)]=(parent,cur_index)
# 			tmp_child,tmp_parent=(child,cur_index), (parent,cur_index)
# 			child_parent_list.append((tmp_child,tmp_parent))
# 		#print cl, depends_on_pos, cur_projections
# 		cur_node=(cur_projections[-1],cur_index)
# 		parent_node=None
# 		if depends_on_pos and depends_on_pos[0].lower()=="v" and cur_index>depends_on:
# 			parent_node=("VP",depends_on)
# 		elif pos.lower()=="md" and depends_on_pos[0].lower()=="v":
# 			parent_node=("VP",depends_on)

# 		else: #attach the current maximal projection to the maximal project of the word that the current word depends on
# 			depends_on_projections=projection_dict.get(depends_on)
# 			if depends_on_projections: parent_node=(depends_on_projections[-1],depends_on)
# 		if parent_node: child_parent_list.append((cur_node,parent_node)) #parent_dict[cur_node]=parent_node
# 		#print "child node", cur_node, "parent node", parent_node

# 	child_parent_list.sort(key=lambda x:x[1])
# 	grouped=[(key,[v[0] for v in list(group)]) for key,group in groupby(child_parent_list,lambda x:x[1])]
# 	child_dict={}
# 	for k,grp in grouped:
# 		#print k, grp
# 		child_dict[k]=grp
# 	#print root_node
	
# 	cur_prb=gen_ptb(child_dict,root_node,"")
# 	print cur_prb
# 	return cur_prb



def gen_ptb(input_child_dict,node,ptb_str,depth=0):	 #nodes are in the form of ("S",5)
	#op_br,cl_br="[","]"
	op_br,cl_br="(",")"
	cur_children=input_child_dict.get(node,[])
	if not cur_children: 
		ptb_str+=node[0]
		return ptb_str
	ptb_str+=op_br+node[0]+" "
	for cc in cur_children:
		ptb_str=gen_ptb(input_child_dict,cc,ptb_str,depth+1)
	ptb_str+=cl_br
	return ptb_str


def get_phon_phrases(conll_chunk,dep_index=6,drel_index=7):
	dep_dict={}
	index_word_dict={}
	word_dict={}
	pos_dict={}
	conll_2d=[v.split("\t") for v in conll_chunk.split("\n") if v]
	for c2 in conll_2d:
		cur_index,depends_on,cur_dep_rel=int(c2[0]),int(c2[dep_index]),c2[drel_index]
		word,pos=c2[1],c2[3]
		dep_dict[depends_on]=dep_dict.get(depends_on,[])+[cur_index]
		index_word_dict[cur_index]=word
		word_dict[cur_index]=word
		pos_dict[cur_index]=pos

	all_phon_phrases=[]
	for head in dep_dict:
		if int(head)==0:  continue
		dependents=dep_dict[head]
		#print(head,dependents)
		cur_phrase=[head]
		for d in dependents:
			if not dep_dict.get(d): cur_phrase.append(d)
		cur_phrase.sort()
		non_consecutive_phrases=consctv(cur_phrase)
		all_phon_phrases.extend(non_consecutive_phrases)
		#print cur_phrase, [index_word_dict[v] for v in cur_phrase]
		#print "-------"
	all_phon_phrases.sort()
	#print "--- phonological phrases ----"
	# for ap in all_phon_phrases:
	# 	#print ap, [index_word_dict[v] for v in ap]
	# 	print " ".join([index_word_dict[v] for v in ap])
	# print "--- syntactic phrases ----"
	phrase_list=[]
	all_syn_phrases=[]
	syn_boundaries={}
	for dd in dep_dict:
		all_children=recursive_dep(dd,dep_dict,[])
		all_children=sorted(list(set(all_children)))
		full_phrase=" ".join([word_dict[v] for v in all_children]) 
		if all_children in all_syn_phrases: continue
		all_syn_phrases.append(all_children)

		#print dd, word_dict.get(dd,""), pos_dict.get(dd,"root"), dep_dict[dd], full_phrase
		phrase_list.append((dd, word_dict.get(dd,""), pos_dict.get(dd,"root"), all_children, full_phrase))
		#print full_phrase, all_children
		#print full_phrase
		last_child=all_children[-1]
		syn_boundaries[last_child]=syn_boundaries.get(last_child,0)+1
	# print "--- syntactic boundaries/closing brackets ----"
	# print syn_boundaries










class conll_old: #class for extracting features from CoNLL string
	def __init__(self,conll_str,dep_index=6,drel_index=7):
		self.dep_dict={}
		self.word_dict={}
		self.pos_dict={}
		self.deprel_dict={}
		self.index_list=[]
		self.conll_2d=[v.split("\t") for v in conll_str.split("\n") if v]
		self.dep_index_pairs=[]

		for c2 in self.conll_2d:
			cur_index,depends_on,cur_dep_rel=int(c2[0]),int(c2[dep_index]),c2[drel_index]
			word,pos=c2[1],c2[3]
			self.index_list.append((cur_index,word))
			self.word_dict[cur_index]=word
			self.pos_dict[cur_index]=pos
			self.deprel_dict[cur_index]=cur_dep_rel
			self.dep_dict[depends_on]=self.dep_dict.get(depends_on,[])+[cur_index]
			self.dep_index_pairs.append((cur_index,depends_on))

		self.word_dep_pairs=[(self.word_dict.get(v[0],""),self.word_dict.get(v[1],"")) for v in self.dep_index_pairs]
		self.word_deprel_pairs=[(self.word_dict.get(v[0],""),self.deprel_dict.get(v[0],"")) for v in self.dep_index_pairs]
		self.word_deprel_word_triplets=[(self.word_dict.get(v[0],""),self.deprel_dict.get(v[0],""),self.word_dict.get(v[1],"")) for v in self.dep_index_pairs]
		self.word_pos_pairs=[(self.word_dict.get(v[0],""),self.pos_dict.get(v[0],"")) for v in self.dep_index_pairs]

		self.phrase_list=[]
		for dd in self.dep_dict:
			all_children=recursive_dep(dd,self.dep_dict,[])
			all_children=sorted(list(set(all_children)))
			full_phrase=" ".join([self.word_dict[v] for v in all_children]) 

			#print dd, word_dict.get(dd,""), pos_dict.get(dd,"root"), dep_dict[dd], full_phrase
			self.phrase_list.append((dd, self.word_dict.get(dd,""), self.pos_dict.get(dd,"root"), all_children, full_phrase))

		self.phrase_list.sort(key=lambda x:(x[-2][-1],-x[-2][0]))
		#print "phrases"
		endings=[]
		for fl in self.phrase_list:
			#print fl
			endings.append(fl[3][-1])
		#print "brackets"#, endings
		self.num_brackets=[]
		for ind,wd in self.index_list:
		        num_brackets=endings.count(ind)
		        #print wd, num_brackets
		        self.num_brackets.append((ind,wd,num_brackets))


cur_conll1="""
1	and	-	CC	CC	B2|C|0|-|-	3	cc	_	_
2	you	-	PRP	PRP	B2|-|0|-|-	3	nsubj	_	_
3	need	-	VBP	VBP	B2|-|0|-|-	0	root	_	_
4	to	-	TO	TO	B2|-|0|-|-	5	aux	_	_
5	make	-	VB	VB	B2|-|0|-|-	3	xcomp	_	_
6	sure	-	JJ	JJ	B2|-|0|-|-	5	acomp	_	_
7	all	-	DT	DT	B2|-|0|-|-	9	predet	_	_
8	your	-	PRP$	PRP$	B2|-|0|-|-	9	poss	_	_
9	bills	-	NNS	NNS	B2|-|0|-|-	10	nsubj	_	_
10	are	-	VBP	VBP	B2|-|0|-|-	6	ccomp	_	_
11	paid	-	JJ	JJ	B2|-|0|-|-	10	acomp	_	_
"""
"1-and-3 2-you-3 3-need-0 4-to-5 5-make-3 6-sure-5 7-all-9 8-your-9 9-bills-10 10-are-6 11-paid-10"
cur_conll2="""
1	and	-	CC	CC	A3|C|0|-|-	9	cc	_	_
2	uh	-	UH	UH	A3|F|0|-|-	9	discourse	_	_
3	our	-	PRP$	PRP$	A3|-|0|-|-	5	poss	_	_
4	family	-	NN	NN	A3|-|0|-|-	5	nn	_	_
5	income	-	NN	NN	A3|-|0|-|-	9	nsubj	_	_
6	at	-	IN	IN	A3|-|0|-|-	5	prep	_	_
7	this	-	DT	DT	A3|-|0|-|-	8	det	_	_
8	point	-	NN	NN	A3|-|0|-|-	6	pobj	_	_
9	is	-	VBZ	VBZ	A3|-|0|-|-	0	root	_	_
10	comfortable	-	JJ	JJ	A3|-|0|-|-	13	amod	_	_
11	upper	-	JJ	JJ	A3|-|0|-|-	13	amod	_	_
12	middle	-	NN	NN	A3|-|0|-|-	13	nn	_	_
13	class	-	NN	NN	A3|-|0|-|-	9	attr	_	_
14	i	-	PRP	PRP	A3|-|0|-|-	15	nsubj	_	_
15	guess	-	VBP	VBP	A3|-|0|-|-	9	parataxis	_	_
16	you	-	PRP	PRP	A3|-|0|-|-	18	nsubj	_	_
17	might	-	MD	MD	A3|-|0|-|-	18	aux	_	_
18	say	-	VB	VB	A3|-|0|-|-	9	ccomp	_	_
"""

cur_conll3="""
1	uh	-	UH	UH	A11|F|0|-|-	4	discourse	_	_
2	housing	-	NN	NN	A11|-|0|-|-	3	nn	_	_
3	prices	-	NNS	NNS	A11|-|0|-|-	4	nsubj	_	_
4	are	-	VBP	VBP	A11|-|0|-|-	0	root	_	_
5	you	-	PRP	PRP	A11|D|0|-|-	6	nsubj	_	_
6	know	-	VBP	VBP	A11|D|0|-|-	4	parataxis	_	_
7	like	-	UH	UH	A11|D|0|-|-	4	discourse	_	_
8	from	-	IN	IN	A11|-|0|-|-	4	prep	_	_
9	four	-	CD	CD	A11|-|0|-|-	8	pobj	_	_
10	to	-	TO	TO	A11|-|0|-|-	12	aux	_	_
11	ten	-	CD	CD	A11|-|0|-|-	12	num	_	_
12	times	-	NNS	NNS	A11|-|0|-|-	4	tmod	_	_
13	more	-	RBR	RBR	A11|-|0|-|-	14	advmod	_	_
14	expensive	-	JJ	JJ	A11|-|0|-|-	12	amod	_	_
15	than	-	IN	IN	A11|-|0|-|-	14	prep	_	_
16	uh	-	UH	UH	A11|F|0|-|-	15	discourse	_	_
17	uh	-	UH	UH	A11|F|0|-|-	19	discourse	_	_
18	they	-	PRP	PRP	A11|-|0|-|-	19	nsubj	_	_
19	were	-	VBD	VBD	A11|-|0|-|-	15	dep	_	_
20	where	-	WRB	WRB	A11|-|0|-|-	22	advmod	_	_
21	i	-	PRP	PRP	A11|-|0|-|-	22	nsubj	_	_
22	came	-	VBD	VBD	A11|-|0|-|-	19	advcl	_	_
23	from	-	IN	IN	A11|-|0|-|-	22	dep	_	_
24	in	-	IN	IN	A11|-|0|-|-	22	prep	_	_
25	uh	-	UH	UH	A11|F|0|-|-	24	discourse	_	_
26	dallas	-	NNP	NNP	A11|-|0|-|-	24	pobj	_	_
"""

cur_conll4="""
1	when	-	WRB	WRB	A23|-|0|-|-	4	advmod	_	_
2	i	-	PRP	PRP	A23|-|0|-|-	4	nsubj	_	_
3	uh	-	UH	UH	A23|F|0|-|-	4	discourse	_	_
4	was	-	VBD	VBD	A23|-|0|-|-	0	root	_	_
5	in	-	IN	IN	A23|-|0|-|-	4	prep	_	_
6	uh	-	UH	UH	A23|F|0|-|-	5	discourse	_	_
7	undergraduate	-	NN	NN	A23|-|0|-|-	8	dep	_	_
8	school	-	NN	NN	A23|-|0|-|-	5	pobj	_	_
9	a	-	DT	DT	A23|-|0|-|-	12	det	_	_
10	long	-	JJ	JJ	A23|-|0|-|-	12	amod	_	_
11	long	-	JJ	JJ	A23|-|0|-|-	12	amod	_	_
12	time	-	NN	NN	A23|-|0|-|-	13	npadvmod	_	_
13	ago	-	RB	RB	A23|-|0|-|-	4	advmod	_	_
14	i	-	PRP	PRP	A23|-|0|-|-	16	dep	_	_
15	uh	-	UH	UH	A23|F|0|-|-	16	discourse	_	_
16	noted	-	VBD	VBD	A23|-|0|-|-	4	ccomp	_	_
17	that	-	IN	IN	A23|-|0|-|-	25	dep	_	_
18	the	-	DT	DT	A23|-|0|-|-	20	det	_	_
19	monthly	-	JJ	JJ	A23|-|1|RM|RM	20	amod	_	_
20	salary	-	NN	NN	A23|-|1|RM|RM	21	npadvmod	_	_
21	starting	-	JJ	JJ	A23|-|0|RR|RR	25	amod	_	_
22	average	-	JJ	JJ	A23|-|0|RR|RR	25	amod	_	_
23	monthly	-	JJ	JJ	A23|-|0|RR|RR	25	amod	_	_
24	salary	-	NN	NN	A23|-|1|RM|RM	25	nn	_	_
25	salary	-	NN	NN	A23|-|0|RR|RR	16	dobj	_	_
26	for	-	IN	IN	A23|-|0|-|-	25	prep	_	_
27	engineers	-	NNS	NNS	A23|-|0|-|-	26	pobj	_	_
28	that	-	WDT	WDT	A23|-|1|RM|RM	30	dobj	_	_
29	you	-	PRP	PRP	A23|D|0|-|-	30	nsubj	_	_
30	know	-	VBP	VBP	A23|D|0|-|-	27	rcmod	_	_
31	in	-	IN	IN	A23|-|0|-|-	30	prep	_	_
32	my	-	PRP$	PRP$	A23|-|0|-|-	33	poss	_	_
33	discipline	-	NN	NN	A23|-|0|-|-	31	pobj	_	_
34	was	-	VBD	VBD	A23|-|0|-|-	4	dep	_	_
35	like	-	IN	IN	A23|-|0|-|-	34	discourse	_	_
36	oh	-	UH	UH	A23|F|0|-|-	40	discourse	_	_
37	six	-	CD	CD	A23|-|0|-|-	39	number	_	_
38	hundred	-	CD	CD	A23|-|0|-|-	39	number	_	_
39	ten	-	CD	CD	A23|-|0|-|-	40	num	_	_
40	dollars	-	NNS	NNS	A23|-|0|-|-	34	attr	_	_
41	a	-	DT	DT	A23|-|0|-|-	42	det	_	_
42	month	-	NN	NN	A23|-|0|-|-	40	npadvmod	_	_
43	or	-	CC	CC	A23|-|0|-|-	40	cc	_	_
44	something	-	NN	NN	A23|-|0|-|-	40	conj	_	_
45	like	-	IN	IN	A23|-|0|-|-	44	prep	_	_
46	that	-	DT	DT	A23|-|0|-|-	45	pobj	_	_

"""

cur_conll5="""
1	I	_	PRP	PRP	_	2	nsubj	_	_
2	saw	_	VB	VBD	_	0	null	_	_
3	the	_	DT	DT	_	4	det	_	_
4	boy	_	NN	NN	_	2	dobj	_	_
5	with	_	IN	IN	_	4	prep	_	_
6	the	_	DT	DT	_	7	det	_	_
7	telescope	_	NN	NN	_	5	pobj	_	_
"""

cur_conll6="""
1	I	_	PRP	PRP	_	2	nsubj	_	_
2	hit	_	VB	VBD	_	0	null	_	_
3	the	_	DT	DT	_	4	det	_	_
4	nail	_	NN	NN	_	2	dobj	_	_
5	with	_	IN	IN	_	2	prep	_	_
6	the	_	DT	DT	_	7	det	_	_
7	hammer	_	NN	NN	_	5	pobj	_	_
"""

cur_conll7="""
1	the	_	DT	DT	_	2	det	_	_
2	boy	_	NN	NN	_	5	nsubj	_	_
3	I	_	PRP	PRP	_	4	nsubj	_	_
4	met	_	VB	VBD	_	2	rcmod	_	_
5	had	_	VB	VBD	_	0	null	_	_
6	a	_	DT	DT	_	7	det	_	_
7	telescope	_	NN	NN	_	5	dobj	_	_
"""

cur_conll10="""
1	who	_	NOUN	WP	_	6	dobj	_	_
2	who	_	NOUN	WP	_	4	nsubjpass	_	_
3	's	_	VERB	VBZ	_	4	auxpass	_	_
4	supposed	_	VERB	VBN	_	0	ROOT	_	_
5	to	_	PART	TO	_	6	aux	_	_
6	make	_	VERB	VB	_	4	xcomp	_	_
7	the	_	DET	DT	_	8	det	_	_
8	change	_	NOUN	NN	_	6	dobj	_	_
"""

cur_conll11="""
1	who	_	PRON	WP	B54|-|1|RM|RM	4	dobj	_	_
2	who	_	PRON	WP	B54|-|0|RR|RR	4	nsubjpass	_	_
3	's	_	VERB	VBZ	B54|-|0|RR|RR	4	auxpass	_	_
4	supposed	_	VERB	VBN	B54|-|0|-|-	0	ROOT	_	_
5	to	_	PRT	TO	B54|-|0|-|-	6	aux	_	_
6	make	_	VERB	VB	B54|-|0|-|-	4	xcomp	_	_
7	the	_	DET	DT	B54|-|0|-|-	8	det	_	_
8	change	_	NOUN	NN	B54|-|0|-|-	6	dobj	_	_
"""

cur_conll12="""
1	who	who	WP	_	_	0	root	_	O
2	who	who	WP	_	_	4	nsubjpass	_	O
3	's	's	VBZ	_	_	4	auxpass	_	O
4	supposed	suppose	VBN	_	_	1	relcl	_	O
5	to	to	TO	_	_	6	aux	_	O
6	make	make	VB	_	_	4	xcomp	_	O
7	the	the	DT	_	_	8	det	_	O
8	change	change	NN	_	_	6	dobj	_	O
"""
cur_conllx="""
1	where	_	ADV	WRB	_	7	advmod	_	_
2	's	_	VERB	VBZ	_	5	auxpass	_	_
3	the	_	DET	DT	_	4	det	_	_
4	money	_	NOUN	NN	_	5	nsubjpass	_	_
5	supposed	_	VERB	VBN	_	0	ROOT	_	_
6	to	_	PART	TO	_	7	aux	_	_
7	come	_	VERB	VB	_	5	xcomp	_	_
8	from	_	ADP	IN	_	7	prep	_	_
"""

test="""
1	i	-	PRP	PRP	_	2	nsubj	_	-
2	think	-	VBP	VBP	_	0	root	_	-
3	there	-	EX	EX	_	8	expl	_	-
4	's	-	BES	BES	_	8	dep	_	-
5	a	-	DT	DT	_	8	dep	_	-
6	a	-	DT	DT	_	8	det	_	-
7	large	-	JJ	JJ	_	8	amod	_	-
8	amount	-	NN	NN	_	2	ccomp	_	-
9	of	-	IN	IN	_	8	prep	_	-
10	corruption	-	NN	NN	_	9	pobj	_	-
11	on	-	IN	IN	_	8	prep	_	-
12	the	-	DT	DT	_	11	dep	_	-
13	the	-	DT	DT	_	14	det	_	-
14	have	-	NN	NN	_	11	pobj	_	-
15	and	-	CC	CC	_	14	cc	_	-
16	the	-	DT	DT	_	17	det	_	-
17	have-nots	-	NNS	NNS	_	14	conj	_	-
18	you	-	PRP	PRP	_	19	nsubj	_	-
19	know	-	VBP	VBP	_	2	parataxis	_	-

"""

test="""
1	i	-	PRP	PRP	_	2	nsubj	_	-
2	think	-	VBP	VBP	_	0	root	_	-
3	there	-	EX	EX	_	8	expl	_	-
4	's	-	BES	BES	_	8	dep	_	-
5	a	-	DT	DT	_	8	dep	_	-
6	a	-	DT	DT	_	8	det	_	-
7	large	-	JJ	JJ	_	8	amod	_	-
8	amount	-	NN	NN	_	2	ccomp	_	-
9	of	-	IN	IN	_	8	prep	_	-
10	corruption	-	NN	NN	_	9	pobj	_	-
11	on	-	IN	IN	_	8	prep	_	-
12	the	-	DT	DT	_	11	dep	_	-
13	the	-	DT	DT	_	14	det	_	-
14	have	-	NN	NN	_	11	pobj	_	-
15	and	-	CC	CC	_	14	cc	_	-
16	the	-	DT	DT	_	17	det	_	-
17	have-nots	-	NNS	NNS	_	14	conj	_	-
18	you	-	PRP	PRP	_	19	nsubj	_	-
19	know	-	VBP	VBP	_	2	parataxis	_	-

"""

test2="""
1	there	-	EX	EX	_	2	expl	_	-
2	was	-	VBD	VBD	_	0	root	_	-
3	some	-	DT	DT	_	4	det	_	-
4	sort	-	NN	NN	_	2	nsubj	_	-
5	of-2p	-	IN	IN	_	4	prep	_	-
6	full-time-2	-	JJ	JJ	_	8	amod	_	-
7	care-3p	-	NN	NN	_	8	nn	_	-
8	place-4	-	NN	NN	_	5	pobj	_	-
9	that	-	WDT	WDT	_	12	nsubjpass	_	-
10	was	-	VBD	VBD	_	12	auxpass	_	-
11	also	-	RB	RB	_	12	advmod	_	-
12	associated-2	-	VBN	VBN	_	8	rcmod	_	-
13	with	-	IN	IN	_	12	prep	_	-
14	it-4	-	PRP	PRP	_	13	pobj	_	-

"""

if __name__ == "__main__":
	#print("dep lib")

	# conll_obj=conll(cur_conll5)

	# phrases=conll_obj.phrase_list
	# brackets=conll_obj.num_brackets
	# index_pairs=conll_obj.dep_index_pairs
	# dep_pairs=conll_obj.word_dep_pairs
	# wd_deprel_pairs=conll_obj.word_deprel_pairs
	# wd_pos_pairs=conll_obj.word_pos_pairs
	# wd_deprel_wd=conll_obj.word_deprel_word_triplets

	our_conll=test
	conll_obj=conll(our_conll)
	augmented=conll_obj.augment()
	for au in augmented:
		print(au[0])
		print(au[1])
		print("-----")
	#print(conll_obj.basic_dep_dict)
	#cur_acestors=get_recursive_heads(12,conll_obj.basic_dep_dict,[])
	#print(cur_acestors)
	# our_list=get_recursive_all_heads(conll_obj.basic_dep_dict,[])
	# for ls in our_list:
	# 	print(ls)


	#features=conll_obj.features_str
	#print features
	# print conll_obj.all_words
	# print conll_obj.all_heads
	# print conll_obj.all_head_offsets
	# print conll_obj.list_depths
	# print conll_obj.list_link_configs
	# print conll_obj.list_dep_configs
	#zipped=zip(conll_obj.all_words, conll_obj.word_dep_pairs, conll_obj.all_heads,conll_obj.all_head_offsets,conll_obj.list_depths,conll_obj.list_link_configs,conll_obj.list_dep_configs, conll_obj.number_phrases_end_index_list, conll_obj.chunk_boundaries, conll_obj.depth_diff)
	#for z1 in zipped:
	#	print "\t".join([str(v) for v in z1])
	# configs=analyze_dep_link_configs(our_conll)
	# conll_2d=conll_obj.conll_2d
	# for i in range(len(configs)):
	# 	c2_items=conll_2d[i]
	# 	new_items=[conll_obj.chunk_boundaries [i],conll_obj.number_phrases_end_index_list[i],configs[i][1]]
	# 	c2_items.extend(new_items)
	# 	print "\t".join([str(v) for v in c2_items])

		#print "%s\t%s\t%s"%(conll_obj.chunk_boundaries [i],conll_obj.number_phrases_end_index_list[i],configs[i][1])
	#print conll_obj.chunk_boundaries 
	#print conll_obj.number_phrases_end_index_list

	#get_phon_phrases(our_conll)
	#print("------")
	#print(printable_conll(our_conll))

	#get_dep_levels(test)


	# for ph in phrases:
	# 	print ">>>>", ph
	# print "------"
	# for br in brackets:
	# 	print br
	# print "------"
	# semantic_coherence_dict={}
	# for ip in index_pairs:
	# 	d0,d1=min(ip), max(ip)
	# 	if d0==0: continue

	# 	#print ip, d0,d1, range(d0,d1)
	# 	for d_ in range(d0,d1):
	# 		semantic_coherence_dict[d_]=semantic_coherence_dict.get(d_,0)+1
	# print "------"
	# #print "now the semantic_coherence_dict"
	# word_dict=conll_obj.word_dict
	# keys=sorted(word_dict.keys())
	#for k in keys:
	#	print k, word_dict.get(k,""), semantic_coherence_dict.get(k,0)

	#dep2const(cur_conll10)
	#dep2const(cur_conll11)
	#dep2const(cur_conll12)

	# conll_obj=conll_new(test,pos_index=3)
	# cur_ptb=conll_obj.get_ptb()
	# print "======"
	# print cur_ptb
	# child_dict=conll_obj.child_dict
	# print "======"
	# print "root_node", conll_obj.root_node
	# for cd in child_dict:
	# 	print cd, child_dict[cd]

	# print "**********"
	# print conll2ptb(cur_conll10,"(")

	

	#cur_ptb=conll_obj.get_ptb()
	#dep2const(cur_conllx)

	# for dp in dep_pairs:
	# 	print dp
	# print "------"
	# for wdp in wd_deprel_pairs:
	# 	print wdp
	# print "------"
	# for w_pos in wd_pos_pairs:
	# 	print w_pos
	# print "------"
	# for w_d_w in wd_deprel_wd:
	# 	print w_d_w
	#==================
	#cur_conll_split=
	# dep_dict={}
	# word_dict={}
	# pos_dict={}
	# index_list=[]
	# conll_2d=[v.split("\t") for v in cur_conll2.split("\n") if v]
	# for c2 in conll_2d:
	# 	cur_index,depends_on=int(c2[0]),int(c2[6])
	# 	word,pos=c2[1],c2[3]
	# 	index_list.append((cur_index,word))
	# 	word_dict[cur_index]=word
	# 	pos_dict[cur_index]=pos
	# 	#print word, cur_index,depends_on
	# 	dep_dict[depends_on]=dep_dict.get(depends_on,[])+[cur_index]
	# #print [conll_2d]

	# final_list=[]
	# for dd in dep_dict:
	# 	all_children=recursive_dep(dd,dep_dict,[])
	# 	all_children=sorted(list(set(all_children)))
	# 	full_phrase=" ".join([word_dict[v] for v in all_children]) 

	# 	#print dd, word_dict.get(dd,""), pos_dict.get(dd,"root"), dep_dict[dd], full_phrase
	# 	final_list.append((dd, word_dict.get(dd,""), pos_dict.get(dd,"root"), all_children, full_phrase))
	# 	#all_children=[]

	# final_list.sort(key=lambda x:(x[-2][-1],-x[-2][0]))
	# print "phrases"
	# endings=[]
	# for fl in final_list:
	# 	print fl
	# 	endings.append(fl[3][-1])
	# print "brackets"#, endings
	# for ind,wd in index_list:
	#         num_brackets=endings.count(ind)
	#         print wd, num_brackets

