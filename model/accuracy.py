# author: Avik Ray (avik.r@samsung.com) 
#
#========================================
"""
Utility functions for computing parsing accuracy
"""

import sys

def reverseDict(d):

    reverse = {}
    for key, val in d.items():
        reverse[val] = key
        
    return reverse
    
def post_process(candidate,vocab):
    new_candidate = []
    c_left = 0
    c_right = 0
    for token in candidate:
        new_candidate.append(token)
        if token == vocab['(']:
            c_left = c_left +1
        elif token == vocab[')']:
            c_right = c_right +1
    

    if c_right > c_left:
        for j in range(c_right - c_left):
            new_candidate.pop()
            
    if c_right < c_left:
        for j in range(c_left - c_right):
            new_candidate.append(vocab[')'])
            

    return new_candidate
    
def to_lisp_tree(toks,vocab):
  def recurse(i):
    if i>=len(toks):
      return None,i
      
    if toks[i] == vocab['(']:
      subtrees = []
      j = i+1
      while True:
        subtree, j = recurse(j)
        
        if subtree!=None:
          subtrees.append(subtree)
            
        if j>=len(toks):
          return None, j
          
        if toks[j] == vocab[')']:
          return subtrees, j + 1
          
    else:
      return toks[i], i+1

  try:
    lisp_tree, final_ind = recurse(0)
    return lisp_tree
  except Exception as e:
    print >> sys.stderr, 'Failed to convert "%s" to lisp tree' % toks
    print(e)
    return None

def sort_args(ans,vocab,comm_dict):
    lisp_tree = to_lisp_tree(ans,vocab)
    if lisp_tree is None: return ans
    
    # Post-order traversal, sorting as we go
    def recurse(node):
      if isinstance(node, str): return
      if isinstance(node, int): return
      
      for child in node:
        recurse(child)
        
      if node[0] in comm_dict:
        node[1:] = sorted(node[1:], key=lambda x: str(x))
        
    recurse(lisp_tree)
        
    return tree_to_list(lisp_tree,vocab)
    
def tree_to_str(node):
    if isinstance(node, str):
        return node
    elif isinstance(node, int):
        return str(node)
    else:
        return '( %s )' % ' '.join(tree_to_str(child) for child in node)
        
def tree_to_list(tree,vocab):
    if isinstance(tree, str):
        return [tree]
    elif isinstance(tree, int):
        return [tree]
        
    ret = [vocab["("]]
    for node in tree:
        ret += tree_to_list(node,vocab)
        
    ret.append(vocab[")"])
    
    return ret
    
def init_comm_dict(vocab):
    comm_dict = {}
    for operator in ['_and', '_or','and','or','and:<>','or:<>','_and_','next_to:<>']:
        if operator in vocab: comm_dict[vocab[operator]] = 1
    return comm_dict
    
def is_all_same(candidate,reference):
    if len(candidate)!=len(reference):
        return False
        
    ans = True
    for i in range(len(candidate)):
        if candidate[i]!=reference[i]:
            ans = False
            break
            
    return ans

# compute accuracy between candidate and reference logical forms
def compute_tree_accuracy(candidate_list,reference_list,vocab,rev_vocab,comm_dict,display):

    sorted_candidate, sorted_reference = None, None
    # used only for overnight dataset
    if ('call' in vocab) and ('SW.filter' in vocab):
        sorted_candidate = flatten_filter(candidate_list,vocab)
        sorted_reference = flatten_filter(reference_list,vocab)
    else:
        sorted_candidate = candidate_list
        sorted_reference = reference_list
        

    sorted_candidate = sort_args(sorted_candidate,vocab,comm_dict)
    sorted_reference = sort_args(sorted_reference,vocab,comm_dict)
    
    if display:
        print()
        sorted_candidate_str = '%s' % ' '.join(rev_vocab[token] for token in sorted_candidate)
        sorted_reference_str = '%s' % ' '.join(rev_vocab[token] for token in sorted_reference)
        print(sorted_reference_str)
        print(sorted_candidate_str)
    
    return is_all_same(sorted_candidate,sorted_reference)

# function to detect special SW.filter tokens in overnight    
# dataset, return [arg1,[arg2,arg3,...]] else return none
def is_filter(node_list,vocab):
    
    sym_call, sym_filter = None,None
    if vocab==None:
        sym_call = 'call'
        sym_filter = 'SW.filter'
    else:
        sym_call = vocab['call']
        sym_filter = vocab['SW.filter']
    
    filter_args = []
    if len(node_list)>=4:
        if node_list[0]==sym_call and node_list[1]==sym_filter:
            for i in range(3,len(node_list)):
                filter_args.append(node_list[i])
           
            res_list = is_filter(node_list[3],vocab)
            if res_list==None:
                return [node_list[3],filter_args]
            else:
                filter_end = res_list[0]
                sub_filter_args = res_list[1]
                for v1 in sub_filter_args:
                    filter_args.append(v1)
                
                return [filter_end,filter_args]            
    else:
        return None
    
    return None

# function which flattens all SW.filter arguments in
# overnight dataset logical forms    
def flatten_filter(lf_list,vocab):
    
    flatten_lisp_tree = []
    lisp_tree = to_lisp_tree(lf_list,vocab)
    #print(lisp_tree)
    for node in lisp_tree:
        if isinstance(node,list):
            # find SW.filter
            res_list = is_filter(node,vocab)
            if res_list != None:
                new_node = []
                new_node.append(node[0]) # add call
                new_node.append(node[1]) # add SW.filter
                filter_end = res_list[0]
                filter_args = res_list[1] 
                new_node.append(filter_end) # add filter end token               
                sorted_args = sort_filter_args(filter_args) 
                for v1 in sorted_args:
                    new_node.append(v1)
                
                flatten_lisp_tree.append(new_node)
            else:
                flatten_lisp_tree.append(node)
            
        else:
            flatten_lisp_tree.append(node)
        
    #print(flatten_lisp_tree)
    return tree_to_list(flatten_lisp_tree,vocab)
    
# function to sort arguments of SW.filter
def sort_filter_args(args):
    sorted_args, arg_dict, key_table  = [], {}, []
    for arg in args:
        arg_key = str(arg)
        if arg_key not in arg_dict:
            arg_dict[arg_key] = arg
            key_table.append(arg_key)
      
    key_table.sort()
    for key in key_table:
        #print(key)
        sorted_args.append(arg_dict[key])

    #print('\n')
    #print(sorted_args)
    return sorted_args

# test function
def acc_test():
    vocab = {"(":"(",")":")","call":"call","SW.filter":"SW.filter"}
    lf = '( call SW.listValue ( call SW.filter ( call SW.getProperty ( call SW.singleton en.restaurant ) ( string ! type ) ) ( string neighborhood ) ( string = ) en.neighborhood.midtown_west ) )'
    lf_list = lf.split()
    for k in lf_list: vocab[k] = k
    flat_lf_list = flatten_filter(lf_list,vocab)
    print(flat_lf_list)
    lf = '( call SW.listValue ( call SW.filter ( call SW.filter ( call SW.getProperty ( call SW.singleton en.restaurant ) ( string ! type ) ) ( string neighborhood ) ( string = ) en.neighborhood.midtown_west ) ( string neighborhood ) ( string = ) en.neighborhood.midtown_west ) )'
    lf_list = lf.split()
    for k in lf_list: vocab[k] = k
    flat_lf_list = flatten_filter(lf_list,vocab)
    print(flat_lf_list)
    
#acc_test()

