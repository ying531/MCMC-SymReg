# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from scipy.stats import invgamma
from scipy.stats import norm
import sklearn
from sklearn.datasets import load_iris
import copy
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn
import random

class Operator:
    def __init__(self,name,function,arity):
        self.name = name
        self.func = function
        self.arity = arity # num of inputs
        

class Node:
    def __init__(self, depth):
        # tree structure attributes
        self.type = -1
            #-1 represents newly grown node (not decided yet)
            #0 represents no child, as a terminal node
            #1 represents one child, 
            #2 represents 2 children
        self.order = 0
        self.left = None
        self.right = None
            #if type=1, the left child is the only one
        self.depth = depth
        self.parent = None
        
        # calculation attributes
        self.operator = None
        self.op_ind = None
            #operator is a string, either "+","*","ln","exp","inv"
        self.data = None
        self.feature = None
            #feature is a int indicating the index of feature in the input data
        # possible parameters
        self.a = None
        self.b = None
    
    def inform(self):
        print("order:",self.order)
        print("type:",self.type)
        print("depth:",self.depth)
        print("operator:",self.operator)
        print("data:",self.data)
        print("feature:",self.feature)
        if self.operator == 'ln':
            print(" ln_a:",self.a)
            print(" ln_b:",self.b)
        
        return
    

# =============================================================================
# # grow from a node, assign an operator or stop as terminal
# =============================================================================

def grow(node,nfeature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b):
    depth = node.depth

    # deciding the number of child nodes
    if node.depth > 0:
        prob = 1/np.power((1+depth),-beta)
        
        test = np.random.uniform(0,1,1)
        if test > prob: #terminal
            node.feature = np.random.randint(0,nfeature,size=1)
            node.type = 0
        else:
            op_ind = np.random.choice(np.arange(len(Ops)),p=Op_weights)
            node.operator = Ops[op_ind]
            node.type = Op_type[op_ind]
            node.op_ind = op_ind
            
    else:#root node, sure to split
        op_ind = np.random.choice(np.arange(len(Ops)),p=Op_weights)
        node.operator = Ops[op_ind]
        node.type = Op_type[op_ind]
        node.op_ind = op_ind
        
    # grow recursively
    if node.type == 0:
        node.feature = np.random.randint(0,nfeature,size=1)
        
    elif node.type == 1:
        node.left = Node(depth+1)
        node.left.parent = node
        if node.operator == 'ln':#linear parameters
            node.a = norm.rvs(loc=1,scale=np.sqrt(sigma_a))
            node.b = norm.rvs(loc=0,scale=np.sqrt(sigma_b))
        grow(node.left,nfeature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
        
    else: #node.type=2
        node.left = Node(depth+1)
        node.left.parent = node
        node.left.order = len(Tree)
        node.right = Node(depth+1)
        node.right.parent = node
        node.right.order = len(Tree)
        grow(node.left,nfeature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
        grow(node.right,nfeature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
        
    return 
    
    


# =============================================================================
# # generate a list storing the nodes in the tree
# # nodes are stored by induction
# # orders are assigned accordingly
# =============================================================================
def genList(node):
    lst = []
    # terminal node
    if node.left is None:
        lst.append(node)
    else:
        if node.right is None:# with one child
            lst.append(node)
            lst = lst + genList(node.left)
        else:# with two children
            lst.append(node)
            lst = lst + genList(node.left)
            lst = lst + genList(node.right)
    for i in np.arange(0,len(lst)):
        lst[i].order = i
    return(lst)
    

# =============================================================================
# # cut all child nodes of a current node
# # turn the node into a terminal one
# =============================================================================
def shrink(node):
    if node.left is None:
        print("Already a terminal node!")
    else:
        node.left = None
        node.right = None
        node.type = 0
        node.operator = None #delete operator
        node.a = None #delete parameters
        node.b = None
    return
    
# =============================================================================
# # upgrade 'order' attribute of nodes in Tree
# # Tree is a list containing nodes of a tree
# =============================================================================
def upgOd(Tree):
    for i in np.arange(0,len(Tree)):
        Tree[i].order = i
    return


# =============================================================================
# # calculate a tree output from node
# =============================================================================
def allcal(node,indata):
    if node.type == 0:#terminal node
        if indata is not None:
            node.data = np.array(indata.iloc[:,node.feature])
    elif node.type == 1:#one child node
        if node.operator == 'ln':
            node.data = node.a * allcal(node.left,indata) + node.b
        elif node.operator == 'exp':
            node.data = np.exp(allcal(node.left,indata))
        elif node.operator == 'inv':
            node.data = 1/allcal(node.left,indata)
        elif node.operator == 'neg':
            node.data = -1 * allcal(node.left,indata)
        elif node.operator == 'sin':
            node.data = np.sin(allcal(node.left,indata))
        elif node.operator == 'cos':
            node.data = np.cos(allcal(node.left,indata))
        else:
            print("No matching type and operator!")
    elif node.type == 2:#two child nodes
        if node.operator == '+':
            node.data = allcal(node.left,indata) + allcal(node.right,indata)
        elif node.operator == '*':
            node.data = allcal(node.left,indata) * allcal(node.right,indata)
        else:
            print("No matching type and operator!")
    elif node.type == -1:#not grown
        print("Not a grown tree!")
    else:
        print("No legal node type!")
            
    return node.data

        
# =============================================================================
# # display the structure of the tree, each node displays operator
# # Tree is a list storing the nodes
# =============================================================================
def display(Tree):
    tree_depth = -1
    for i in np.arange(0,len(Tree)):
        if Tree[i].depth > tree_depth:
            tree_depth = Tree[i].depth
    dlists = []
    for d in np.arange(0,tree_depth+1):
        dlists.append([])
    for i in np.arange(0,len(Tree)):
        dlists[Tree[i].depth].append(Tree[i])
    
    for d in np.arange(0,len(dlists)):
        st = " "
        for i in np.arange(0,len(dlists[d])):
            if dlists[d][i].type > 0:
                st = st + dlists[d][i].operator + " "
            else:
                st = st + str(dlists[d][i].feature) + " "
        print(st)
    return



# =============================================================================
# # get the height of a (sub)tree with node being root node
# # equivalently, the maximum distance from node to its descendent
# # only a root node has height 0
# # terminal nodes has height 0
# =============================================================================
def getHeight(node):
    if node.type == 0:
        return 0
    elif node.type == 1:
        return getHeight(node.left)+1
    else:
        lheight = getHeight(node.left)
        rheight = getHeight(node.right)
        return max(lheight,rheight)+1
    
    
# =============================================================================
# # get the number of nodes of a (sub)tree with node being root
# =============================================================================
def getNum(node):
    if node.type == 0:
        return 1
    elif node.type == 1:
        return getNum(node.left)+1
    else:
        lnum = getNum(node.left)
        rnum = getNum(node.right)
        return (lnum+rnum+1)
        
# =============================================================================
# # get the number of lt() operators of a (sub)tree with node being root
# =============================================================================
def numLT(node):
    if node.type == 0:
        return 0
    elif node.type == 1:
        if node.operator == 'ln':
            return 1+numLT(node.left)
        else:
            return numLT(node.left)
    else:
        return numLT(node.left)+numLT(node.right)


# =============================================================================
# # returns a string of the expression of the tree
# # node is the root of the tree
# =============================================================================
def Express(node):
    expr = ""
    if node.type == 0:#terminal
        expr = "x"+str(node.feature)
        return(expr)
    elif node.type == 1:
        if node.operator == 'exp':
            expr = "exp(" + Express(node.left) +")"
        elif node.operator == 'ln':
            expr = str(round(node.a,4)) + "*(" + Express(node.left) + ")+" + str(round(node.b,4))
        elif node.operator == 'inv':# node.operator == 'inv':
            expr = "1/[" + Express(node.left) +"]"
        elif node.operator == 'sin':
            expr = "sin(" + Express(node.left) +")"
        elif node.operator == 'cos':
            expr = "cos(" + Express(node.left) +")"
        else:#note.operate=='neg'
            expr = "-(" + Express(node.left) +")"
            
    else:#node.type==2
        if node.operator == '+':
            expr = Express(node.left) + "+" + Express(node.right)
        else:
            expr = "(" + Express(node.left) + ")*(" + Express(node.right) +")"
    return(expr)


# =============================================================================
# # compute the likelihood of tree structure f(S)
# # P(M,T)*P(theta|M,T)*P(theta|sigma_theta)*P(sigma_theta)*P(theta)
# =============================================================================
def fStruc(node,n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b):
    loglike = 0 #log.likelihood of structure S=(T,M)
    loglike_para = 0 #log.likelihood of linear paras
    
    '''
    # contribution of hyperparameter sigma_theta
    if node.depth == 0:#root node
        loglike += np.log(invgamma.pdf(node.sigma_a,1))
        loglike += np.log(invgamma.pdf(node.sigma_b,1))
    '''
    
    # contribution of splitting the node or becoming terminal
    if node.type == 0:#terminal node
        loglike += np.log(1 - 1/np.power((1+node.depth),-beta))#* np.power(node.depth,beta) #contribution of choosing terminal
        loglike -= np.log(n_feature) #contribution of feature selection
    elif node.type == 1:#unitary operator
        # contribution of splitting
        if node.depth == 0: #root node
            loglike += np.log(Op_weights[node.op_ind])
        else:
            loglike += np.log((1+node.depth))*beta + np.log(Op_weights[node.op_ind])
        # contribution of parameters of linear nodes
        if node.operator == 'ln':
            loglike_para -= np.power((node.a-1),2) / (2*sigma_a)
            loglike_para -= np.power(node.b,2) / (2*sigma_b)
            loglike_para -= 0.5 * np.log(2*np.pi*sigma_a)
            loglike_para -= 0.5 * np.log(2*np.pi*sigma_b)
    else:#binary operator
        # contribution of splitting
        if node.depth == 0: #root node
            loglike += np.log(Op_weights[node.op_ind])
        else:
            loglike += np.log((1+node.depth))*beta + np.log(Op_weights[node.op_ind])
            
    
    # contribution of child nodes
    if node.left is None:#no child nodes
        return [loglike, loglike_para]
    else:
        fleft = fStruc(node.left,n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
        loglike += fleft[0]
        loglike_para += fleft[1]
        if node.right is None:#only one child
            return [loglike, loglike_para]
        else:
            fright = fStruc(node.right,n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
            loglike += fright[0]
            loglike_para += fright[1]
    
    return [loglike, loglike_para]


# =============================================================================
# # propose a new tree from existing Root        
# # and calculate the ratio
# # five possible actions: stay, grow, prune, ReassignOp, ReassignFea.
# =============================================================================
def Prop(Root,n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b):
    
    ###############################
    ######### preparations ########
    ###############################
    
    # make a copy of Root
    oldRoot = copy.deepcopy(Root)
    # get necessary auxiliary information
    depth = -1
    Tree = genList(Root)
    for i in np.arange(0,len(Tree)):
        if Tree[i].depth > depth:
            depth = Tree[i].depth
            
    # preserve pointers to originally linear nodes
    lnPointers = []
    last_a = []
    last_b = []
    for i in np.arange(0,len(Tree)):
        if Tree[i].operator == 'ln':
            lnPointers.append(Tree[i])
            last_a.append(Tree[i].a)
            last_b.append(Tree[i].b)
    
    # get the list of terminal nodes
    Term = []#terminal
    Nterm = []#non-terminal
    cnode = None
    for i in np.arange(0,len(Tree)):
        if Tree[i].type == 0:
            Term.append(Tree[i])
        else:
            Nterm.append(Tree[i])
            
    # get the list of lt() nodes
    Lins = []
    for i in np.arange(0,len(Tree)):
        if Tree[i].operator == 'ln':
            Lins.append(Tree[i])
            
    # get necessary quantities
    ltNum = len(Lins)
    nodeNum = len(Tree)
    height = getHeight(Root)
            
    # record expansion and shrinkage
    # expansion occurs when num of lt() increases
    # shrinkage occurs when num of lt() decreases
    change = ''     
    Q = Qinv = 1

    
    
    ###############################
    # decide which action to take #
    ###############################

    # probs of each action
    p_stay = 0.25 * ltNum / (ltNum+3)
    p_grow = 0.75*(1-p_stay)* min(1,5/(len(Nterm)+4))
    p_prune = (1-p_stay)*0.75 - p_grow
    p_rop = (1-p_stay)/8
    
    # auxiliary
    test = np.random.uniform(0,1,1)[0]

    ###############################
    ########### take action #######
    ###############################
    
    # stay
    if test <= p_stay:
        action = 'stay'
        #print('action:',action)
        # calculate Q and Qinv
        Q = p_stay
        Qinv = p_stay
    
    # grow
    elif test <= p_stay + p_grow:
        action = 'grow'
        #print("action:",action)
        
        #pick a terminal node
        pod = np.random.randint(0,len(Term),1)[0]
        # grow the node
        # the likelihood is exactly the same as fStruc(), starting from assigning operator
        grow(Term[pod],n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
        
        if Term[pod].type == 0:#grow to be terminal
            Q = Qinv = 1
        else:
            # calculate Q
            fstrc = fStruc(Term[pod],n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b) 
            Q = p_grow * np.exp(fstrc[0])/len(Term)
            # calculate Qinv (equiv to prune)
            new_ltNum = numLT(Root)
            new_height = getHeight(Root)
            new_nodeNum = getNum(Root)
            newTerm = []#terminal
            newTree = genList(Root)
            new_nterm = []#non-terminal
            for i in np.arange(0,len(newTree)):
                if newTree[i].type == 0:
                    newTerm.append(newTree[i])
                else:
                    new_nterm.append(newTree[i])
            new_termNum = len(newTerm)
            new_p = (1- 0.25 * new_ltNum/(ltNum+3))*0.75* (1- min(1,5/(len(new_nterm)+4)))
            Qinv = new_p / (new_nodeNum-new_termNum-1) #except root node
            
            if new_ltNum > ltNum:
                change = 'expansion'
    
    # prune
    elif test <= p_stay + p_grow + p_prune:
        action = 'prune'
        #print("action:",action)
        
        #pick a node to prune
        pod = np.random.randint(1,len(Nterm),1)[0] # except root node
        fstrc = fStruc(Nterm[pod],n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
        pruned = copy.deepcopy(Nterm[pod]) #preserve a copy

        # preserve pointers to all cutted ln
        p_ltNum = numLT(pruned)
        if p_ltNum > 0:
            change = 'shrinkage'
        
        #prune the node
        Nterm[pod].left = None
        Nterm[pod].right = None
        Nterm[pod].operator = None
        Nterm[pod].type = 0
        Nterm[pod].feature = np.random.randint(0,n_feature,1)
        #print("prune and assign feature:",Par[pod].feature)
        
        #quantities for new tree
        new_ltNum = numLT(Root)
        new_height = getHeight(Root)
        new_nodeNum = getNum(Root)
        newTerm = []#terminal
        new_nTerm = []#non-terminal
        newTree = genList(Root)
        for i in np.arange(0,len(newTree)):
            if newTree[i].type == 0:
                newTerm.append(newTree[i])
            else:
                new_nTerm.append(newTree[i])
        
        #calculate Q
        Q = p_prune / ((len(Nterm)-1)*n_feature)
        
        #calculate Qinv (correspond to grow)
        pg = 1 - 0.25 * new_ltNum/(new_ltNum+3)*0.75 * min(1,5/(len(new_nTerm)+4))
        Qinv = pg * np.exp(fstrc[0])/len(newTerm)
        

    
    # reassignOperator
    elif test <= p_stay + p_grow + p_prune + p_rop:
        action = 'ReassignOperator'
        #print("action:",action)
        pod = np.random.randint(0,len(Nterm),1)[0]
        last_op = Nterm[pod].operator
        last_op_ind = Nterm[pod].op_ind
        last_type = Nterm[pod].type
        #print('replaced operator:',last_op)
        cnode = Nterm[pod]########pointer to the node changed#######
        # a deep copy of the changed node and its descendents
        replaced = copy.deepcopy(Nterm[pod])
        
        new_od = np.random.choice(np.arange(0,len(Ops)),p=Op_weights)
        new_op = Ops[new_od]
        #print('assign new operator:',new_op)
        new_type = Op_type[new_od]
            
        # originally unary
        if last_type == 1:
            if new_type == 1: # unary to unary
                # assign operator and type
                Nterm[pod].operator = new_op
                if last_op == 'ln': #originally linear
                    if new_op != 'ln': #change from linear to other ops 
                        cnode.a = None
                        cnode.b = None
                        change = 'shrinkage'
                else: # orignally not linear
                    if new_op == 'ln':# linear increases by 1
                        ###### a and b is not sampled
                        change = 'expansion'
                
                # calculate Q, Qinv (equal)
                Q = Op_weights[new_od]
                Qinv = Op_weights[last_op_ind]
                
            else:# unary to binary
                # assign operator and type
                cnode.operator = new_op
                cnode.type = 2
                if last_op == 'ln':
                    cnode.a = None
                    cnode.b = None
                    # grow a new sub-tree rooted at right child
                cnode.right = Node(cnode.depth +1)
                grow(cnode.right,n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
                fstrc = fStruc(cnode.right,n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
                
                # calculate Q
                Q = p_rop * np.exp(fstrc[0]) * Op_weights[new_od] / (len(Nterm))
                # calculate Qinv
                    # get necessary quantities
                new_height = getHeight(Root)
                new_nodeNum = getNum(Root)
                newTerm = []#terminal
                newTree = genList(Root)
                new_ltNum = numLT(Root)
                for i in np.arange(0,len(newTree)):
                    if newTree[i].type == 0:
                        newTerm.append(newTree[i])
                    # reversed action is binary to unary
                new_p0 = new_ltNum/(4*(new_ltNum+3))
                Qinv = 0.125 * (1-new_p0) * Op_weights[last_op_ind] / (new_nodeNum-len(newTerm))
                
                # record change of dim
                if new_ltNum > ltNum:
                    change = 'expansion'
                elif new_ltNum < ltNum:
                    change = 'shrinkage'
                    
                

        
        # originally binary
        else:
            if new_type == 1: # binary to unary
                # assign operator and type
                cutted = copy.deepcopy(cnode.right) #deep copy root of the cutted subtree
                # preserve pointers to all cutted ln
                p_ltNum = numLT(cutted)
                if p_ltNum > 1:
                    change = 'shrinkage'
                elif new_op == 'ln':
                    if p_ltNum == 0:
                        change = 'expansion'
                
                cnode.right = None
                cnode.operator = new_op
                cnode.type = new_type
                
                # calculate Q
                Q = p_rop * Op_weights[new_od] / len(Nterm)
                # calculate Qinv
                    # necessary quantities
                new_height = getHeight(Root)
                new_nodeNum = getNum(Root)
                newTerm = []#terminal
                newTree = genList(Root)
                new_ltNum = numLT(Root)
                    # reversed action is unary to binary and grow
                new_p0 = new_ltNum/(4*(new_ltNum+3))
                fstrc = fStruc(cutted,n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
                Qinv = 0.125 * (1-new_p0) * np.exp(fstrc[0]) * Op_weights[last_op_ind] / ((new_nodeNum-len(newTerm)))
                
                
                
            else:# binary to binary
                # assign operator
                cnode.operator = new_op
                # calculate Q,Qinv(equal)
                Q = Op_weights[new_od]
                Qinv = Op_weights[last_op_ind]
    

    # reassign feature            
    else:
        action = 'ReassignFeature'
        #print("action:",action)
        
        #pick a terminal node
        pod = np.random.randint(0,len(Term),1)[0]
        #pick a feature and reassign
        fod = np.random.randint(0,n_feature,1)
        Term[pod].feature = fod
        # calculate Q,Qinv (equal)
        Q = Qinv = 1
        
        
        
        
    return [oldRoot,Root,lnPointers,change,Q,Qinv,last_a,last_b,cnode]

        

# =============================================================================
# # calculate the likelihood of genenerating auxiliary variable
# # change is a string with value of 'expansion' or 'shrinkage'
# # oldRoot is the root of the original Tree
# # Root is the root of the new tree
# # cnode is a pointer to the node just changed (expand or shrink 'ln') (if applicable)
# # last_a and last_b is the parameters for the shrinked node (if applicable)
# # lnPointers is list of pointers to originally linear nodes in Tree before changing
# =============================================================================
def auxProp(change,oldRoot,Root,lnPointers,sigma_a,sigma_b,last_a,last_b,cnode=None):
    # record the informations of linear nodes other than the shrinked or expanded
    odList = [] #list of orders of linear nodes
    Tree = genList(Root)
    
    for i in np.arange(0,len(Tree)):
        if Tree[i].operator == 'ln':
            odList.append(i)
    
    # sample new sigma_a2 and sigma_b2
    new_sa2 = invgamma.rvs(1)
    new_sb2 = invgamma.rvs(1)
    old_sa2 = sigma_a
    old_sb2 = sigma_b
            
    if change == 'shrinkage':
        prsv_aList = []
        prsv_bList = []
        cut_aList = []
        cut_bList = []
        # find the preserved a's
        for i in np.arange(0,len(lnPointers)):
            if lnPointers[i].operator == 'ln':#still linear
                prsv_aList.append(last_a[i])
                prsv_bList.append(last_b[i])
            else:#no longer linear
                cut_aList.append(last_a[i])
                cut_bList.append(last_b[i])
        # substitute those cutted with newly added if cut and add
        for i in np.arange(0,len(odList)-len(prsv_aList)):
            prsv_aList.append(cut_aList[i])
            prsv_bList.append(cut_bList[i])
                
        n0 = len(prsv_aList)


        # sample auxiliary U
        UaList = []
        UbList = []
        for i in np.arange(0,n0):
            UaList.append(norm.rvs(loc=0,scale=np.sqrt(new_sa2)))
            UbList.append(norm.rvs(loc=0,scale=np.sqrt(new_sb2)))
    
        # generate inverse auxiliary U*
        NaList = [] #Theta* with a 
        NbList = [] #Theta* with b
        NUaList = [] #U* with a
        NUbList = [] #U* with b
        for i in np.arange(0,n0):
            NaList.append(prsv_aList[i]+UaList[i])
            NbList.append(prsv_bList[i]+UbList[i])
            NUaList.append(prsv_aList[i]-UaList[i])
            NUbList.append(prsv_bList[i]-UbList[i])
        NUaList = NUaList + last_a
        NUbList = NUbList + last_b
    
        # hstar is h(U*|Theta*,S*,S^t) (here we calculate the log) corresponding the shorter para
        # h is h(U|Theta,S^t,S*) corresponding longer para
        # Theta* is the 
        logh = 0
        loghstar = 0
        
        # contribution of new_sa2 and new_sb2 
        logh += np.log(invgamma.pdf(new_sa2,1))
        logh += np.log(invgamma.pdf(new_sb2,1))
        loghstar += np.log(invgamma.pdf(old_sa2,1))
        loghstar += np.log(invgamma.pdf(old_sb2,1))
        
        for i in np.arange(0,len(UaList)):
            # contribution of UaList and UbList
            logh += np.log(norm.pdf(UaList[i],loc=0,scale=np.sqrt(new_sa2)))
            logh += np.log(norm.pdf(UbList[i],loc=0,scale=np.sqrt(new_sb2)))
        
        for i in np.arange(0,len(NUaList)):
            # contribution of NUaList and NUbList
            loghstar += np.log(norm.pdf(NUaList[i],loc=0,scale=np.sqrt(old_sa2)))
            loghstar += np.log(norm.pdf(NUbList[i],loc=0,scale=np.sqrt(old_sb2)))
        
        hratio = np.exp(loghstar - logh)
        #print("hratio:",hratio)
        
        # determinant of jacobian 
        detjacob = np.power(2,2*len(prsv_aList))
        #print("detjacob:",detjacob)
        
        
        #### assign Theta* to the variables
        # new values of Theta
        # new sigma_a, sigma_b are directly returned
        for i in np.arange(0,len(odList)):
            Tree[odList[i]].a = NaList[i]
            Tree[odList[i]].b = NbList[i]

            
        return [hratio,detjacob,new_sa2,new_sb2]
        
        
        
    elif change == 'expansion':
        # sample new sigma_a2 and sigma_b2
        new_sa2 = invgamma.rvs(1)
        new_sb2 = invgamma.rvs(1)
        old_sa2 = sigma_a
        old_sb2 = sigma_b
        
        # lists of theta_0 and expanded ones
        # last_a is the list of all original a's
        # last_b is the list of all original b's
        odList = []
        for i in np.arange(0,len(Tree)):
            if Tree[i].operator == 'ln':
                odList.append(i)
        
        # sample auxiliary U
        UaList = []
        UbList = []
        for i in np.arange(0,len(last_a)):
            UaList.append(norm.rvs(loc=0,scale=np.sqrt(new_sa2)))
            UbList.append(norm.rvs(loc=0,scale=np.sqrt(new_sb2)))
        
        # generate inverse auxiliary U* and new para Theta*
        NaList = [] #Theta*_a
        NbList = [] #Theta*_b
        NUaList = [] #U*_a
        NUbList = [] #U*_b
        for i in np.arange(0,len(last_a)):
            NaList.append((last_a[i]+UaList[i])/2)
            NbList.append((last_b[i]+UbList[i])/2)
            NUaList.append((last_a[i]-UaList[i])/2)
            NUbList.append((last_b[i]-UbList[i])/2)
            
        # append newly generated a and b into NaList and NbList
        nn = len(odList) - len(last_a) # number of newly added ln
        for i in np.arange(0,nn):
            u_a = norm.rvs(loc=1,scale=np.sqrt(new_sa2))
            u_b = norm.rvs(loc=0,scale=np.sqrt(new_sb2))
            NaList.append(u_a)
            NbList.append(u_b)
            
        # calculate h ratio
        # logh is h(U|Theta,S,S*) correspond to jump from short to long (new)
        # loghstar is h(Ustar|Theta*,S*,S) correspond to jump from long to short
        logh = 0
        loghstar = 0
        
        # contribution of sigma_ab
        logh += np.log(invgamma.pdf(new_sa2,1))
        logh += np.log(invgamma.pdf(new_sb2,1))
        loghstar += np.log(invgamma.pdf(old_sa2,1))
        loghstar += np.log(invgamma.pdf(old_sb2,1))
        
        # contribution of u_a, u_b
        for i in np.arange(len(last_a),nn):
            logh += norm.pdf(NaList[i],loc=1,scale=np.sqrt(new_sa2))
            logh += norm.pdf(NbList[i],loc=0,scale=np.sqrt(new_sb2))
        
        # contribution of U_theta
        for i in np.arange(0,len(UaList)):
            logh += np.log(norm.pdf(UaList[i],loc=0,scale=np.sqrt(new_sa2)))
            logh += np.log(norm.pdf(UbList[i],loc=0,scale=np.sqrt(new_sb2)))
        
        for i in np.arange(0,len(NUaList)):
            loghstar += np.log(norm.pdf(NUaList[i],loc=0,scale=np.sqrt(old_sa2)))
            loghstar += np.log(norm.pdf(NUbList[i],loc=0,scale=np.sqrt(old_sb2)))
            
        # compute h ratio
        hratio = np.exp(loghstar - logh)
        
        # determinant of jacobian 
        detjacob = 1/np.power(2,2*len(last_a))
        
        #### assign Theta* to the variables
        # new values of sigma_a sigma_b
        # new values of Theta
        for i in np.arange(0,len(odList)):
            Tree[odList[i]].a = NaList[i]
            Tree[odList[i]].b = NbList[i]


        return [hratio,detjacob,new_sa2,new_sb2]
        
        
    else:# same set of parameters
        # record the informations of linear nodes other than the shrinked or expanded
        odList = [] #list of orders of linear nodes
        Tree = genList(Root)

        old_sa2 = sigma_a
        old_sb2 = sigma_b
        
        for i in np.arange(0,len(Tree)):
            if Tree[i].operator == 'ln':
                odList.append(i)
        
        NaList = []
        NbList = []
        new_sa2 = invgamma.rvs(1)
        new_sb2 = invgamma.rvs(1)
        for i in np.arange(0,len(odList)):
            NaList.append(norm.rvs(loc=1,scale=np.sqrt(new_sa2)))
            NbList.append(norm.rvs(loc=0,scale=np.sqrt(new_sb2)))
        '''
        # log likelihood of parameters (including sigma_ab)
        logT = np.log(invgamma.pdf(old_sa2,1)) + np.log(invgamma.pdf(old_sb2,1))
        for i in np.arange(0,len(aList)):
            logT += np.log(norm.pdf(aList[i],loc=1,scale=np.sqrt(old_sa2)))
            logT += np.log(norm.pdf(bList[i],loc=0,scale=np.sqrt(old_sb2)))
        # log likelihood of new parameters
        logTstar = np.log(invgamma.pdf(new_sa2,1)) + np.log(invgamma.pdf(new_sb2,1))
        for i in np.arange(0,len(NaList)):
            logTstar += np.log(norm.pdf(NaList[i],loc=1,scale=np.sqrt(new_sa2)))
            logTstar += np.log(norm.pdf(NbList[i],loc=0,scale=np.sqrt(new_sb2)))
        
        fratio = np.exp(logT-logTstar)
        return fratio
        '''
        
        # new values of Theta
        for i in np.arange(0,len(odList)):
            Tree[odList[i]].a = NaList[i]
            Tree[odList[i]].b = NbList[i]

        
        return [new_sa2,new_sb2]



# =============================================================================
# # calculate the log likelihood f(y|S,Theta,x) 
# # (S,Theta) is represented by node Root
# # prior is y ~ N(output,sigma)
# =============================================================================
def ylogLike(y,indata,Root,sigma):
    output = allcal(Root,indata)
    error = 0
    for i in np.arange(0,len(y)):
        error += (y[i]-output[i,0])*(y[i]-output[i,0])
    error = np.sqrt(error/len(y))
    #print("mean error:",error)
    
    log_sum = 0
    for i in np.arange(0,len(y)):
        temp = np.power(y[i]-output[i,0],2) #np.log(norm.pdf(y[i],loc=output[i,0],scale=np.sqrt(sigma)))
        
        #print(i,temp)
        log_sum += temp
    log_sum = -log_sum / (2*sigma*sigma)
    log_sum -= 0.5*len(y) * np.log(2*np.pi*sigma*sigma)
    return(log_sum)



# =============================================================================
# # prop new structure, sample new parameters and decide whether to accept
# # Root is the root node of the tree to be changed
# # sigma is for output to y
# # sigma_a, sigma_b are (squared) hyper-paras for linear paras
# =============================================================================
def newProp(Root,sigma,y,indata,n_feature,Ops,Op_weights,Op_type,eta,sigma_a,sigma_b):
    
    [oldRoot,Root,lnPointers,change,Q,Qinv,last_a,last_b,cnode] = Prop(Root,n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
    sig = 4
    new_sigma = invgamma.rvs(sig)
    
    if change == 'shrinkage':
        # the parameters are upgraded as well
        # contribution of h and determinant of jacobian
        [hratio,detjacob,new_sa2,new_sb2] = auxProp(change,oldRoot,Root,lnPointers,sigma_a,sigma_b,last_a,last_b,cnode)
        #print("hratio:",hratio)
        #print("detjacob:",detjacob)
        
        # contribution of f(y|S,Theta,x)
        #print("new sigma:",round(new_sigma,3))
        yllstar = ylogLike(y,indata,Root,new_sigma)
        #print("sigma:",round(sigma,3))
        yll = ylogLike(y,indata,oldRoot,sigma)
        
        log_yratio = yllstar - yll
        #print("log yratio:",log_yratio)
        
        # contribution of f(Theta,S)
        strucl = fStruc(Root,n_feature,Ops,Op_weights,Op_type,beta,new_sa2,new_sb2)
        struclstar = fStruc(oldRoot,n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
        sl = strucl[0]+strucl[1]
        slstar = struclstar[0]+struclstar[1]
        log_strucratio = slstar-sl#struclstar / strucl
        #print("log strucratio:",log_strucratio)
        
        # contribution of proposal Q and Qinv
        log_qratio = np.log(Qinv / Q)
        #print("log qratio:",log_qratio) 
        
        # R
        logR = log_yratio + log_strucratio + log_qratio + np.log(hratio) + np.log(detjacob)
        logR = logR + np.log(invgamma.pdf(new_sigma,sig)) - np.log(invgamma.pdf(sigma,sig))
        #print("logR:",logR)
        
    elif change == 'expansion':
        # the parameters are upgraded as well
        # contribution of h and determinant of jacobian
        [hratio,detjacob,new_sa2,new_sb2] = auxProp(change,oldRoot,Root,lnPointers,sigma_a,sigma_b,last_a,last_b,cnode)
        #print("hratio:",hratio)
        #print("detjacob:",detjacob)
        
        # contribution of f(y|S,Theta,x)
        #print("new sigma:",round(new_sigma,3))
        yllstar = ylogLike(y,indata,Root,new_sigma)
        #print("sigma:",round(sigma,3))
        yll = ylogLike(y,indata,oldRoot,sigma)

        log_yratio = yllstar - yll
        #print("log yratio:",log_yratio)
        
        # contribution of f(Theta,S)
        strucl = fStruc(Root,n_feature,Ops,Op_weights,Op_type,beta,new_sa2,new_sb2)
        struclstar = fStruc(oldRoot,n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
        sl = strucl[0]+strucl[1]
        slstar = struclstar[0]+struclstar[1]
        log_strucratio = slstar-sl#struclstar / strucl
        #print("log strucratio:",log_strucratio)
        
        # contribution of proposal Q and Qinv
        log_qratio = np.log(Qinv / Q)
        #print("log qratio:",log_qratio) 
        
        # R
        logR = log_yratio + log_strucratio + log_qratio + np.log(hratio) + np.log(detjacob)
        logR = logR + np.log(invgamma.pdf(new_sigma,sig)) - np.log(invgamma.pdf(sigma,sig))
        #print("logR:",logR)
        
    else: # no dimension jump
        # the parameters are upgraded as well
        # contribution of fratio
        [new_sa2,new_sb2] = auxProp(change,oldRoot,Root,lnPointers,sigma_a,sigma_b,last_a,last_b,cnode)
        
        # contribution of f(y|S,Theta,x)
        #print("new sigma:",round(new_sigma,3))
        yllstar = ylogLike(y,indata,Root,new_sigma)
        #print("sigma:",round(sigma,3))
        yll = ylogLike(y,indata,oldRoot,sigma)
        
        log_yratio = yllstar - yll
        #print("log yratio:",log_yratio)
        #yratio = np.exp(yllstar-yll)
        
        # contribution of f(Theta,S)                                                                                                                                                                                                                      
        strucl = fStruc(Root,n_feature,Ops,Op_weights,Op_type,beta,new_sa2,new_sb2)[0]
        struclstar = fStruc(oldRoot,n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)[0]
        log_strucratio = struclstar - strucl
        #print("log strucratio:",log_strucratio)
        
        # contribution of proposal Q and Qinv
        log_qratio = np.log(Qinv / Q)
        #print("log qratio:",log_qratio)
        
        # R
        logR = log_yratio + log_strucratio + log_qratio 
        logR = logR + np.log(invgamma.pdf(new_sigma,sig)) - np.log(invgamma.pdf(sigma,sig))
        #print("logR:",logR)
        
    
    alpha = min(logR,0)
    test = np.random.uniform(low=0,high=1,size=1)[0]
    if np.log(test) >= alpha: #no accept
        #print("no accept")
        Root = oldRoot
        return [False, sigma, copy.deepcopy(oldRoot),sigma_a,sigma_b]
    else:
        #print("||||||accepted||||||")
        return [True, new_sigma,  copy.deepcopy(Root),new_sa2,new_sb2]
















# =============================================================================
# # data processing
# =============================================================================
'''
# load data
data = load_iris()['data']
indata = pd.DataFrame(data)

# define the ground truth
y = 10/indata.iloc[:,3] + indata.iloc[:,1]
yy = 10*indata.iloc[:,2] + 1 + np.exp(indata.iloc[:,3])
yy = np.exp(indata.iloc[:,0]) + indata.iloc[:,1] * indata.iloc[:,2] * np.exp(indata.iloc[:,3])

yy = indata.iloc[:,0]*2.5 + indata.iloc[:,1] + np.exp(indata.iloc[:,3]) +1


noise = norm.rvs(loc=0,scale=0.5,size=len(yy)) 
y = yy + noise
'''

# load data

dat_amount = pd.read_csv("amount.csv")
dat_close = pd.read_csv("close.csv")
dat_high = pd.read_csv("high.csv")
dat_low = pd.read_csv("low.csv")
dat_open = pd.read_csv("open.csv")
dat_rate = pd.read_csv("rate.csv")
dat_volume = pd.read_csv("volume.csv")

#amount = dat_amount.iloc[:,1]/1e+8
close = dat_close.iloc[:,1]/500
high = dat_high.iloc[:,1]/500
low = dat_low.iloc[:,1]/500
openn = dat_low.iloc[:,1]/500
volume = dat_volume.iloc[:,1]/1e+5
rate = dat_rate.iloc[:,1]


rtn_shift = 1
retro = 3

data = pd.concat([close,high,low,volume],axis=1)
data.columns = ['close','high','low','volume']
data['rtn'] = 0
for i in np.arange(0,data.shape[0]-rtn_shift):
    data.iloc[i,4] = data.iloc[i+rtn_shift]['close']/data.iloc[i]['close']-1
data = copy.deepcopy(data.iloc[:data.shape[0]-rtn_shift,:])



for i in np.arange(1,retro):
    temp = copy.deepcopy(data[['close','high','low','volume']])
    for j in np.arange(-temp.shape[0]+1,-i+1):
        temp.iloc[-j] = temp.iloc[-j-i]
    temp.columns = ['close'+str(i),'high'+str(i),'low'+str(i),'volume'+str(i)]
    data = pd.concat([temp,data],axis=1)
data = copy.deepcopy(data.iloc[2:,:])
data.index = np.arange(data.shape[0])

data['rtn'] = data['rtn']*100

for cl in np.arange(0,len(data.columns)):
    data = data[~data[data.columns[cl]].isin([0.0])]


indata = copy.deepcopy(data.iloc[:,0:data.shape[1]-1])
indata.index = np.arange(indata.shape[0])
y = copy.deepcopy(data['rtn'])
y.index = np.arange(len(y))

n_data = len(y)
n_train = int(np.floor(n_data*0.8))
n_test = n_data-n_train
train_ind = np.random.choice(n_data,size=n_train,replace=False)
all_ind = np.arange(0,n_data)
test_ind = np.array([item for item in all_ind if item not in train_ind])

test_data = copy.deepcopy(indata.iloc[test_ind,:])
test_y = y[test_ind]
#test_yy = yy[test_ind]


train_data = copy.deepcopy(indata.iloc[train_ind,:])
train_y = y[train_ind]

y = list(y)
test_y = list(test_y)
train_y = list(train_y)
train_data.index = np.arange(0,n_train)
test_data.index = np.arange(0,n_test)
#test_yy = list(test_yy)


# =============================================================================
# # y=6*sinx1*cosx2
# =============================================================================
random.seed(1)
n = 100
x1 = np.random.uniform(0.1,5.9,n)
x2 = np.random.uniform(0.1,5.9,n)
x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)
train_data = pd.concat([x1,x2],axis=1)
train_y = 6 * np.sin(train_data.iloc[:,0]) * np.cos(train_data.iloc[:,1])
xx1 = []
xx2 = []
for i in np.arange(31):
    for j in np.arange(31):
        xx1.append(-0.15 + 0.2*i)
        xx2.append(-0.15 + 0.2*j)
xx1 = pd.DataFrame(xx1)
xx2 = pd.DataFrame(xx2)
test_data = pd.concat([xx1,xx2],axis=1)
test_y = 6 * np.sin(test_data.iloc[:,0]) * np.cos(test_data.iloc[:,1])


# =============================================================================
# # MCMC algorithm
# =============================================================================

n_feature = train_data.shape[1]
n_train = train_data.shape[0]
n_test = test_data.shape[0]

alpha1 = 0.4
alpha2 = 0.4
beta = -1.2

Ops = ['inv','ln','neg','sin','cos','+','*']
Op_weights = [0.15,0.15,0.1,0.15,0.15,0.15,0.15]
Op_type = [1,1,1,1,1,2,2]
n_op = len(Ops)

K = 5
N = 500
LLHS = np.zeros([N,K])

RootLists = []
ErrLists = []
TotalLists = []
TestLists = []
DentLists = []

val = 50000

# run for multiple chains
for count in np.arange(0,K):
    # create a new Root node
    # tree is stored as a list of nodes
    Tree = []
    Root = Node(0)
    sigma_a = invgamma.rvs(1)
    sigma_b = invgamma.rvs(1)
    sigma = invgamma.rvs(1) #for output y
    
    # initialization
    # grow a tree from the Root node
    grow(Root,n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
    Tree = genList(Root)
        
    total = 0
    accepted = 0
    errList = []
    rootList = []
    totList = []
    llhList = []
    testList = []
    dentList = []
    
    print("-----------------")
    print(count,"th chain:")
    print("-----------------")
    
    while sum(totList)<2*val and total < val:
        
        [res, sigma, Root, sigma_a, sigma_b] = newProp(Root,sigma,train_y,train_data,n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
        total += 1
        
        if res is True:
            accepted += 1
            fstruc = fStruc(Root,n_feature,Ops,Op_weights,Op_type,beta,sigma_a,sigma_b)
            llh = fstruc[0]+fstruc[1]
            llhList.append(llh)
            rootList.append(copy.deepcopy(Root))
            
            output = allcal(Root,train_data)
            error = 0
            for i in np.arange(0,n_train):
                error += (output[i,0]-train_y[i])*(output[i,0]-train_y[i])
            rmse = np.sqrt(error/n_train)
            errList.append(rmse)
            
            toutput = allcal(Root,test_data)
            terror = 0
            for i in np.arange(0,n_test):
                terror += (toutput[i,0]-test_y[i])*(toutput[i,0]-test_y[i])
            trmse = np.sqrt(terror/n_test)
            testList.append(trmse)
            
            '''
            dterror = 0
            for i in np.arange(0,n_test):
                dterror += (toutput[i,0]-test_yy[i])*(toutput[i,0]-test_yy[i])
            dtrmse = np.sqrt(dterror/n_test)
            dentList.append(dtrmse)
            '''
            print("accept",accepted,"th after",total,"proposals")
            print("sigma:",round(sigma,5),"error:",round(rmse,5))#,"log.likelihood:",round(llh,5))
            #print("denoised rmse:",round(dtrmse,5))
            
            display(genList(Root))
            print("---------------")
            totList.append(total)
            total = 0 



    LLHS[0:len(llhList),count] = llhList
    RootLists.append(rootList)
    ErrLists.append(errList)
    TotalLists.append(totList)
    TestLists.append(testList)
    DentLists.append(dentList)

    '''
    plt.plot(np.arange(0,len(errList)),errList)
    plt.plot(np.arange(0,len(testList)),testList,color='red')
    plt.show()
    
    plt.plot(np.arange(0,len(totList)),totList)
    plt.show()
    
    plt.plot(np.arange(0,len(llhList)),llhList)
    plt.show()
    '''
    print("------")
    print("mean rmse of last 5 accepts:",np.mean(errList[-6:-1]))
    print("mean rmse of last 5 tests:",np.mean(testList[-6:-1]))
    
    for i in np.arange(0,len(output)):
        print(output[i],train_y[i])
        
    for i in np.arange(len(rootList)-10,len(rootList)):
        display(genList(rootList[i]))
        print("----")
        print("")
        


# =============================================================================
# =============================================================================
'''
Root = copy.deepcopy(rootList[-1])
output = allcal(Root,indata)
for i in np.arange(0,len(y)):
    print(y[i],output[i,0])


Tree = genList(Root)
display(Tree)
Express(Root)

'''
plt.yticks(np.arange(0,50,5))
plt.grid(color='gray', linestyle='--', linewidth=1,alpha=0.3)
cols = ['r','b','g','k','m']
#plt.plot(np.arange(len(errList)),errList,color=cols[0],linewidth=1)
#plt.plot(np.arange(len(testList)),testList,color=cols[1],linewidth=1)

for i in np.arange(K):
    plt.plot(np.arange(len(ErrLists[i][-10:-1])),ErrLists[i][-10:-1],color=cols[i],linewidth=1)
#plt.savefig('errplot.png',dpi=400)
plt.show()


# =============================================================================
# # diagnose
# =============================================================================

print('---------------------')
print("Gelman-Rubin Diagnosis:",pm.diagnostics.gelman_rubin(LLHS))

props = []
for i in np.arange(K):
    props.append(sum(TotalLists[i]))
print('---------------------')
print("Number of all proposals:",props)







# =============================================================================
# # Genetic Programming
# =============================================================================

import gplearn
from gplearn.genetic import SymbolicRegressor

def _protected_exponent(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 100, np.exp(x1), 0.)

expo = gplearn.functions.make_function(_protected_exponent,name='exp',arity=1)

est_gp = SymbolicRegressor(population_size=500,
                           generations=200, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0,
                           function_set=['add','sub','mul','div','sin','cos',expo],
                           metric='rmse')

est_gp.fit(train_data,train_y)

print('---------------------')
print("Estimated model by Genetic Programming:")
print(est_gp._program)
estd = est_gp.predict(test_data)
gperror = np.sqrt(sum((estd-test_y)**2)/len(estd))
print("-----")
print("RMSE of test data:",round(gperror,5))
#gpderror = np.sqrt(sum((estd-test_yy)**2)/len(estd))
#print("RMSE of denoised test data:",round(gpderror,5))





        