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
            #operator is a string, either "+","*","ln","exp","inv"
        self.data = None
        self.feature = None
            #feature is a int indicating the index of feature in the input data
        # possible parameters
        self.a = None
        self.b = None
        self.sigma_a = None
        self.sigma_b = None
    
    def inform(self):
        print("order:",self.order)
        print("type:",self.type)
        print("depth:",self.depth)
        print("operator:",self.operator)
        print("data:",self.data)
        print("feature:",self.feature)
        
        return
    


def grow(node,nfeature,alpha1,alpha2,beta,sigma_a,sigma_b):
    depth = node.depth

    # deciding the number of child nodes
    if node.depth > 0:
        prob1 = alpha1 * np.power(depth,beta)
        prob2 = alpha2 * np.power(depth,beta)
        test = np.random.uniform(0,1,1)
        print("test:",test,"prob1:",prob1,"prob2:",prob2)
        if test <= prob1:
            num = 1
            #self.likelihood = self.likelihood * prob1 / 3
        elif test <= prob1+prob2:
            num = 2
            #self.likelihood = self.likelihood * prob2 * 0.5
        else:
            num= 0
            #self.likelihood = self.likelihood * (1-prob1-prob2)
    else:
        prob1 = alpha1/(alpha1+alpha2)
        prob2 = alpha2/(alpha1+alpha2)
        test = np.random.uniform(0,1,1)
        if test <= prob1:
            num = 1
            #self.likelihood = self.likelihood * prob1 / 3
        else:
            num = 2
            
    node.type = num 
    print("num:",num)

    # deciding the operators and creating new nodes
    if num == 1:
        node.left = Node(depth+1)
        node.left.parent = node
        node.left.sigma_a = node.sigma_a
        node.left.sigma_b = node.sigma_b
        
        test = np.random.uniform(0,1,1)
        if test <= 1/3:#exponential
            node.operator = 'exp'
        elif test <= 2/3:#inverse
            node.operator = 'inv'
        else:#linear transformation
            node.operator = 'ln'
            node.a = norm.rvs(loc=1,scale=np.sqrt(sigma_a))
            #self.likelihood *= norm.pdf(loc=1,scale=np.sqrt(self.sigma_a))
            node.b = norm.rvs(loc=0,scale=np.sqrt(sigma_b))

        print("assigned operator:",node.operator)

    elif num == 2:
        node.left = Node(depth+1)
        node.left.parent = node
        node.left.order = len(Tree)
        node.right = Node(depth+1)
        node.right.parent = node
        node.right.order = len(Tree)
        
        node.left.sigma_a = node.sigma_a
        node.left.sigma_b = node.sigma_b
        node.right.sigma_a = node.sigma_a
        node.right.sigma_b = node.sigma_b
        
        test = np.random.uniform(0,1,1)
        if test <= 1/2:
            node.operator = '+'
        else:
            node.operator = '*'
        print("assigned operator:",node.operator)
    else:
        print("set as terminal")
        node.feature = np.random.randint(0,nfeature,size=1)
    return
    
    


# generate a list storing the nodes in the tree
# nodes are stored by induction
# orders are assigned accordingly
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
    

# cut all child nodes of a current node
# turn the node into a terminal one
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
    
# upgrade 'order' attribute of nodes in Tree
# Tree is a list containing nodes of a tree
def upgOd(Tree):
    for i in np.arange(0,len(Tree)):
        Tree[i].order = i
    return


# calculate a tree output from node
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

        
# display the structure of the tree, each node displays operator
# Tree is a list storing the nodes
# 
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

# returns a string of the expression of the tree
# node is the root of the tree
def Express(node):
    expr = ""
    if node.type == 0:#terminal
        expr = "x"+str(node.feature)
        return(expr)
    elif node.type == 1:
        if node.operator == 'exp':
            expr = "exp(" + Express(node.left) +")"
        elif node.operator == 'ln':
            expr = "(" + str(round(node.a,4)) + "*" + Express(node.left) + "+" + str(round(node.b,4)) + ")"
        else:# node.operator == 'inv':
            expr = "1/" + Express(node.left)
    else:#node.type==2
        if node.operator == '+':
            expr = "(" + Express(node.left) + "+" + Express(node.right) + ")"
        else:
            expr = Express(node.left) + "*" + Express(node.right)
    return(expr)


# compute the likelihood of tree structure f(S)
# P(M,T)*P(theta|M,T)*P(theta|sigma_theta)*P(sigma_theta)*P(theta)
def fStruc(node,n_feature,alpha1,alpha2,beta):
    loglike = 0
    loglike_para = 0
    
    # contribution of hyperparameter sigma_theta
    if node.depth == 0:#root node
        loglike += np.log(invgamma.pdf(node.sigma_a,1))
        loglike += np.log(invgamma.pdf(node.sigma_b,1))
        
    # contribution of splitting the node or becoming terminal
    if node.type == 0:#terminal node
        loglike += np.log(1 - (alpha1+alpha2)) + beta * np.log(node.depth)#* np.power(node.depth,beta) #contribution of choosing terminal
        loglike -= np.log(n_feature) #contribution of feature selection
    elif node.type == 1:#unitary operator
        # contribution of splitting
        if node.depth == 0: #root node
            loglike += np.log(alpha1/(alpha1+alpha2))
        else:
            loglike += np.log(alpha1) + np.log(node.depth)*beta - np.log(3)
        # contribution of parameters of linear nodes
        if node.operator == 'ln':
            loglike_para -= np.power((node.a-1),2) / (2*node.sigma_a)
            loglike_para -= np.power(node.b,2) / (2*node.sigma_b)
            loglike_para -= 0.5 * np.log(2*np.pi*node.sigma_a)
            loglike_para -= 0.5 * np.log(2*np.pi*node.sigma_b)
    else:#binary operator
        # contribution of splitting
        if node.depth == 0: #root node
            loglike += np.log(alpha2/(alpha1+alpha2))
        else:
            loglike += np.log(alpha2) + np.log(node.depth)*beta - np.log(2)
            
    
    # contribution of child nodes
    if node.left is None:#no child nodes
        return [loglike, loglike_para]
    else:
        fleft = fStruc(node.left,n_feature,alpha1,alpha2,beta)
        loglike += fleft[0]
        loglike_para += fleft[1]
        if node.right is None:#only one child
            return [loglike, loglike_para]
        else:
            fright = fStruc(node.right,n_feature,alpha1,alpha2,beta)
            loglike += fright[0]
            loglike_para += fright[1]
    
    return [loglike, loglike_para]


# propose a new tree from existing Root        
# and calculate the ratio
# four possible actions: grow, prune, ReassignOp, ReassignFea.
def Prop(Root,n_feature,alpha1,alpha2):
    # make a copy of Root
    oldRoot = copy.deepcopy(Root)
    # get necessary auxiliary information
    depth = -1
    Tree = genList(Root)
    for i in np.arange(0,len(Tree)):
        if Tree[i].depth > depth:
            depth = Tree[i].depth
    # get the list of terminal nodes
    Term = []
    Nterm = []
    cnode = None
    for i in np.arange(0,len(Tree)):
        if Tree[i].type == 0:
            Term.append(Tree[i])
        else:
            Nterm.append(Tree[i])
    #get the list of parents of terminal nodes
    Par = [] 
    for i in np.arange(0,len(Tree)):
        #print(i)
        if Tree[i].type == 1:
            if Tree[i].left.type == 0:
                Par.append(Tree[i])
        elif Tree[i].type == 2:
            if Tree[i].left.type == 0 and Tree[i].right.type == 0:
                Par.append(Tree[i])
    #record expansion and shrinkage
    change = '' 
    last_a = None
    last_b = None
    # expansion occurs when grow into 'ln' node or reassign 'ln' operator
    # shrinkage occurs when pruning a 'ln' node or reassign from 'ln'
    
    
    # decide which action to take
    test = np.random.uniform(0,1,1)[0]
    
    Pgrow = 2.5/(len(Tree)+depth+2)
    print("Pgrow:",Pgrow)
    
    #perform action
    if test <= Pgrow:
        action = 'grow'
        print("action:",action)
        pod = np.random.randint(0,len(Term),1)[0]#pick a terminal node
        test = np.random.uniform(0,1,1)
        if test <= alpha1/ (alpha1+alpha2):#unary
            Term[pod].type = 1
            Term[pod].left = Node(Term[pod].depth+1)
            Term[pod].left.type = 0
            Term[pod].left.feature = np.random.randint(0,n_feature,1)
            # choose operator
            pop = np.random.randint(0,3,1)[0]
            op1 = ['exp','inv','ln']
            Term[pod].operator = op1[pop]
            print("grow 1 node and assign operator",Term[pod].operator)
            # record expansion
            if op1[pop] == 'ln':
                #Term[pod].a = norm.rvs(loc=1,scale=np.sqrt(Term[pod].sigma_a))
                #Term[pod].b = norm.rvs(loc=0,scale=np.sqrt(Term[pod].sigma_b))
                change = 'expansion' #expans = True
                #new_a = Term[pod].a
                #new_b = Term[pod].b
                cnode = Term[pod] #pointer to the node changed
            # calculate Q
            Q = Pgrow*alpha1 / (3*len(Term)*n_feature*(alpha1+alpha2))
            # calculate Qinv (correpond to prune)
                #calculate new depth
            newdepth = -1
            Tree = genList(Root)
            for i in np.arange(0,len(Tree)):
                if Tree[i].depth > newdepth:
                    newdepth = Tree[i].depth
            Qinv = (1-2.5/(len(Tree)+newdepth+2))/(len(Par)*n_feature)
        else:#binary
            Term[pod].type = 2
            Term[pod].left = Node(Term[pod].depth+1)
            Term[pod].left.type = 0
            Term[pod].left.feature = np.random.randint(0,n_feature,1)
            Term[pod].right = Node(Term[pod].depth+1)
            Term[pod].right.type = 0
            Term[pod].right.feature = np.random.randint(0,n_feature,1)
            # choose operator
            pop = np.random.randint(0,2,1)[0]
            op2 = ['+','*']
            Term[pod].operator = op2[pop]
            print("grow 2 nodes and assign operator",Term[pod].operator)
            # calculate Q
            Q = Pgrow * alpha2 / ((alpha1+alpha2)*2*len(Term)*n_feature*n_feature)
            # calculate Qinv (correpond to prune)
                #calculate new depth
            newdepth = -1
            Tree = genList(Root)
            for i in np.arange(0,len(Tree)):
                if Tree[i].depth > newdepth:
                    newdepth = Tree[i].depth
            Qinv = (0.5-2.5/(len(Tree)+newdepth+2))/(len(Par)*n_feature)
            
    elif test <= 0.5:
        action = 'prune'
        print("action:",action)
        #pick a node to prune
        pod = np.random.randint(0,len(Par),1)[0]
        lastop = Par[pod].operator
        lasttype = Par[pod].type
        if lastop == 'ln':
            change = 'shrinkage'
            last_a = Par[pod].a
            last_b = Par[pod].b
            cnode = Par[pod]
            
        #prune the node
        Par[pod].left = None
        Par[pod].right = None
        Par[pod].operator = None
        Par[pod].type = 0
        Par[pod].feature = np.random.randint(0,n_feature,1)
        print("prune and assign feature:",Par[pod].feature)
        
        #calculate Q
        Q = (0.5-Pgrow) / (len(Par)*n_feature)
        
        #calculate Qinv (correspond to grow)
        newdepth = -1
        Tree = genList(Root)
        for i in np.arange(0,len(Tree)):
            if Tree[i].depth > newdepth:
                newdepth = Tree[i].depth
        newPgrow = 2.5 / (len(Tree)+newdepth+2)
        if lasttype == 1:#originally unary
            Qinv = newPgrow * alpha1 / ((alpha1+alpha2)*3*len(Term)*n_feature)
        else:#originally binary
            Qinv = newPgrow * alpha2 / ((alpha1+alpha2)*2*(len(Term)-1)*n_feature*n_feature)
            
    elif test <= 0.75:
        action = 'ReassignOperator'
        print("action:",action)
        pod = np.random.randint(0,len(Nterm),1)[0]
        if Nterm[pod].type == 1:#unary
            pop = np.random.randint(0,3,1)[0]
            op1 = ['exp','inv','ln']
            # Q = Qinv
            if Nterm[pod].operator == 'ln':
                if pop < 2:
                    change = 'shrinkage'
                    last_a = Nterm[pod].a
                    last_b = Nterm[pod].b
                    cnode = Nterm[pod] #pointer to the node shrinked
                    Nterm[pod].a = None
                    Nterm[pod].b = None
                    Nterm[pod].operator = op1[pop]
            else:#originally not linear
                if pop < 2:
                    Nterm[pod].operator = op1[pop]
                else:
                    change = 'expansion'
                    Nterm[pod].operator = op1[pop]
                    cnode = Nterm[pod] #pointer to the node expanded
            Q = Qinv = 1
        else:#binary
            pop = np.random.randint(0,2,1)[0]
            op2 = ['+','*']
            Nterm[pod].operator = op2[pop]
            # Q = Qinv
            Q = Qinv = 1
        print("reassign operator:",Nterm[pod].operator)
        
    else:
        action = 'ReassignFeature'
        print("action:",action)
        pod = np.random.randint(0,len(Term),1)[0]
        Term[pod].feature = np.random.randint(0,n_feature,1)
        print("reassign feature:",Term[pod].feature)
        # Q = Qinv
        Q = Qinv = 1
        
    return [oldRoot,Root,cnode,change,Q,Qinv,last_a,last_b]
        
        

# calculate the likelihood of genenerating auxiliary variable
# change is a string with value of 'expansion' or 'shrinkage'
# oldRoot is the root of the original Tree
# Root is the root of the new tree
# cnode is a pointer to the node just changed (expand or shrink 'ln') (if applicable)
# last_a and last_b is the parameters for the shrinked node (if applicable)
def auxProp(change,oldRoot,Root,cnode=None,last_a=None,last_b=None):
    # record the informations of linear nodes other than the shrinked or expanded
    odList = [] #list of orders of linear nodes
    aList = [] #list of "a" of linear nodes
    bList = [] #list of "b" of linear nodes
    Tree = genList(Root)
    oldTree = genList(oldRoot)
    for i in np.arange(0,len(Tree)):
        if Tree[i].operator == 'ln' and cnode is not Tree[i]:
            odList.append(i)
            aList.append(Tree[i].a)
            bList.append(Tree[i].b)
            
    if change == 'shrinkage':
        # sample new sigma_a2 and sigma_b2
        new_sa2 = invgamma.rvs(1)
        new_sb2 = invgamma.rvs(1)
        old_sa2 = Root.sigma_a
        old_sb2 = Root.sigma_b

        # sample auxiliary U
        UaList = []
        UbList = []
        for i in np.arange(0,len(aList)):
            UaList.append(norm.rvs(loc=0,scale=np.sqrt(new_sa2)))
            UbList.append(norm.rvs(loc=0,scale=np.sqrt(new_sb2)))
            
        # generate inverse auxiliary U*
        NaList = [] #Theta* with a 
        NbList = [] #Theta* with b
        NUaList = [] #U* with a
        NUbList = [] #U* with b
        for i in np.arange(0,len(aList)):
            NaList.append(aList[i]+UaList[i])
            NbList.append(bList[i]+UbList[i])
            NUaList.append(aList[i]-UaList[i])
            NUbList.append(bList[i]-UbList[i])
        # the last element of NU
        NUaList.append(last_a)
        NUbList.append(last_b)
        
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
            loghstar += np.log(norm.pdf(NUaList[i],loc=1,scale=np.sqrt(old_sa2)))
            loghstar += np.log(norm.pdf(NUbList[i],loc=0,scale=np.sqrt(old_sb2)))
        
        hratio = np.exp(loghstar - logh)
        #print("hratio:",hratio)
        
        # determinant of jacobian 
        detjacob = np.power(2,2*len(aList))
        #print("detjacob:",detjacob)
        
        
        #### assign Theta* to the variables
        # new values of sigma_a sigma_b
        # new values of Theta
        for i in np.arange(0,len(odList)):
            Tree[odList[i]].a = NaList[i]
            Tree[odList[i]].b = NbList[i]

        for i in np.arange(0,len(Tree)):
            Tree[i].sigma_a = new_sa2
            Tree[i].sigma_b = new_sb2
            
        return [hratio,detjacob]
        
        
        
    elif change == 'expansion':
        # sample new sigma_a2 and sigma_b2
        new_sa2 = invgamma.rvs(1)
        new_sb2 = invgamma.rvs(1)
        old_sa2 = Root.sigma_a
        old_sb2 = Root.sigma_b
        
        # sample auxiliary U
        UaList = []
        UbList = []
        for i in np.arange(0,len(aList)):
            UaList.append(norm.rvs(loc=1,scale=np.sqrt(new_sa2)))
            UbList.append(norm.rvs(loc=0,scale=np.sqrt(new_sb2)))
        
        # generate inverse auxiliary U* and new para Theta*
        NaList = []
        NbList = []
        NUaList = []
        NUbList = []
        for i in np.arange(0,len(aList)):
            NaList.append((aList[i]+UaList[i])/2)
            NbList.append((bList[i]+UbList[i])/2)
            NUaList.append((aList[i]-UaList[i])/2)
            NUbList.append((bList[i]-UbList[i])/2)
            
        # append newly generated a and b into NaList and NbList
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
        logh += norm.pdf(u_a,loc=1,scale=np.sqrt(new_sa2))
        logh += norm.pdf(u_b,loc=1,scale=np.sqrt(new_sb2))
        
        # contribution of U_theta
        for i in np.arange(0,len(UaList)):
            logh += np.log(norm.pdf(UaList[i],loc=1,scale=np.sqrt(new_sa2)))
            logh += np.log(norm.pdf(UbList[i],loc=0,scale=np.sqrt(new_sb2)))
        
        for i in np.arange(0,len(NUaList)):
            loghstar += np.log(norm.pdf(NUaList[i],loc=0,scale=np.sqrt(old_sa2)))
            loghstar += np.log(norm.pdf(NUbList[i],loc=0,scale=np.sqrt(old_sb2)))
            
        # compute h ratio
        hratio = np.exp(loghstar - logh)
        
        # determinant of jacobian 
        detjacob = 1/np.power(2,2*len(aList))
        
        #### assign Theta* to the variables
        # new values of sigma_a sigma_b
        # new values of Theta
        for i in np.arange(0,len(odList)):
            Tree[odList[i]].a = NaList[i]
            Tree[odList[i]].b = NbList[i]

        cnode.a = u_a
        cnode.b = u_b
        
        for i in np.arange(0,len(Tree)):
            Tree[i].sigma_a = new_sa2
            Tree[i].sigma_b = new_sb2
        
        return [hratio,detjacob]
        
        
    else:# same set of parameters
        # record the informations of linear nodes other than the shrinked or expanded
        odList = [] #list of orders of linear nodes
        aList = [] #list of "a" of linear nodes
        bList = [] #list of "b" of linear nodes
        Tree = genList(Root)
        for i in np.arange(0,len(Tree)):
            if Tree[i].operator == 'ln':
                odList.append(i)
                aList.append(Tree[i].a)
                bList.append(Tree[i].b)
        old_sa2 = Root.sigma_a
        old_sb2 = Root.sigma_b
        
        NaList = []
        NbList = []
        new_sa2 = invgamma.rvs(1)
        new_sb2 = invgamma.rvs(1)
        for i in np.arange(0,len(aList)):
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

        for i in np.arange(0,len(Tree)):
            Tree[i].sigma_a = new_sa2
            Tree[i].sigma_b = new_sb2
        
        return




# calculate the log likelihood f(y|S,Theta,x) 
# (S,Theta) is represented by node Root
# prior is y ~ N(output,sigma)
def ylogLike(y,indata,Root,sigma):
    output = allcal(Root,indata)
    error = 0
    for i in np.arange(0,len(y)):
        error += (y[i]-output[i,0])*(y[i]-output[i,0])
    error = np.sqrt(error/len(y))
    print("mean error:",error)
    
    log_sum = 0
    for i in np.arange(0,len(y)):
        temp = np.power(y[i]-output[i,0],2) #np.log(norm.pdf(y[i],loc=output[i,0],scale=np.sqrt(sigma)))
        
        #print(i,temp)
        log_sum += temp
    log_sum = -log_sum / (2*sigma*sigma)
    log_sum -= 0.5*len(y) * np.log(2*np.pi*sigma*sigma)
    return(log_sum)





# load data
data = load_iris()['data']
indata = pd.DataFrame(data)

# define the ground truth
'''
y = indata.iloc[:,0]*2 + indata.iloc[:,1] + np.exp(indata.iloc[:,3])
y = np.exp(indata.iloc[:,0]) + indata.iloc[:,1] * indata.iloc[:,2] * np.exp(indata.iloc[:,3])
y = 10*indata.iloc[:,2] + 1 + np.exp(indata.iloc[:,3])
'''
y = 10/indata.iloc[:,3] + indata.iloc[:,1]


n_feature = 4


# create a new Root node
# tree is stored as a list of nodes
Tree = []
Root = Node(0)

Root.sigma_a = invgamma.rvs(1)
Root.sigma_b = invgamma.rvs(1)

sigma = invgamma.rvs(1)

alpha1 = 0.4
alpha2 = 0.4
beta = -0.8


# initialization
grow(Root,n_feature,alpha1,alpha2,beta,Root.sigma_a,Root.sigma_b)
Tree = genList(Root)

flag = True
while flag == True:
    flag = False
    for i in np.arange(0,len(Tree)):
        if Tree[i].type == -1:
            grow(Tree[i],n_feature,alpha1,alpha2,beta,Root.sigma_a,Root.sigma_b)
            flag = True
    Tree = genList(Root)
    


    
total = 0
accepted = 0
errList = []
rootList = []


while accepted<20:
    [res, sigma, Root] = newProp(Root,sigma,y,indata,n_feature,alpha1,alpha2)
    total += 1
    
    if res is True:
        accepted += 1
        rootList.append(copy.deepcopy(Root))
        output = allcal(Root,indata)
        error = 0
        for i in np.arange(0,len(y)):
            error += (output[i,0]-y[i])*(output[i,0]-y[i])
        mean_error = np.sqrt(error/len(y))
        errList.append(mean_error)
        print("sigma:",sigma,"error:",mean_error)
        





def newProp(Root,sigma,y,indata,n_feature,alpha1,alpha2):
    
    [oldRoot,Root,cnode,change,Q,Qinv,last_a,last_b] = Prop(Root,n_feature,alpha1,alpha2)
    oldTree = genList(oldRoot)
    Tree = genList(Root)
    display(oldTree)
    print("")
    display(Tree)
    print("change:",change)
    
    new_sigma = invgamma.rvs(1)
    
    
    
    if change == 'shrinkage':
        # the parameters are upgraded as well
        # contribution of h and determinant of jacobian
        [hratio,detjacob] = auxProp(change,oldRoot,Root,cnode,last_a,last_b)
        print("hratio:",hratio)
        print("detjacob:",detjacob)
        
        # contribution of f(y|S,Theta,x)
        print("new sigma:",round(new_sigma,3))
        yllstar = ylogLike(y,indata,Root,new_sigma)
        print("sigma:",round(sigma,3))
        yll = ylogLike(y,indata,oldRoot,sigma)
        
        log_yratio = yllstar - yll
        print("log yratio:",log_yratio)
        
        # contribution of f(Theta,S)
        strucl = fStruc(Root,n_feature,alpha1,alpha2,beta)
        struclstar = fStruc(oldRoot,n_feature,alpha1,alpha2,beta)
        sl = strucl[0]+strucl[1]
        slstar = struclstar[0]+struclstar[1]
        log_strucratio = slstar-sl#struclstar / strucl
        print("log strucratio:",log_strucratio)
        
        # contribution of proposal Q and Qinv
        log_qratio = np.log(Qinv / Q)
        print("log qratio:",log_qratio) 
        
        # R
        logR = log_yratio + log_strucratio + log_qratio + np.log(hratio) + np.log(detjacob)
        logR = logR + np.log(invgamma.pdf(new_sigma,1)) - np.log(invgamma.pdf(sigma,1))
        print("logR:",logR)
        
    elif change == 'expansion':
        # the parameters are upgraded as well
        # contribution of h and determinant of jacobian
        [hratio,detjacob] = auxProp(change,oldRoot,Root,cnode)
        print("hratio:",hratio)
        print("detjacob:",detjacob)
        
        # contribution of f(y|S,Theta,x)
        print("new sigma:",round(new_sigma,3))
        yllstar = ylogLike(y,indata,Root,new_sigma)
        print("sigma:",round(sigma,3))
        yll = ylogLike(y,indata,oldRoot,sigma)

        log_yratio = yllstar - yll
        print("log yratio:",log_yratio)
        
        # contribution of f(Theta,S)
        strucl = fStruc(Root,n_feature,alpha1,alpha2,beta)
        struclstar = fStruc(oldRoot,n_feature,alpha1,alpha2,beta)
        sl = strucl[0]+strucl[1]
        slstar = struclstar[0]+struclstar[1]
        log_strucratio = slstar-sl#struclstar / strucl
        print("log strucratio:",log_strucratio)
        
        # contribution of proposal Q and Qinv
        log_qratio = np.log(Qinv / Q)
        print("log qratio:",log_qratio) 
        
        # R
        logR = log_yratio + log_strucratio + log_qratio + np.log(hratio) + np.log(detjacob)
        logR = logR + np.log(invgamma.pdf(new_sigma,1)) - np.log(invgamma.pdf(sigma,1))
        print("logR:",logR)
        
    else: # no dimension jump
        # the parameters are upgraded as well
        # contribution of fratio
        auxProp(change,oldRoot,Root)
        
        # contribution of f(y|S,Theta,x)
        print("new sigma:",round(new_sigma,3))
        yllstar = ylogLike(y,indata,Root,new_sigma)
        print("sigma:",round(sigma,3))
        yll = ylogLike(y,indata,oldRoot,sigma)
        
        log_yratio = yllstar - yll
        print("log yratio:",log_yratio)
        #yratio = np.exp(yllstar-yll)
        
        # contribution of f(Theta,S)                                                                                                                                                                                                                      
        strucl = fStruc(Root,n_feature,alpha1,alpha2,beta)[0]
        struclstar = fStruc(oldRoot,n_feature,alpha1,alpha2,beta)[0]
        log_strucratio = struclstar - strucl
        print("log strucratio:",log_strucratio)
        
        # contribution of proposal Q and Qinv
        log_qratio = np.log(Qinv / Q)
        print("log qratio:",log_qratio)
        
        # R
        logR = log_yratio + log_strucratio + log_qratio 
        logR = logR + np.log(invgamma.pdf(new_sigma,1)) - np.log(invgamma.pdf(sigma,1))
        print("logR:",logR)
        
        
    alpha = min(logR,0)
    test = np.random.uniform(low=0,high=1,size=1)[0]
    if np.log(test) >= alpha: #no accept
        print("no accept")
        Root = oldRoot
        return [False, sigma, copy.deepcopy(oldRoot)]
    else:
        print("||||||accepted||||||")
        return [True, new_sigma,  copy.deepcopy(Root)]
        
























        