# MCMC-SymReg
symbolic regression using mcmc sampling

## Symbolic_Regression_Tree_MCMC.pdf
Note for proposed algorithm.

## demo.py
The first version, implementing the mcmc method for symbolic regression.

## demo2.py
In the ReassignOperator transition, unary and binary operators are able to change to each other.

## demo3.py
Let the input operators be stored in a list and can assign weight to the specified operators.
The probability of Grow is $\frac{1-p_0}{2}\cdot \min \{ 1,\frac{5}{N+d+2} \}$.

## demo4.py(editing)
Add new actions, transform and detransform.
Trans: 
Uniformly pick a node and add a unary nonlinear node between it and its parent.
Detrans: 
Uniformly pick a unary nonlinear node and substitute it with its child.

P_grow = (1-p_0)/3 * min {1,5/(N+d+2)}
P_prune = (1-p_0)/4 - P_grow
P_detrans = (1-p_0)/3 * (n1 / 2+ n1), n1 is the number of unary nonlinear nodes
P_trans = (1-p_0)/ [3*(N+5)], N is the number of all nodes
P_reop = P_refeat = (1-P_grow-P_prune-P_detrans-P_trans)/2