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

## demo4.py
Add new actions, transform and detransform.

### Trans: 
Uniformly pick a node and add a unary nonlinear node between it and its parent.
### Detrans: 
Uniformly pick a unary nonlinear node and substitute it with its child.
### Proposal:
P_grow = (1-p_0)/3 * min {1,5/(N_nt+2)}, N_nt is the number of non-terminal nodes

P_prune = (1-p_0)/3 - P_grow

P_detrans = (1-p_0)/3 * (n1 / 2+ n1), n1 is the number of unary nonlinear nodes

P_trans = (1-p_0)/ [3*(N+5)], N is the number of all nodes

P_reop = P_refeat = (1-P_grow-P_prune-P_detrans-P_trans)/2

## demo5.py
Modify the definition of transform and detransform.

### Trans:
take a candidate node and place some operator as its parent
### Detrans:
take a candidate node and delete it; the candidate should not be 'ln', should not be the root if its child nodes are all terminal.

if it has two child nodes, randomly preserve one child (need to be non-terminal).
### Proposal:
P_grow = (1-p_0)/3 * min {1,4/(N_nt+2)}, N_nt is the number of non-terminal nodes

P_prune = (1-p_0)/3 - P_grow

P_detrans = (1-p_0)/3 * (Nc / Nc+3), Nc is the number of candidates for detransformation.

P_trans = (1-p_0)/ 3 - P_detrans; The probs added nodes should be proportional to the preset weights.

P_reop = P_refeat = (1-p_0)/ 6