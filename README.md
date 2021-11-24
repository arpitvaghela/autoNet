
# autoNet: Differentiable Architecture Search and Weighted Random Hyperparameter Selection

## Introduction

We address the problem of Architecture Search through the method of DARTS: Differentiable Architecture Search and enhance the same with hyperparameter search via Weighted Random Hyperparameter search with fixed two-point initialisation.

As a part of the university curriculum, this project was an Artificial Intelligence and Cloud Computing Joint Project.

## Approach 
![Alt text](DARTS.PNG?raw=true "Title")

The methodology of the Differentiable Architecture Search (DARTS), referring to the above figure, can be summarised as (a)  Operations  on  the  edges  are  initially unknown. (b) Continuous relaxation of the search space by placing a mixture of  candidate  operations  on  each  edge.  (c)  Joint  optimization  of  the  mixing probabilities and the network weights by solving a bilevel optimization problem. (d) Inducing the final architecture from the learned mixing probabilities.

![Alt text](weighted_random.jpg?raw=true "Title")

The  approach  of  Weighted  Random  search  starts  initiates with  identifying  the  range  of  hyperparameters  that  formulate a  finite N-dimensional  space.  The  next  step  is  identifying the  diagonal  of  the  n-dim  space  such  that  one  end  lies  onthe  minimal  values  of  hyperparams  and  the  other  lies  on  the maximum. Two points should be selected such that these two points divide the diagonal in three equal parts. Model training shall be performed with the two sets of hyperparameters that are represented by these hyperparameters.The remaining steps are repeated iteratively as many times as  desired.  Given  the  selected  points  in  the  n-dim  space,  a validation  loss  is  associated  with  the  each  and  every  set.  A probabilistic  random  selection  shall  happen  given  two  points with  the  minimum  validation  loss  at  a  given  stage. The probability  of  the  next  point  being  selected  has  the  nearest point  as x1 shall  depend  on  the  validation  loss v1 tied  with x1.


## Results
![Alt text](Darts-1.PNG?raw=true "Title")

The result above represents the performance of the optimised architecture through DARTS in image classification on CIFAR-10 dataset

![Alt text](DARTS-2.PNG?raw=true "Title")

The result above represents the performance of the optimised architecture through DARTS in image classification on Penn Treebank dataset

![Alt text](wr_loss.png?raw=true "Title")

The result above represents the variation of Validation Loss in architecture with increasing iterations of Weighted Random Hyperparameter search with two point fixed initialisation

![Alt text](wr_acc.png?raw=true "Title")

The result above represents the variation of Accuracy in architecture with increasing iterations of Weighted Random Hyperparameter search with two point fixed initialisation

## Installation

Prior to running the project, install the dependencies with the following command
```sh
pip install requirements.txt
```

## References
arXiv:1806.09055
