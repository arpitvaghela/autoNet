
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

## Remotely train model for iris dataset

The worker process trains the model and returns the log (acurracy) to the master process

- Start two terminal (two processes)

```sh
# Process 1
$ python master.py

Epoch 0 Accuracy: 0.699999988079071
Epoch 10 Accuracy: 0.7333333492279053
Epoch 20 Accuracy: 0.7333333492279053
Epoch 30 Accuracy: 0.7666666507720947
Epoch 40 Accuracy: 0.800000011920929
Epoch 50 Accuracy: 0.8333333134651184
Epoch 60 Accuracy: 0.8333333134651184
Epoch 70 Accuracy: 0.8999999761581421
Epoch 80 Accuracy: 0.8999999761581421
Epoch 90 Accuracy: 0.9333333373069763
Model(
  (layer1): Linear(in_features=4, out_features=50, bias=True)
  (layer2): Linear(in_features=50, out_features=50, bias=True)
  (layer3): Linear(in_features=50, out_features=3, bias=True)
)
```

```sh
# Process 2
$ python worker.py

Worker Started
100%|█████████████████████████| 100/100 [00:00<00:00, 1065.68it/s]
```
