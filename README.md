
# autoNet: Differentiable Architecture Search and Weighted Random Hyperparameter Selection

## Introduction

We address the problem of Architecture Search through the method of DARTS: Differentiable Architecture Search and enhance the same with hyperparameter search via Weighted Random Hyperparameter search with fixed two-point initialisation.

As a part of the university curriculum, this project was an Artificial Intelligence and Cloud Computing Joint Project.

## Approach 
![Alt text](weighted_random.png?raw=true "Title")

## Cloud Platform
-   We build a platform using a high performance microservices architecture with advanced asynchronous intercommunication between microservices using queues and scheduling
    
-   A robust system is built by extensive use of docker containerisation, Network virtualization and GPU passthrough
    
-   Infinite scaling of workers is supported and load balancing on inference APIs. All services represented under one umbrella API gateway.
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
