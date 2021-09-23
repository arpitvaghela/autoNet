import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tqdm
import torch.distributed.rpc as rpc

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

def print_acc(ep, acc):
    print(f"Epoch {ep} Accuracy: {acc}")

def train():
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    names = iris["target_names"]
    feature_names = iris["feature_names"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=2)

    model     = Model(X_train.shape[1])
    optimizer = th.optim.Adam(model.parameters(), lr=0.001)
    loss_fn   = nn.CrossEntropyLoss()
    EPOCHS  = 100
    X_train = Variable(th.from_numpy(X_train)).float()
    y_train = Variable(th.from_numpy(y_train)).long()
    X_test  = Variable(th.from_numpy(X_test)).float()
    y_test  = Variable(th.from_numpy(y_test)).long()

    loss_list     = np.zeros((EPOCHS,))
    accuracy_list = np.zeros((EPOCHS,))

    for epoch in tqdm.trange(EPOCHS):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss_list[epoch] = loss.item()
        
        # Zero gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with th.no_grad():
            y_pred = model(X_test)
            correct = (th.argmax(y_pred, dim=1) == y_test).type(th.FloatTensor)
            acc = correct.mean()
            if epoch%10 == 0:
                rpc.rpc_async("master", print_acc, args=(epoch, acc))
            accuracy_list[epoch] = acc 
    
    return model

def add(x:th.Tensor,y:th.Tensor):
    print(x)
    print(y)
    print(x+y)
    return x + y