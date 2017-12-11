import torch 
import torch.nn as nn 
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 2)
        # self.fc3 = nn.Linear(dims[2], dims[3])

    def forward(self, x):
        x = x.view(-1, 4)
        y = F.relu(self.fc1(x))
        # y = F.relu(self.fc2(y))
        y = self.fc2(y)
        return y


class MLP_Dueling(nn.Module):
    def __init__(self, ob_space, ac_space, hiddens):
        super(MLP_Dueling, self).__init__()
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.hiddens = hiddens 
        mlps = []
        
        input_shape = ob_space
        for h in hiddens:
            mlps.append(nn.Linear(in_features=input_shape, out_features=h))
            mlps.append(nn.ReLU(inplace=True))
            input_shape = h

        self.feats = nn.Sequential(*mlps)
        self.value = nn.Linear(in_features=input_shape, out_features=1)
        self.advantage = nn.Linear(in_features=input_shape, out_features=ac_space)


    def forward(self, x):
        x = x.view(-1, self.ob_space)
        x = self.feats(x)

        value = self.value(x)
        advantage = self.advantage(x)

        q = value.expand_as(advantage) + (advantage - advantage.mean(1, keepdim=True))
        return q


class MLP_PLUS(nn.Module):
    def __init__(self, ob_space, ac_space, hiddens):
        super(MLP_PLUS, self).__init__()
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.hiddens = hiddens 
        mlps = []
        
        input_shape = ob_space
        for h in hiddens:
            mlps.append(nn.Linear(in_features=input_shape, out_features=h))
            mlps.append(nn.ReLU(inplace=True))
            input_shape = h

        self.feats = nn.Sequential(*mlps)
        self.q = nn.Linear(in_features=input_shape, out_features=ac_space)
    
    def forward(self, x):
        x = x.view(-1, self.ob_space)
        x = self.feats(x)
        out = self.q(x)
        return out
        

class MLP_FAILED(nn.Module):
    '''
    -> This fails -> The list doesn't register the parameters
    -> Needs a different way to do this
    '''
    def __init__(self, ob_space, ac_space, hiddens):
        super(MLP, self).__init__()
        self.hiddens = hiddens 
        self.mlps = []
        
        input_shape = ob_space
        for h in hiddens:
            self.mlps.append(nn.Linear(in_features=input_shape, out_features=h))
            input_shape = h
        
        self.q = nn.Linear(in_features=input_shape, out_features=ac_space)
        self.train()


    def forward(self, x):
        for linear in self.mlps:
            x = F.relu(linear(x))

        q_out = self.q(x)
        return q_out

