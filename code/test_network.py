import torch
import torch.nn as nn

PATH = "models/test_test_1.pt"

model = nn.Sequential(nn.Flatten(),
                      nn.Linear(2,2),
                      nn.ReLU(),
                      nn.Linear(2,2),
                      nn.ReLU(),
                      nn.Linear(2,2))

# set weights
with torch.no_grad():
    model[1].weight = nn.Parameter(torch.tensor([[1., 1.], [1., -1.]]))
    model[1].bias = nn.Parameter(torch.tensor([0., 0.]))

    model[3].weight = nn.Parameter(torch.tensor([[1., 1.], [1., -1.]]))
    model[3].bias = nn.Parameter(torch.tensor([-0.5, 0.]))

    model[5].weight = nn.Parameter(torch.tensor([[-1., 1.], [0., 1.]]))
    model[5].bias = nn.Parameter(torch.tensor([3., 0.]))


torch.save(model.state_dict(), PATH)