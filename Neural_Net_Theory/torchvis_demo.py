import torch
import torch.nn as nn
import torchvision.models as models
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter


# 1. Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 2. Visualize SimpleNet with torchviz
simple_model = SimpleNet()
simple_input = torch.randn(1, 10)
simple_output = simple_model(simple_input)

# Generate and save the computation graph
# dot = make_dot(simple_output, params=dict(simple_model.named_parameters()))
# dot.render("simple_net", format="png")

# 3. Visualize ResNet18 with TensorBoard
resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Create a dummy input with the expected shape (batch size, channels, height, width)
resnet_input = torch.randn(1, 3, 224, 224)

# Log the graph
writer = SummaryWriter('runs/resnet18_experiment_1')
writer.add_graph(resnet_model, resnet_input)
writer.close()

