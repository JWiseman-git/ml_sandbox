import torch
from model_builder import model_builder


device = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(42)
model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)