# %%
import torch
from loss_torch import loss, plot_diff
from read_scale import load_image
import matplotlib.pyplot as plt

# %%
N = 32

matrix_param = torch.nn.Parameter(torch.randn((N, N), requires_grad=True, ))  # Random initialization
optimizer = torch.optim.Adam([matrix_param], lr=0.1)

img = torch.tensor(load_image("et.png", square=True), dtype=torch.float32)

for epoch in range(500):
    optimizer.zero_grad()
    
    # Ensuring symmetry and bounding the values between 0 and 1
    matrix = matrix_param.triu(0) + matrix_param.triu(1).T
    # torch.sigmoid_(matrix)

    loss_, occlusion = loss(matrix, img)

    loss_.backward()

    optimizer.step()

    print(f"Epoch {epoch}: {loss_}")
    plot_diff(img, occlusion)


# %%
