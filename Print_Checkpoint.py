import numpy as np
import torch
import matplotlib.pyplot as plt

Files_num = [1]
# Files_num = list(range(1,6))

results_path = []
for num in Files_num:
    file = f'Data/checkpoint{num}.pth'
    results_path.append(file)

checkpoints = []
for path in results_path:
    checkpoints.append(torch.load(path))


for i in range(len(checkpoints)):
    fig, ax_list = plt.subplots(2,1, figsize = (10,4))
    fig.suptitle(f'{results_path[i]} epochs: {checkpoints[i]["epoch"]}')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax_list[0].plot(checkpoints[i]['scores'])
    ax_list[0].title.set_text("Game score")
    ax_list[1].plot(checkpoints[i]['loss'])
    ax_list[1].title.set_text("loss")
    plt.tight_layout()

plt.show()