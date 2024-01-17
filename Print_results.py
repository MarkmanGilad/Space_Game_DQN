import numpy as np
import torch
import matplotlib.pyplot as plt

Directory = 'Data'
Files_num = []
# Files_num = list(range(1,18+1))
results_path = []

for num in Files_num:
    file = f'results{num}.pth'
    results_path.append(file)
    
results_path = ["results.pth"]


results = []
for path in results_path:
    results.append(torch.load(Directory+'/'+path))

print("res every 100 epochs", results[0][0]) 
print("loses every 100 epochs",results[0][1]) 

# for i in range(len(results)):
    
#     print(results_path[i], max(results[i]['results']), np.argmax(results[i]['results']), len(results[i]['results']))
#     results[i]['avglosses'] = list(filter(lambda k:  0< k <100, results[i]['avglosses'] ))

with torch.no_grad():
    for i in range(len(results)):
        fig, ax_list = plt.subplots(2,1)
        fig.suptitle(results_path[i])
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        ax_list[0].plot(results[i][0])
        ax_list[0].title.set_text("Game score")
        ax_list[1].plot(results[i][1])
        ax_list[1].title.set_text("loss")
        plt.tight_layout()

plt.show()