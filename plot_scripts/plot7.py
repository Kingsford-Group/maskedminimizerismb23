from config.std_config import *
from src.masked_minimizer import *

seed(2603, 2603)
w, k = 10, 15
seq = ['L16', 'L11', 'L14', 'L17']
title = ['BTR1', 'BTR2', 'BTR3', 'BTR4']
exp_dir = f'../artifact/exp7/'

SCALE = 1.5
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True,
                       figsize=(SCALE*3, SCALE*1.2),dpi=200)

for i, s in enumerate(seq):
    res_dir = f'{exp_dir}{s}/w{w}_k{k}_allmask.pt'
    res = torch.load(res_dir)['all_mask_performance']
    val, numone = [], []
    for j, r in enumerate(res):
        val.append(r.item())
        numone.append(bin(j).count('1'))
    ax[i // 2][i % 2].scatter(numone, val, s=3, marker='x')
    ax[i // 2][i % 2].set_title(title[i], x=0.95, y=0.1, loc='right', fontsize=9)
    if i % 2 == 0:
        ax[i // 2][i % 2].set_ylabel('GSS', fontsize=8)
        ax[i // 2][i % 2].set_yticks([0.0, 0.25, 0.50, 0.75])

    if i // 2 == 1:
        ax[i // 2][i % 2].set_xlabel('No. 1-entries', fontsize=8)
        ax[i // 2][i % 2].set_xticks([0, 2, 4, 6, 8, 10])


fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.25, hspace=0.2)
fig.savefig(f'{exp_dir}exp7.pdf')

