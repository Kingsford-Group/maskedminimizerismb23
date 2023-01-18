from config.std_config import *
from src.masked_minimizer import *

seed(2603, 2603)
w, k = 10, 15
seq = ['L16', 'L11', 'L14']
title = ['BTR1', 'BTR2', 'BTR3']
exp_dir = f'../artifact/exp7/'
SCALE = 1.2
fig, ax = plt.subplots(1, 3, sharey=True, figsize=(SCALE*7, SCALE*1.2), dpi=200)
LABEL_FONT_SIZE = 10
for i, s in enumerate(seq):
    res_dir = f'{exp_dir}{s}/w{w}_k{k}_allmask.pt'
    res = torch.load(res_dir)['all_mask_performance']
    val, numone = [], []
    for j, r in enumerate(res):
        val.append(r.item())
        numone.append(bin(j).count('1'))
    ax[i].scatter(numone[1:], val[1:], s=3, marker='x')
    ax[i].set_title(title[i], x=0.95, y=0.1, loc='right', fontsize=LABEL_FONT_SIZE)
    if i == 0:
        ax[i].set_ylabel('GSS', fontsize=LABEL_FONT_SIZE)
    ax[i].set_yticks([0.0, 0.25, 0.50, 0.75])
    ax[i].set_xlabel('No. 1-entries', fontsize=LABEL_FONT_SIZE)
    ax[i].set_xticks(np.arange(w) + 1)


fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.3, wspace=0.2)
fig.savefig(f'{exp_dir}exp7b.pdf')

