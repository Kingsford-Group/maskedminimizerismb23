from utils.util import *
from config.std_config import *
from src.masked_minimizer import *

seed(2603, 2603)
SCALE = 1.3
fig, ax = plt.subplots(1, 2, figsize=(SCALE*5.1, SCALE*1.2), dpi=200)
LABEL_FONT_SIZE = 10

# right plot
w, k = 10, 15
seq = 'homopoly-100000'
res_dir = f'../artifact/exp8/{seq}/w{w}_k{k}_homopoly.pt'
res = torch.load(res_dir)['mask_performance']
res_ops, res_cms = [], []
for offset in range(w):
    res_ops.append(res[2 ** offset].item())
    res_cms.append(res[(2 ** w) - 1 - (2 ** offset)].item())

ax[1].scatter(np.arange(w)[::-1] + 1, res_ops, marker='^', label='Syncmer')
ax[1].scatter(np.arange(w)[::-1] + 1, res_cms, marker='v', label='Complement')
ax[1].axhline(y=res[2 ** w - 1].item(), linestyle='--', label='Minimizer')
ax[1].set_xlabel('Offset', fontsize=LABEL_FONT_SIZE)
ax[1].set_xticks(np.arange(w)+1)
ax[1].set_yticks([0.0, 0.2, 0.4, 0.6])
ax[1].set_ylabel('GSS', fontsize=LABEL_FONT_SIZE)
ax[1].legend(loc='lower right', fontsize=7)

# left plot
seq = 'L17'
res_dir = f'../artifact/exp7/L17/w{w}_k{k}_allmask.pt'
res = torch.load(res_dir)['all_mask_performance']
val, numone = [], []
for j, r in enumerate(res):
    val.append(r.item())
    numone.append(bin(j).count('1'))
ax[0].scatter(numone[1:], val[1:], s=3, marker='x')
ax[0].set_ylabel('GSS', fontsize=LABEL_FONT_SIZE)
ax[0].set_yticks([0.0, 0.25, 0.50, 0.75])
ax[0].set_xlabel('No. 1-entries', fontsize=LABEL_FONT_SIZE)
ax[0].set_xticks(np.arange(w)+1)

# savefig
fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.3, wspace=0.3)
fig.savefig(f'../artifact/exp8/exp78.pdf')