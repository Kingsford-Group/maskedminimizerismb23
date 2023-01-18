from utils.util import *
from config.std_config import *
from src.masked_minimizer import *

seed(2603, 2603)

w, k = 10, 15
seq = 'homopoly-100000'
res_dir = f'../artifact/exp8/{seq}/w{w}_k{k}_homopoly.pt'
res = torch.load(res_dir)['mask_performance']
res_ops, res_cms = [], []
for offset in range(w):
    res_ops.append(res[2 ** offset].item())
    res_cms.append(res[(2 ** w) - 1 - (2 ** offset)].item())


print(res_ops, res_cms)
print(res[2 ** w - 1].item())
SCALE = 1.5
fig, ax = plt.subplots(1, 1, figsize=(SCALE*1.5, SCALE*1.2), dpi=200)

ax.scatter(np.arange(w)[::-1] + 1, res_ops, marker='^', label='Syncmer')
ax.scatter(np.arange(w)[::-1] + 1, res_cms, marker='v', label='Complement')
ax.axhline(y=res[2 ** w - 1].item(), linestyle='--', label='Minimizer')
ax.set_xlabel('Offset')
ax.set_xticks(np.arange(w)+1)
ax.set_yticks([0.0, 0.2, 0.4, 0.6])
ax.set_ylabel('GSS')
# plt.tight_layout()
ax.legend(loc='lower right', fontsize=7)
fig.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.25)
fig.savefig(f'../artifact/exp8/{seq}/exp8.pdf')