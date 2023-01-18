from utils.util import *
from pprint import pprint
from texttable import Texttable
import latextable

l = [100, 200, 500, 999]
w, k = 10, 15
seq = 'random-1000'
artifact_root = f'../artifact/exp6/'
plot_dir = f'{artifact_root}/{seq}/plot'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

SCALE = 1
FONTSIZE=20
fig, ax = plt.subplots(1, 3, dpi=200)
fig.set_figheight(SCALE*4)
fig.set_figwidth(SCALE*19)
style = ['-x', '-v', '-^', '-o']
endpt=11
for i, lval in enumerate(l):
    st = style[i]
    res = torch.load(f'{artifact_root}{seq}/w{w}_k{k}_l{lval}.pt')
    rel_con = [
        1.0 - res['con'][i]/res['den'][i] if res['den'][i] > 0.0 else 0.0
        for i in range(len(res['con']))
    ]
    gss = np.array(rel_con) * np.array(res['cov'])
    ax[0].plot(res['epoch'][:endpt], rel_con[:endpt], st, label=f'$\ell={1000 if i==3 else lval}$')
    ax[1].plot(res['epoch'][:endpt], res['cov'][:endpt], st)
    ax[2].plot(res['epoch'][:endpt], gss[:endpt], st)

labels = ['Rel. Conservation','Coverage','GSS']
for i, a in enumerate(ax):
    a.set_ylabel(labels[i], fontsize=FONTSIZE)
    a.set_xticks([0, 200, 400, 600, 800, 1000, 1000])
    a.tick_params('x', labelsize=FONTSIZE-1)
    a.tick_params('y', labelsize=FONTSIZE-1)
fig.legend(ncol=4, loc='lower center', fontsize=FONTSIZE)
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.3, wspace=0.25)
fig.savefig(f'{plot_dir}/exploit.pdf')
