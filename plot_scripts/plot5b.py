from utils.util import *
from pprint import pprint
from texttable import Texttable
import latextable

MASK = {
    'mnz': 'Minimizers $(\mathcal{M})$',
    'ops': 'Syncmers $(\mathcal{O}_{w/2})$',
    'cms': 'Complement $(\mathcal{C}_{w/2})$'
}
LOSS = {
    'denloss': '$\mathcal{L}_{DM}$',
    'eptloss': '$\mathcal{L}_{ept}$',
    'conloss': '$\mathcal{L}_{con}$',
    'hbdloss': '$\mathcal{L}_{gss}$'
}
FONT_SIZE = 6
TICK_SIZE = 2.5
LINE_WIDTH = 0.8

def fig1(_ax, _w, _k, mask='mnz', loss='denloss', legend=False, mkfreq=2, title=False):
    res = torch.load(f'{artifact_root}{seq}/w{_w}_k{_k}_{mask}_{loss}.pt')
    _ax.plot(res['epoch'], res['den'], '-o',
             markevery=mkfreq, label='Density' if legend else '',
             markersize=TICK_SIZE, linewidth= LINE_WIDTH)
    _ax.plot(res['epoch'], res['con'], '-x',
             markevery=mkfreq, label='Conservation'  if legend else '',
             markersize=TICK_SIZE, linewidth= LINE_WIDTH)
    _ax.plot(res['epoch'], res['cov'], '-s',
             markevery=mkfreq, label='Coverage'  if legend else '',
             markersize=TICK_SIZE, linewidth= LINE_WIDTH)
    _ax.plot(res['epoch'], res['gss'], '-^',
             markevery=mkfreq, label='GSS' if legend else '',
             markersize=TICK_SIZE, linewidth= LINE_WIDTH)
    relcon = np.array(res['con'])/np.array(res['den'])
    _ax.plot(res['epoch'], relcon, '-v',
             markevery=mkfreq, label='Rel. Conservation' if legend else '',
             markersize=TICK_SIZE, linewidth= LINE_WIDTH)
    _ax.set_xticks([0, 150, 300])
    _ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _ax.tick_params('x', labelsize=FONT_SIZE)
    _ax.tick_params('y', labelsize=FONT_SIZE)
    _ax.set_xlabel('Epoch', fontsize=FONT_SIZE)
    if title:
        _ax.set_title(f'{MASK[mask]}\n{LOSS[loss]}', fontsize=FONT_SIZE + 1.5)
    else:
        _ax.set_title(f'\n{LOSS[loss]}', fontsize=FONT_SIZE + 1.5)

w, k = 7, 15
seq = 'L16'
artifact_root = f'../artifact/exp5/'
plot_dir = f'{artifact_root}/{seq}/plot'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


SCALE=1.4

fig, ax = plt.subplots(1, 9, sharey=True, dpi=400, figsize=(SCALE*4.5, SCALE * 1.1))
for i, mask in enumerate(MASK):
    fig1(ax[3 * i + 0], w, k, mask, 'denloss')
    fig1(ax[3 * i + 1], w, k, mask, 'conloss', title=True)
    fig1(ax[3 * i + 2], w, k, mask, 'hbdloss', legend=(i==2))
ax[0].set_ylabel('Metric', fontsize=FONT_SIZE + 1)
fig.legend(ncol=5, loc='lower center', fontsize=FONT_SIZE)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.35, top=0.8, wspace=0.38)
fig.savefig(f'{plot_dir}/w{w}_k{k}.png')
fig.savefig(f'{plot_dir}/w{w}_k{k}.pdf')

