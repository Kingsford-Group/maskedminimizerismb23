from utils.util import *
from pprint import pprint
from texttable import Texttable
import latextable

MASK = {
    'mnz': 'Minimizer',
    'ops': 'Syncmer',
    'cms': 'Complement'
}
LOSS = {
    'denloss': '$\mathcal{L}_{DM}$',
    'eptloss': '$\mathcal{L}_{ept}$',
    'conloss': '$\mathcal{L}_{con}$',
    'hbdloss': '$\mathcal{L}_{gss}$'
}
FONT_SIZE = 16
def fig1(_ax, _w, _k, mask='mnz', loss='denloss', legend=False, mkfreq=2):
    res = torch.load(f'{artifact_root}{seq}/w{_w}_k{_k}_{mask}_{loss}.pt')
    _ax.plot(res['epoch'], res['den'], '-o', markevery=mkfreq, label='Density' if legend else '')
    _ax.plot(res['epoch'], res['con'], '-x', markevery=mkfreq, label='Conservation'  if legend else '')
    # _ax.plot(res['epoch'], res['cov'], '-o', markevery=mkfreq, label='Coverage'  if legend else '')
    _ax.plot(res['epoch'], res['gss'], '-^', markevery=mkfreq, label='GSS' if legend else '')
    relcon = np.array(res['con'])/np.array(res['den'])
    _ax.plot(res['epoch'], relcon, '-v', markevery=mkfreq, label='Rel. Conservation' if legend else '')
    _ax.set_xticks([0, 100, 200, 300])
    _ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _ax.tick_params('x', labelsize=FONT_SIZE-2)
    _ax.tick_params('y', labelsize=FONT_SIZE-2)
    _ax.set_xlabel('Epochs', fontsize=FONT_SIZE)
    _ax.set_title(f'{MASK[mask]}\n{LOSS[loss]}', fontsize=FONT_SIZE + 1)


w, k = 7, 15
seq = 'L16'
artifact_root = f'../artifact/exp5/'
plot_dir = f'{artifact_root}/{seq}/plot'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


SCALE=1.4
for mask in MASK:
    fig, ax = plt.subplots(1, 3, sharey=True, dpi=200)
    fig.set_figheight(SCALE*3)
    fig.set_figwidth(SCALE*4)
    fig1(ax[0], w, k, mask, 'denloss')
    fig1(ax[1], w, k, mask, 'conloss')
    fig1(ax[2], w, k, mask, 'hbdloss', legend=True)
    fig.legend(ncol=2, loc='lower center', fontsize=FONT_SIZE)
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.35, top=0.85)
    fig.savefig(f'{plot_dir}/w{w}_k{k}_{mask}.pdf')

