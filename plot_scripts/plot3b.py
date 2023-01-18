from utils.util import *

METRIC_DESC = {
    'best_gss': 'Best GSS',
    'best_den': 'Best Density',
    'best_cov': 'Best Coverage',
    'best_con': 'Best Conservation',
    'gss': 'GSS',
    'den': 'Density',
    'cov': 'Coverage',
    'con': 'Conservation',
}
MASK = ['mnz', 'ops', 'cms']
MASK_DESC = {
    'mnz': 'Minimizer $(\mathcal{M}$)',
    'ops': 'Syncmer $(\mathcal{O}_{w/2}$)',
    'cms': 'Comp. ($\mathcal{C}_{w/2}$)'
}
SCALE = 1.4
FONTSIZE = 18
def fig2(_artifact_root, _plot_dir, _seq, _w, _k, metric='best_gss', _marker=1):
    style = ['-x', '-v', '-^']
    fig, ax = plt.subplots(1, 4, sharey=True, dpi=200)
    fig.set_figheight(SCALE*3)
    fig.set_figwidth(SCALE*6)
    for j, wval in enumerate(_w):
        for i, m in enumerate(MASK):
            fn = f'w{wval}_k{_k}_{m}.pt'
            res = torch.load(f'{_artifact_root}/{_seq}/{fn}')
            ax[j].plot(res['epoch'], res[metric], style[i], markevery=_marker,
                       linewidth=2, label=MASK_DESC[m] if j==len(_w)-1 else '')
        ax[j].set_xlabel('Epoch', fontsize=FONTSIZE)
        ax[j].set_xticks([0, 200, 400, 600])
        ax[j].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax[j].tick_params('x', labelsize=FONTSIZE-1)
        ax[j].tick_params('y', labelsize=FONTSIZE-1)
        if j==0:
            ax[j].set_ylabel(METRIC_DESC[metric], fontsize=FONTSIZE)
        ax[j].set_title(f'w={wval}', fontsize=FONTSIZE)
    fig.legend(ncol=3, loc='lower center', fontsize=FONTSIZE-1)
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.3)
    fig.savefig(f'{_plot_dir}/compare_{metric}_vs_epoch_{_seq}.pdf')

artifact_root = '../artifact/exp3'
w, k = [25, 40, 55, 70], 15

seq = 'chr1'
marker = 1
plot_dir = f'{artifact_root}/{seq}/plot/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
fig2(artifact_root, plot_dir, seq, w, k, 'best_gss', _marker=marker)
seq = 'chrXC'

marker = [0, 10, 20, 30, 40, 50, 59]
plot_dir = f'{artifact_root}/{seq}/plot/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
fig2(artifact_root, plot_dir, seq, w, k, 'best_gss', _marker=marker)
