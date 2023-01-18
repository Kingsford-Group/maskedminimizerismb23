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
    'mnz': 'Minimizer',
    'ops': 'Syncmer',
    'cms': 'Complement'
}
SCALE = 1.4
FONTSIZE = 18
def fig2(_artifact_root, _plot_dir, _seq, _w, _k, metric='best_gss'):
    style = ['-x', '-v', '-^']
    fig, ax = plt.subplots(1, 4, sharey=True, dpi=200)
    fig.set_figheight(SCALE*3)
    fig.set_figwidth(SCALE*6)
    for j, kval in enumerate(_k):
        for i, m in enumerate(MASK):
            fn = f'w{_w}_k{kval}_{m}.pt'
            res = torch.load(f'{_artifact_root}/{_seq}/{fn}')
            marker = [i for i in range(len(res['epoch']))
                      if (i % 10 == 0) or (i == (len(res['epoch']) -1))]
            ax[j].plot(res['epoch'], res[metric], style[i], markevery=marker,
                       linewidth=2, label=MASK_DESC[m] if j==len(_k)-1 else '')
        ax[j].set_xlabel('Epoch', fontsize=FONTSIZE)
        ax[j].set_xticks([0, 200, 400, 600])
        ax[j].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax[j].tick_params('x', labelsize=FONTSIZE-1)
        ax[j].tick_params('y', labelsize=FONTSIZE-1)
        if j==0:
            ax[j].set_ylabel(METRIC_DESC[metric], fontsize=FONTSIZE)
        ax[j].set_title(f'k={kval}', fontsize=FONTSIZE+1)
    #fig.legend(ncol=3, loc='lower center', fontsize=FONTSIZE)
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.3)
    fig.savefig(f'{_plot_dir}/compare_{metric}_vs_epoch_{seq}.pdf')
    fig.savefig(f'{_plot_dir}/compare_{metric}_vs_epoch_{seq}.png')

artifact_root = '../artifact/exp1'
seq = 'chrXC'
w, k = 15, [25, 40, 55, 70]
plot_dir = f'{artifact_root}/{seq}/plot/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

fig2(artifact_root, plot_dir, seq, w, k, 'best_gss')