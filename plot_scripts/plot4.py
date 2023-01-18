from utils.util import *
from pprint import pprint
from texttable import Texttable
import latextable


METRIC_DESC = {
    'best_gss': 'Best Generalized Sketch Score',
    'best_den': 'Best Density',
    'best_cov': 'Best Coverage',
    'best_con': 'Best Conservation',
    'gss': 'Generalized Sketch Score',
    'den': 'Density',
    'cov': 'Coverage',
    'con': 'Conservation',
}

MASK = ['mnz', 'ops', 'cms', 'opt']
LFN = ['conloss', 'denloss', 'hbdloss']
LFNAME = {
    'conloss': 'Conservation Loss',
    'denloss': 'DeepMinimizer Loss',
    'hbdloss': 'Combined Loss'
}

def fig1(_artifact_root, _plot_dir, _seq, _w, _k, width=0.2, metric='best_gss'):
    for wval in _w:
        for kval in _k:
            plt.figure()
            max_val = 0.0
            for offset, lf in enumerate(LFN):
                tally = {}
                for m in MASK:
                    desc = f'w{wval}_k{kval}_{m}_{lf}.pt'
                    res = torch.load(f'{_artifact_root}/{_seq}/{desc}')
                    tally[m] = res[metric][-1]
                    max_val = max(max_val, tally[m])
                x = np.arange(len(tally.keys()))
                plt.bar(x + (offset - 1) * width, tally.values(), width=width)
                plt.xticks(x, list(tally.keys()))
            plt.ylabel(METRIC_DESC[metric])
            plt.ylim([0.0, max_val * 1.4])
            plt.legend(LFNAME.values())
            plt.savefig(f'{_plot_dir}/w{wval}_k{kval}_{metric}.png')

def table1(_artifact_root, _plot_dir, _seq, _w, _k, metric='best_gss', show_init=True):
    table= Texttable()
    table.set_cols_align(['C'] * (1 + len(LFN) * len(MASK)))
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_row([
        '$(w,k)$', '$\mathcal{M}$', '$\mathcal{O}_{w/2}$', '$\mathcal{C}_{w/2}$', '$\mathcal{V}$',
        '$\mathcal{M}$', '$\mathcal{O}_{w/2}$', '$\mathcal{C}_{w/2}$', '$\mathcal{V}$',
        '$\mathcal{M}$', '$\mathcal{O}_{w/2}$', '$\mathcal{C}_{w/2}$', '$\mathcal{V}$',
        # '$\mathcal{M}$', '$\mathcal{O}_{w/2}$', '$\mathcal{C}_{w/2}$', '$\mathcal{V}$'
    ])
    for wval in _w:
        for kval in _k:
            row = [f'${wval},{kval}$']
            numerical_row = []
            random_avg = {m: 0.0 for m in MASK}
            for lf in LFN:
                for m in MASK:
                    desc = f'w{wval}_k{kval}_{m}_{lf}.pt'
                    res = torch.load(f'{_artifact_root}/{_seq}/{desc}')
                    numerical_row.append(res[metric][-1])
                    if m == 'opt':
                        numerical_row[-1] = max(max(numerical_row[-2:]), numerical_row[-4])
                    random_avg[m] += res[metric][0] / len(MASK)
            if show_init:
                for m in MASK:
                    numerical_row.append(random_avg[m])
            numerical_row = np.array(numerical_row)
            best_col = np.argmax(numerical_row)
            for i, v in enumerate(numerical_row):
                if i != best_col:
                    row.append(f'${v:.3f}$')
                else:
                    row.append('$\mathbf{' + f'{v:.3f}' +'}$')
            # print(row)
            table.add_row(row)
    print(latextable.draw_latex(table, caption="A table with position.", label="table:position", position='ht'))

w, k = [10, 15, 20], [10, 15]
seq = 'chrXC'
artifact_root = f'../artifact/exp4/'
plot_dir = f'{artifact_root}/{seq}/plot'


if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# fig1(artifact_root, plot_dir, seq, w, k, metric='best_gss')
# fig1(artifact_root, plot_dir, seq, w, k, metric='best_con')
# fig1(artifact_root, plot_dir, seq, w, k, metric='best_den')
# fig1(artifact_root, plot_dir, seq, w, k, metric='best_cov')

table1(artifact_root, plot_dir, seq, w, k, metric='best_gss', show_init=False)