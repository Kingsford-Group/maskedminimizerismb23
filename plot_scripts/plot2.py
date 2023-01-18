from utils.util import *
from pprint import pprint
from texttable import Texttable
import latextable

METRIC_DESC = {
    'g': 'Generalized Sketch Score',
    'd': 'Density',
    'v': 'Coverage',
    'c': 'Conservation',
}

def fig1(_w, _k, _mm, _artifact_root, _plot_dir, _seq, width=0.2, metric='c'):
    for wval in _w:
        for kval in _k:
            plt.figure()
            max_val = 0.0
            for offset, m in enumerate(_mm):
                res = torch.load(f'{_artifact_root}/{_seq}/{m}_w{wval}_k{kval}.pt')
                tally = {}

                for label in res.keys():
                    tally[label] = res[label][metric]
                    max_val = max(tally[label], max_val)
                x = np.arange(len(tally.keys()))
                plt.bar(x + (offset - 1) * width, tally.values(), width=width)
                plt.xticks(x, list(tally.keys()))
            plt.ylabel(METRIC_DESC[metric])
            plt.ylim([0.0, max_val * 1.4])
            plt.legend(_mm)
            plt.savefig(f'{_plot_dir}/w{wval}_k{kval}_{metric}.png')

def table1(_w, _k, _mm, _artifact_root, _plot_dir, _seq, metric='g'):
    table= Texttable()
    table.set_cols_align(['C'] * 13)
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_row([
        '$(w,k)$', '$\mathcal{M}$', '$\mathcal{O}_{w/2}$', '$\mathcal{C}_{w/2}$', '$\mathcal{O}_{w-1}$',
        '$\mathcal{M}$', '$\mathcal{O}_{w/2}$', '$\mathcal{C}_{w/2}$', '$\mathcal{O}_{w-1}$',
        '$\mathcal{M}$', '$\mathcal{O}_{w/2}$', '$\mathcal{C}_{w/2}$', '$\mathcal{O}_{w-1}$',
    ])
    for wval in _w:
        for kval in _k:
            row = [f'${wval},{kval}$']
            numerical_row = []
            for m in mm:
                desc = f'{m}_w{wval}_k{kval}.pt'
                res = torch.load(f'{_artifact_root}/{_seq}/{desc}')
                numerical_row.append(res['mnz'][metric].item())
                numerical_row.append(res[f'ops_{wval//2}'][metric])
                numerical_row.append(res[f'cms_{wval//2}'][metric])
                numerical_row.append(res[f'ops_{2}'][metric])
            numerical_row = np.array(numerical_row)
            best_col = np.argmax(numerical_row)
            for i, v in enumerate(numerical_row):
                if i != best_col:
                    row.append(f'${v:.3f}$')
                else:
                    row.append('$\mathbf{' + f'{v:.3f}' +'}$')
            table.add_row(row)
    print(latextable.draw_latex(table, caption="A table with position.", label="table:position", position='ht'))

w, k = [10, 15, 20], [10, 15]
seq = 'chrXC'
artifact_root = f'../artifact/exp2/'
plot_dir = f'{artifact_root}/{seq}/plot'
mm = ['miniception', 'pasha', 'random']
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# fig1(w,k,mm,artifact_root,plot_dir,seq,metric='c')
# fig1(w,k,mm,artifact_root,plot_dir,seq,metric='d')
# fig1(w,k,mm,artifact_root,plot_dir,seq,metric='v')
# fig1(w,k,mm,artifact_root,plot_dir,seq,metric='g')
table1(w,k,mm,artifact_root,plot_dir,seq,metric='g')
