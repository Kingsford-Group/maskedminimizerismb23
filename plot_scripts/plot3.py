from utils.util import *

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
def fig1(_mask, _artifact_root, _plot_dir, _seq, _w, _k, metric='best_gss'):
    """
    Compare best_gss of three masks by metric:
        mnz: minimizer
        ops: open syncmer with offset t=w//2
        cms: complement syncmer with offset t=w//2
    """
    plt.figure()
    style = ['-x', '-v', '-^']
    for i, m in enumerate(_mask):
        best = []
        for wval in _w:
            fn = f'w{wval}_k{_k}_{m}.pt'
            res = torch.load(f'{_artifact_root}/{_seq}/{fn}')
            best.append(res[metric][-1])
        plt.plot(_w, best, style[i], linewidth=2, label=m)
    plt.xlabel('window size')
    plt.ylabel(METRIC_DESC[metric])
    plt.legend()
    plt.title(f'Comparing {METRIC_DESC[metric]} on {_seq}, k={k}')
    plt.savefig(f'{_plot_dir}/compare_{metric}_vary_w.png')

def fig2(_mask, _artifact_root, _plot_dir, _seq, _w, _k, metric='best_gss'):
    style = ['-x', '-v', '-^']
    for wval in _w:
        plt.figure()
        for i, m in enumerate(_mask):
            fn = f'w{wval}_k{_k}_{m}.pt'
            res = torch.load(f'{_artifact_root}/{_seq}/{fn}')
            plt.plot(res['epoch'], res[metric], style[i], markevery=5, linewidth=2, label=m)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(METRIC_DESC[metric])
        plt.title(f'{METRIC_DESC[metric]} vs. epoch on {_seq}, w={wval}, k={_k}')
        plt.savefig(f'{_plot_dir}/compare_{metric}_vs_epoch_w{wval}.png')
artifact_root = '../artifact/exp3'
seq = 'chr1'
w, k = [10, 25, 40, 55, 70], 15
mask = ['mnz', 'ops', 'cms']
plot_dir = f'{artifact_root}/{seq}/plot/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

fig1(mask, artifact_root, plot_dir, seq, w, k, 'best_gss')
fig1(mask, artifact_root, plot_dir, seq, w, k, 'best_cov')
fig1(mask, artifact_root, plot_dir, seq, w, k, 'best_con')
fig1(mask, artifact_root, plot_dir, seq, w, k, 'best_den')
fig2(mask, artifact_root, plot_dir, seq, w, k, 'best_gss')
fig2(mask, artifact_root, plot_dir, seq, w, k, 'best_cov')
fig2(mask, artifact_root, plot_dir, seq, w, k, 'best_con')
fig2(mask, artifact_root, plot_dir, seq, w, k, 'best_den')