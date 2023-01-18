# This experiment compares compatible
#   minimizer,
#   open-syncmer and
#   complement-syncmer
#   fix k, varying w, all trained with DeepMinimizer
from config.std_config import *
from src.masked_minimizer import *

seed(2603, 2603)
w, k = [25, 40, 55, 70], 15
seq = f'chr1'
for wval in w:
    mask = {
        'ops': ops_mask(wval, wval//2),
        'mnz': mnz_mask(wval),
        'cms': cms_mask(wval, wval//2)
    }
    try_mask = 'mnz'
    config = create_config(
        l=1500, w=wval, k=k, seq=seq, d=64, mask=mask[try_mask]
    )
    desc = f'w{wval}_k{k}_{try_mask}.pt'
    print(f'Experiment 3, w={wval}, k={k}, mask={try_mask}')
    if not os.path.exists(f'../artifact/exp3/{seq}/'):
        os.makedirs(f'../artifact/exp3/{seq}/')
    config['save_path'] = f'../artifact/exp3/{seq}/{desc}'
    minimizer = MaskedMinimizer(config)
    minimizer.train_minimizer(
        n_epochs=601,
        eval_interval=100
    )