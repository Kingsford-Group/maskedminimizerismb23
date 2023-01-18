# This experiment compares compatible
#   minimizer,
#   open-syncmer and
#   complement-syncmer
#   fix w, varying k, all trained with DeepMinimizer
from config.std_config import *
from src.masked_minimizer import *

seed(2603, 2603)
w, k = 15, [10, 25, 40, 55, 70]
seq = f'chr1'
mask = {
    'ops': ops_mask(w, w//2),
    'mnz': mnz_mask(w),
    'cms': cms_mask(w, w//2)
}
try_mask = 'cms'
for kval in k:
    config = create_config(
        l=1500, w=w, k=kval, seq=seq, d=64, mask=mask[try_mask]
    )
    desc = f'w{w}_k{kval}_{try_mask}.pt'
    print(f'Experiment 1, w={w}, k={kval}, mask={try_mask}')
    if not os.path.exists(f'../artifact/exp1/{seq}/'):
        os.makedirs(f'../artifact/exp1/{seq}/')
    config['save_path'] = f'../artifact/exp1/{seq}/{desc}'
    minimizer = MaskedMinimizer(config)
    minimizer.train_minimizer(
        n_epochs=600,
        eval_interval=10
    )