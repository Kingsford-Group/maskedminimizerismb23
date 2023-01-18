# This experiment compares compatible
#   minimizer,
#   open-syncmer and
#   complement-syncmer
#   fix w, varying k, all trained with DeepMinimizer
from config.std_config import *
from src.masked_minimizer import *

seed(2603, 2603)
w, k = 10, 15
seq = f'random-1000'
loss_wrapper = {
    'ept': EptLossWrapper,
    # 'den': DenLossWrapper,
    # 'con': ConLossWrapper,
    # 'hbd': HbdLossWrapper,
}
mask = {
    'ops': lambda t: ops_mask(w, t),
    # 'mnz': mnz_mask(w),
    # 'cms': cms_mask(w, w//2)
}
l = [
    # 100, 200, 500
    999
]
m = 'ops'
loss = 'ept'
for lval in l:
    config = create_config(
        l=lval, w=w, k=k, seq=seq, d=64, mask=mask[m], n_mutations=5
    )
    config['loss_wrapper'] = loss_wrapper[loss]
    desc = f'w{w}_k{k}_l{lval}.pt'
    print(f'Experiment 6, w={w}, k={k}, mask={m}, loss_wrapper={loss}')
    if not os.path.exists(f'../artifact/exp6/{seq}/'):
        os.makedirs(f'../artifact/exp6/{seq}/')
    config['save_path'] = f'../artifact/exp6/{seq}/{desc}'
    minimizer = MaskedMinimizer(config)
    minimizer.train_minimizer(
        n_epochs=2001,
        eval_interval=100
    )