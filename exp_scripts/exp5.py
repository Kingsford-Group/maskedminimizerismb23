# This experiment compares compatible
#   minimizer,
#   open-syncmer and
#   complement-syncmer
#   fix w, varying k, all trained with DeepMinimizer
from config.std_config import *
from src.masked_minimizer import *

seed(2603, 2603)
w, k = 7, 15
seq = f'L16'
try_mask = 'cms'
loss_wrapper = {
    # 'ept': EptLossWrapper,
    'den': DenLossWrapper,
    'con': ConLossWrapper,
    'hbd': HbdLossWrapper,
}
mask = {
    'ops': ops_mask(w, w//2),
    'mnz': mnz_mask(w),
    'cms': cms_mask(w, w//2)
}

for lw in loss_wrapper.keys():
    config = create_config(
        l=200, w=w, k=k, seq=seq, d=64, mask=mask[try_mask], n_mutations=3
    )
    config['loss_wrapper'] = loss_wrapper[lw]
    desc = f'w{w}_k{k}_{try_mask}_{lw}loss.pt'
    print(f'Experiment 5, w={w}, k={k}, mask={try_mask}, loss_wrapper={lw}')
    if not os.path.exists(f'../artifact/exp5/{seq}/'):
        os.makedirs(f'../artifact/exp5/{seq}/')
    config['save_path'] = f'../artifact/exp5/{seq}/{desc}'
    minimizer = MaskedMinimizer(config)
    minimizer.train_minimizer(
        n_epochs=301,
        eval_interval=25
    )