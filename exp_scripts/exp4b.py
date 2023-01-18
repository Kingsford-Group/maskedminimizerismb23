# This experiment compares compatible
#   minimizer,
#   open-syncmer and
#   complement-syncmer
#   fix w, varying k, all trained with DeepMinimizer
from config.std_config import *
from src.masked_minimizer import *

seed(2603, 2603)
w, k = [10, 15, 20], [10, 15]
seq = f'chrXC'
loss_wrapper = {
    'hbd': HbdLossWrapper,
    'den': DenLossWrapper,
    'con': ConLossWrapper,
}
for wval in w:
    for kval in k:
        for lw in loss_wrapper.keys():
            config = create_config(
                l=1500, w=wval, k=kval, seq=seq, d=64, mask=None
            )
            config['loss_wrapper'] = loss_wrapper[lw]
            desc = f'w{wval}_k{kval}_opt_{lw}loss.pt'
            print(f'Experiment 4b, w={wval}, k={kval}, mask=opt, loss_wrapper={lw}')
            if not os.path.exists(f'../artifact/exp4/{seq}/'):
                os.makedirs(f'../artifact/exp4/{seq}/')
            config['save_path'] = f'../artifact/exp4/{seq}/{desc}'
            minimizer = MaskedMinimizer(config)
            minimizer.train_minimizer(
                n_epochs=501,
                eval_interval=100
            )