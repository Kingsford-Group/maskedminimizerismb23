# This experiment compares compatible
#   minimizer,
#   open-syncmer and
#   complement-syncmer
#   fix w, varying k, all trained with DeepMinimizer
from config.std_config import *
from src.masked_minimizer import *
from src.masked_minimizer_opt import *

seed(2603, 2603)
w, k = [10, 15, 20], [10, 15]
seq = f'chrXC'
loss_wrapper = {
    'hbd': MaskedHbdLossWrapper,
    'den': MaskedDenLossWrapper,
    'con': MaskedConLossWrapper,
}
for wval in w:
    for kval in k:
        for lw in loss_wrapper.keys():
            config = create_config(
                l=1500, w=wval, k=kval, seq=seq, d=64
            )
            config['loss_function'] = MaskedMaxpoolDelta(wval)
            config['loss_wrapper'] = loss_wrapper[lw]
            desc = f'w{wval}_k{kval}_{lw}loss_maskopt.pt'
            print(f'Experiment 10, w={wval}, k={kval}, lw={lw}')
            if not os.path.exists(f'../artifact/exp10/{seq}/'):
                os.makedirs(f'../artifact/exp10/{seq}/')
            config['save_path'] = f'../artifact/exp10/{seq}/{desc}'
            minimizer = MaskedMinimizerOpt(config)
            minimizer.train_minimizer(
                n_epochs=501,
                eval_interval=100
            )