from config.std_config import *
from src.masked_minimizer import *

seed(2603, 2603)
w, k = 10, 15
seq = 'L17'

config = create_config(
    l=1500, w=w, k=k, seq=seq, d=64, mask=mnz_mask(w), n_mutations=5
)
config['loss_wrapper'] = HbdLossWrapper

print(f'Experiment 7, w={w}, k={k}, seq={seq}')
desc = f'w{w}_k{k}_allmask.pt'
if not os.path.exists(f'../artifact/exp7/{seq}/'):
    os.makedirs(f'../artifact/exp7/{seq}/')
config['save_path'] = f'../artifact/exp7/{seq}/{desc}'
minimizer = MaskedMinimizer(config)
minimizer.train_minimizer(
    n_epochs=500,
    eval_interval=500
)
res = torch.load(config['save_path'])
bar = trange(2 ** w)
res['all_mask_performance'] = []
for t in bar:
    minimizer.mask = num_to_mask(w, t).to(device)
    _, _, _, gss = minimizer.eval_minimizer(bar, n_mutations=5)
    res['all_mask_performance'].append(gss)
torch.save(res, config['save_path'])