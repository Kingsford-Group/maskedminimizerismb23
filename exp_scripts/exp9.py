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
    'ept_v2': EptLossWrapper_v2,
    'ept': EptLossWrapper,
    # 'den': DenLossWrapper,
    # 'con': ConLossWrapper,
    # 'hbd': HbdLossWrapper,
}
mask = {
    'ops': lambda o: ops_mask(w, o),
    # 'mnz': mnz_mask(w),
    # 'cms': cms_mask(w, w//2)
}
l = 200
nm = [1, 5, 10, 20]
t=9
#t = 3,5,7,9
mask = mask['ops'](t)
loss = 'ept'

for nmval in nm:
    config = create_config(
        l=l, w=w, k=k, seq=seq, d=64, mask=mask, n_mutations=nmval
    )
    config['loss_wrapper'] = loss_wrapper[loss]
    config['batch_size'] = int(1000 / l)
    desc = f'w{w}_k{k}_t{t}_nm{nmval}.pt'
    print(f'Experiment 9, w={w}, k={k}, offset={t}, num_mutations={nmval}')
    if not os.path.exists(f'../artifact/exp9/{seq}/'):
        os.makedirs(f'../artifact/exp9/{seq}/')
    config['save_path'] = f'../artifact/exp9/{seq}/{desc}'
    minimizer = MaskedMinimizer(config)
    minimizer.train_minimizer(
        n_epochs=2001,
        eval_interval=100,
        save_model=True
    )
    seq_data = torch.load(f'../artifact/{seq}.dat')
    sequence = F.one_hot(seq_data.seq.squeeze(1), len(chmap.keys())).transpose(1, 2).float()
    minimizer.history['score_curve'] = minimizer.priority_net(sequence.to(device))
    torch.save(minimizer.history, config['save_path'])