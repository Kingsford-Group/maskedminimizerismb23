from utils.util import *
from config.std_config import *
from src.masked_minimizer import *

seed(2603, 2603)

# Simulate sequence with lots of homopolymers
def generate_homopoly_seq(seg_per_line=10, segment_length=100,
                          homopoly_rate=0.2, num_segments=1000):
    with open(f'../seqdata/homopoly-{num_segments * segment_length}.seq', 'w') as f:
        st = ''
        for i in range(num_segments):
            if random.random() < homopoly_rate:
                st += random.choice(list(chmap.keys())) * segment_length
            else:
                st += random_sequence(segment_length)
            if (i + 1) % seg_per_line == 0:
                f.write(f'{st}\n')
                print(st)
                st = ''

w, k = 10, 15
seq = 'homopoly-100000'
config = create_config(
    l=1500, w=w, k=k, seq=seq, d=64, mask=mnz_mask(w), n_mutations=5
)
config['loss_wrapper'] = HbdLossWrapper
print(f'Experiment 8, w={w}, k={k}, seq={seq}')
desc = f'w{w}_k{k}_homopoly.pt'
if not os.path.exists(f'../artifact/exp8/{seq}/'):
    os.makedirs(f'../artifact/exp8/{seq}/')
config['save_path'] = f'../artifact/exp8/{seq}/{desc}'
minimizer = MaskedMinimizer(config)
minimizer.train_minimizer(
    n_epochs=500,
    eval_interval=500
)
res = torch.load(config['save_path'])
bar = trange(w)
_, _, _, gss = minimizer.eval_minimizer(bar)
res['mask_performance'] = {2 ** w - 1: gss}
for i in bar:
    # ops-i
    minimizer.mask = num_to_mask(w, 2 ** i).to(device)
    _, _, _, gss = minimizer.eval_minimizer(bar)
    res['mask_performance'][2 ** i] = gss
    # cms-i
    minimizer.mask = num_to_mask(w, (2 ** w) - 1 - (2 ** i)).to(device)
    _, _, _, gss = minimizer.eval_minimizer(bar)
    res['mask_performance'][(2 ** w) - 1 - (2 ** i)] = gss
torch.save(res, config['save_path'])