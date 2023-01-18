from utils.util import *
from config.std_config import *
from collections import defaultdict

nm = [1, 5, 10, 20]
w, k = 10, 15
t=6
seq = 'random-1000'
artifact_root = f'../artifact/exp9/'
plot_dir = f'{artifact_root}/{seq}/plot'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Count consecutive runs
res = torch.load(f'{artifact_root}{seq}/w{w}_k{k}_t{t}_nm20.pt')
exploit_itr = -1
for itr, cov in enumerate(res['cov']):
    if cov == 0.0:
        exploit_itr = itr
        break
config = create_config(
    l=200, w=w, k=k, seq=seq, d=64, mask=ops_mask(w, t), n_mutations=20)
config['priority_net'].load_state_dict(res['model'][exploit_itr])
config['priority_net'].to(device)
seq_data = torch.load(f'../artifact/{seq}.dat')
sequence = F.one_hot(
    seq_data.seq.squeeze(1), len(chmap.keys())
).transpose(1, 2).float().to(device)
score = config['priority_net'](sequence).squeeze()
score = (score - torch.min(score)) / torch.max(score)
score = np.array((1.0 - score).tolist())
change = [score[i] - score[i-1] for i in range(1, score.shape[0])]
current_run = 1
current_sign = change[0] >= 0
pos = defaultdict(int)
neg = defaultdict(int)
for i in range(1, len(change)):
    c = change[i]
    if (c >= 0) == current_sign:
        current_run += 1
    else:
        if current_sign:
            if current_run >= t:
                pos[f'>{t-1}'] += 1
            else:
                pos[current_run] += 1
        else:
            if current_run >= t:
                neg[f'>{t-1}'] += 1
            else:
                neg[current_run] += 1
        current_run = 1
        current_sign = (c >= 0)

SCALE = 1
FONTSIZE=7
TICKSIZE=3
LINEWIDTH=1
fig, ax = plt.subplots(1, 3, dpi=200, figsize=(SCALE*6, SCALE*1.3))
style = ['-x', '-v', '-^', '-o']
endpt=exploit_itr+1
width=0.2
for i, nmval in enumerate(nm):
    st = style[i]
    res = torch.load(f'{artifact_root}{seq}/w{w}_k{k}_t{t}_nm{nmval}.pt')
    rel_con = [
        1.0 - res['con'][i]/res['den'][i] if res['den'][i] > 0.0 else 0.0
        for i in range(len(res['con']))
    ]
    gss = np.array(rel_con) * np.array(res['cov'])
    ax[0].plot(res['epoch'][:endpt], rel_con[:endpt], st,
               label=f'$n={nmval}$', markersize=TICKSIZE,
               linewidth=LINEWIDTH)
    ax[1].plot(res['epoch'][:endpt], res['cov'][:endpt], st,
               label=f'$n={nmval}$', markersize=TICKSIZE,
               linewidth=LINEWIDTH)
    # ax[2].plot(res['epoch'][:endpt], gss[:endpt], st)
pos_bar = [pos[i] for i in range(1, t)] + [pos[f'>{t-1}']]
neg_bar = [neg[i] for i in range(1, t)] + [neg[f'>{t-1}']]
ax[2].bar(np.arange(t)+1-width, pos_bar, width * 2, label='Mono. Increase')
ax[2].bar(np.arange(t)+1+width, neg_bar, width * 2, label='Mono. Decrease')
# ax[2].set_yticks([0, 50, 100, 150, 200, 250])
ax[2].set_xticks(np.arange(t) + 1, [str(i+1) for i in range(t-1)] + [f'>{t-1}'])
ax[2].tick_params('x', labelsize=FONTSIZE-1)
ax[2].tick_params('y', labelsize=FONTSIZE-1)
ax[2].set_ylabel('Count', fontsize=FONTSIZE)
ax[2].set_xlabel('Segment length', fontsize=FONTSIZE)
ax[2].legend(fontsize=FONTSIZE-1)
labels = ['Rel. Conservation','Coverage','GSS']
for i in range(2):
    ax[i].set_ylabel(labels[i], fontsize=FONTSIZE)
    ax[i].set_xlabel('Epoch', fontsize=FONTSIZE)
    # ax[i].set_xticks([0, 200, 400, 600, 800, 1000, 1000])
    ax[i].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax[i].tick_params('x', labelsize=FONTSIZE-1)
    ax[i].tick_params('y', labelsize=FONTSIZE-1)
ax[0].legend(ncol=2, loc='lower left', fontsize=FONTSIZE-1)
ax[1].set_title(f't={t}', x=0.5, y=0.7, fontsize=FONTSIZE)
# fig.legend(ncol=4, loc='lower left', fontsize=FONTSIZE-1)
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.3, wspace=0.35)
fig.savefig(f'{plot_dir}/exploit_t{t}.pdf')
fig.savefig(f'{plot_dir}/exploit_t{t}.png')
