# This experiment compares compatible
#   minimizer,
#   open-syncmer and
#   complement-syncmer
#   fix k, varying w, all trained with DeepMinimizer

from src.control_minimizer import *

seq = 'chrXC'
seq_dataset = RawSequenceDataset(list(chmap.keys()), f'{SEQ_DIR}/{seq}.seq')
seq_dataset.create_fragments(5000)
w, k = [10, 15, 20], 15
mm = 'random'
artifact_root = f'../artifact/exp2/{seq}'
if not os.path.exists(artifact_root):
    os.makedirs(artifact_root)
control_mm = CONTROL[mm](k)
for wval in w:
    mask_mnz = list(np.arange(wval))
    mask_ops, mask_cms = [],[]
    for t in range(wval):
        mask_ops.append([t])
        mask_cms.append([])
        for j in range(wval):
            if j != t:
                mask_cms[-1].append(j)

    # minimizer control
    print(f'{mm}-{wval}-{k}-minimizer')
    config = {
        'w': wval, 'k': k, 'mm': control_mm,
        'mask': mask_mnz,
        'seq_dataset': seq_dataset
    }
    ctrl_mnz = ControlMinimizer(config)
    d, v, c, g = ctrl_mnz.eval_minimizer(2)
    artifact = {
        'mnz': {'d': d, 'v': v, 'c': c, 'g': g}
    }
    torch.save(artifact, f'{artifact_root}/{mm}_w{wval}_k{k}.pt')
    for t in [2, wval//2, wval-1]:
        # ops control
        print(f'{mm}-{wval}-{k}-{t}-opensyncmer')
        config = {
            'w': wval, 'k': k, 'mm': control_mm,
            'mask': mask_ops[t],
            'seq_dataset': seq_dataset
        }
        ctrl_ops = ControlMinimizer(config)
        d, v, c, g = ctrl_ops.eval_minimizer(2)
        artifact[f'ops_{t}'] = {'d': d, 'v': v, 'c': c, 'g': g}
        torch.save(artifact, f'{artifact_root}/{mm}_w{wval}_k{k}.pt')
        # cms control
        print(f'{mm}-{wval}-{k}-{t}-complementsyncmer')
        config = {
            'w': wval, 'k': k, 'mm': control_mm,
            'mask': mask_cms[t],
            'seq_dataset': seq_dataset
        }
        ctrl_cms = ControlMinimizer(config)
        d, v, c, g = ctrl_cms.eval_minimizer(2)
        artifact[f'cms_{t}'] = {'d': d, 'v': v, 'c': c, 'g': g}
        torch.save(artifact, f'{artifact_root}/{mm}_w{wval}_k{k}.pt')