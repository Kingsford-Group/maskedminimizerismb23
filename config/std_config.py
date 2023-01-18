from src.priority_net import *
from src.template_net import *
from src.loss_functions import *
from src.sequence_env import *
from src.mutate_net import *

def create_config(
    l, w, k, seq,
    d=64, eps=2.0,
    mask=None,
    artifact_root='../artifact',
    n_mutations=1
):
    if mask is None:
        mask = torch.ones(w)
    config = {
        'l': l, 'w': w, 'k': k,
        'batch_size': 32,
        'batch_per_epoch': 10,
        'n_mutations': n_mutations,
        'priority_net': PriorityNet(l, k, w, len(chmap.keys()), d=d),
        'template_net': TemplateNet(l, k, w, len(chmap.keys()), d=d, eps=eps),
        'loss_function': MaxpoolDelta(w),
        'loss_wrapper': HbdLossWrapper,
        'mutation_net': VectorizedMutateNet(len(chmap.keys())),
        'mask': mask
    }
    if os.path.exists(f'{artifact_root}/{seq}.dat'):
        config['seq_dataset'] = torch.load(f'{artifact_root}/{seq}.dat')
        config['seq_dataset'].create_fragments(l)
    else:
        config['seq_dataset'] = SequenceDataset(list(chmap.keys()), f'{SEQ_DIR}{seq}.seq')
        torch.save(config['seq_dataset'], f'{artifact_root}/{seq}.dat')
        config['seq_dataset'].create_fragments(l)
    return config

cms_mask = lambda w, t: torch.ones(w) - F.one_hot(torch.tensor([t]), w).float()
mnz_mask = lambda w: torch.ones(w)
ops_mask = lambda w, t: F.one_hot(torch.tensor([t]), w).float()