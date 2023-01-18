from utils.util import *
from src.benchmark import *
from src.metrics import *
from src.sequence_env import *
CONTROL = {
    'miniception': lambda kv: Miniception(kv, k0=5),
    'random': lambda kv: RandomMinimizers(kv),
    'pasha': lambda kv: PASHA(kv)
}

class ControlMinimizer:
    def __init__(self, config):
        self.w = config['w']
        self.k = config['k']
        self.mask = config['mask']
        self.seq_dataset = config['seq_dataset']
        self.mm = config['mm']
        self.mp = nn.MaxPool1d(self.w, stride=1, return_indices=True)
        self.up = nn.MaxUnpool1d(self.w, stride=1)

    def mutate_seq(self, s, rate=0.2):
        mutation_loc = torch.bernoulli(rate * torch.ones(len(s)))
        mutation_loc = torch.where(mutation_loc==1)[0].tolist()
        sprime = deepcopy(s)
        for loc in mutation_loc:
            sprime[loc] = random.choice(list(chmap.keys()))
        return sprime

    def select(self, score):
        select = torch.zeros(score.shape[0])
        score = score.unfold(-1, self.w, 1)
        minimizer = torch.argmax(score, dim=-1)
        for i in range(minimizer.shape[0]):
            if minimizer[i] in self.mask:
                select[i + minimizer[i]] = 1.
        return select

    def eval_minimizer(self, n_mutations=5):
        with torch.no_grad():
            metric = GeneralizedSketchScore(self.w)
            bar = trange(len(self.seq_dataset))
            for i in bar:
                s = self.seq_dataset[i]
                score = torch.tensor([score for score in self.mm.stream_kmer_level(s)])
                original_mnz = self.select(score).view(1, 1, -1)
                mutated_mnz = []
                for m in range(n_mutations):
                    sm = self.mutate_seq(s)
                    score = torch.tensor([score for score in self.mm.stream_kmer_level(sm)])
                    mutated_mnz.append(self.select(score).view(1, 1, -1))
                metric.update(original_mnz, torch.stack(mutated_mnz, dim=0))
                den, cov, con, gss = metric.tally_all()
                bar.set_postfix_str(f'Metric(d/v/c/g):{den:.3f}/{cov:.3f}/{con:.3f}/{gss:.3f}')
        return metric.tally_all()

