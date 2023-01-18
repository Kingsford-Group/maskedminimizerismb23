import torch.distributions

from utils.util import *

class MutateNet(nn.Module):
    def __init__(self, vocab_size, mutation_prob=0.2):
        super(MutateNet, self).__init__()
        self.vocab_size = vocab_size
        self.mutation_prob = mutation_prob
        self.mutation_model = torch.distributions.Categorical(torch.ones(self.vocab_size))

    # S dim: [N * V * L]
    def forward(self, s):
        n, v, l = s.shape
        mutation_loc = torch.bernoulli(self.mutation_prob * torch.ones((n, l)))
        loc_n, loc_l = torch.where(mutation_loc==1.0)
        loc = list(zip(loc_n.tolist(), loc_l.tolist()))
        s_prime = torch.clone(s.transpose(1, 2))
        for nid, lid in loc:
            s_prime[nid, lid] = F.one_hot(self.mutation_model.sample(), self.vocab_size)

        s_prime = s_prime.transpose(1, 2)
        return s_prime

class VectorizedMutateNet(nn.Module):
    def __init__(self, vocab_size, mutation_prob=0.2):
        super(VectorizedMutateNet, self).__init__()
        self.vocab_size = vocab_size
        self.mutation_prob = mutation_prob

    def forward(self, s):
        n, v, l = s.shape
        s_prime = s.transpose(1, 2)
        mutation_mask = torch.bernoulli(self.mutation_prob * torch.ones(n, l)).to(s.device)
        random_s = F.one_hot(torch.randint(v, (n, l)), v).to(s.device)
        s_prime = s_prime * (1. - mutation_mask)[:,:,None] + random_s * mutation_mask[:,:,None]
        return s_prime.transpose(1, 2)