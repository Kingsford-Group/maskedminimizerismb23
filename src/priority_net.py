from utils.util import *

class PriorityNet(nn.Module):
    def __init__(self, l, k, w, vocab_size, d=256):
        super(PriorityNet, self).__init__()
        self.l, self.k, self.w = l, k, w
        self.vocab_size = vocab_size
        self.score_net = nn.Sequential(
            # kmer embedding 1: [N * V * L] -> [N * D * (L - K + 1)]
            nn.Conv1d(
                in_channels=vocab_size,
                out_channels=d,
                kernel_size=self.k
            ),
            nn.ReLU(),
            # kmer embedding 2: [N * D * (L - K + 1)] -> [N * (D / 2) * (L - k + 1)]
            nn.Conv1d(
                in_channels=d,
                out_channels=d//2,
                kernel_size=1
            ),
            nn.ReLU(),
            # kmer score: [N * D/2 * (L - K + 1)] -> [N * 1 * (L - k + 1)]
            nn.Conv1d(
                in_channels=d//2,
                out_channels=1,
                kernel_size=1
            ),
            nn.Sigmoid(),
        )
        self.mp = nn.MaxPool1d(w, stride=1, return_indices=True)
        self.up = nn.MaxUnpool1d(w, stride=1)

    # S dim: [N * V * L]
    def forward(self, s):
        return self.score_net(s)

    def select(self, s, mask=None):
        score = self.score_net(s)
        val, idx = self.mp(score + 1e-4)
        peak = self.up(val, idx, output_size=score.shape)
        minimizer = peak / ((peak == 0) + peak)
        if mask is None:
            return peak, minimizer
        else:
            # mask is a binary vec
            local_idx = idx - torch.arange(idx.shape[-1]).to(idx.device)
            local_sel = torch.sum(F.one_hot(local_idx, self.w) * mask[None, None, :], dim=-1)
            window_sel = F.one_hot(idx, peak.shape[-1]) * local_sel[:, :, :, None]
            global_sel = torch.sum(window_sel, dim=-2) > 0
            return peak * global_sel.float(), minimizer * global_sel.float()

# if __name__ == '__main__':
#     # Unit test code
#     n, l, k, w = 3, 100, 4, 5
#     vocab_size = 4
#     pnet = PriorityNet(l,k,w,vocab_size)
#     s = torch.randint(vocab_size, (n, l))
#     s = F.one_hot(s, vocab_size).transpose(1, 2).float()
#     mask = torch.tensor([0, 0, 1, 1, 0])
#     mask_peak, mask_minimizer = pnet.select(s, mask)
#     peak, minimizer = pnet.select(s)
