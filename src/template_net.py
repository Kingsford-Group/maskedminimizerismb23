from utils.util import *

class DelayNet(nn.Module):
    def __init__(self, k, w, vocab_size, d=64, eps=1.0):
        super(DelayNet, self).__init__()
        self.eps = eps
        self.delay_net = nn.Sequential(
            # window-level embedding:
            # [N * V * L] -> [N * D * (L-W-K+2)]
            nn.Conv1d(
                in_channels=vocab_size,
                out_channels=d,
                kernel_size=w + k -1
            ),
            nn.ReLU(),
            # window-level embedding:
            # [N * D * (L-W-K+2)] -> [N * (D / 2) * (L-W-K+2)]
            nn.Conv1d(
                in_channels=d,
                out_channels=d//2,
                kernel_size=1
            ),
            nn.ReLU(),
            # deconvolution into positional delay:
            # [N * D/2 * (L-W-K+2)] -> [N * 1 * (L-K+1)]
            nn.ConvTranspose1d(
                in_channels=d//2,
                out_channels=1,
                kernel_size=w
            ),
            nn.Tanh()
        )
    def forward(self, x):
        return self.eps * self.delay_net(x)

class TemplateNet(nn.Module):
    def __init__(self, l, k, w, vocab_size, d=64, n_waves=10, eps=1.0):
        super(TemplateNet, self).__init__()
        self.l, self.k, self.w = l, k, w
        self.n_kmer = l - k + 1
        self.n_waves = n_waves
        self.vocab_size = vocab_size
        if eps > 1e-7:
            self.delay_net = DelayNet(k, w, vocab_size, d, eps)
        else:
            self.delay_net = lambda x: 0.0
        self.sin_amps = nn.Parameter(torch.ones(n_waves), requires_grad=True)
        self.cos_amps = nn.Parameter(torch.ones(n_waves), requires_grad=True)
        self.period = torch.tensor(2.0 * math.pi / self.w)
        self.bias = nn.Parameter(torch.tensor(0.), requires_grad=True)

    # Expect x of size [N * V * L]
    def forward(self, x):
        position = torch.arange(x.shape[-1] - self.k + 1).view(1, 1, -1).float()
        position = position.repeat(x.shape[0], 1, 1).to(device)
        position += self.delay_net(x)
        template = []
        for i in range(self.n_waves):
            template.append(self.sin_amps[i] * torch.sin(self.period * position * (i+1)))
            template.append(self.cos_amps[i] * torch.cos(self.period * position * (i+1)))
        return F.sigmoid(torch.sum(torch.cat(template, dim=1), dim=1) + self.bias)

class EnsembleTemplateNet(nn.Module):
    def __init__(self, l, k, w, vocab_size, d=64, n_waves=10, eps=1.0):
        super(EnsembleTemplateNet, self).__init__()
        self.l, self.k, self.w = l, k, w
        self.n_kmer = l - k + 1
        self.vocab_size = vocab_size
        self.n_waves = n_waves
        if eps > 1e-7:
            self.delay_net = DelayNet(k, w, vocab_size, d, eps)
        else:
            self.delay_net = lambda x: 0.0
        self.cos_amps = nn.Parameter(torch.ones(n_waves), requires_grad=True)
        self.period = torch.tensor(2.0 * math.pi / self.w)
        self.bias = nn.Parameter(torch.tensor(0.), requires_grad=True)

    # Expect x of size [N * V * L]
    def forward(self, x):
        position = torch.arange(x.shape[-1] - self.k + 1).view(1, 1, -1).float()
        position = position.repeat(x.shape[0], 1, 1).to(device)
        position += self.delay_net(x)
        template = []
        for i in range(self.n_waves):
            template.append(self.cos_amps[i] * torch.cos(self.period * (position + float(i))))
        return F.sigmoid(torch.sum(torch.cat(template, dim=1), dim=1) + self.bias)