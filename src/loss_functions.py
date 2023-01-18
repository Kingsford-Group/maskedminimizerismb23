from utils.util import *

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, priority, template):
        return F.mse_loss(priority, template)

class RegL2Loss(nn.Module):
    def __init__(self):
        super(RegL2Loss, self).__init__()

    def forward(self, priority, template):
        return F.mse_loss(priority, template) - torch.norm(template)

class MaxpoolDelta(L2Loss):
    def __init__(self, w):
        super(MaxpoolDelta, self).__init__()
        self.mp = nn.MaxPool1d(w, stride=1, return_indices=True)
        self.up = nn.MaxUnpool1d(w, stride=1)

    def forward(self, priority, template):
        val, idx = self.mp(priority + 1e-4)
        peak = self.up(val, idx, output_size=priority.shape)
        matching_loss = torch.sum(template * (template - priority) ** 2)
        r1 = torch.norm(priority - peak) ** 2.0
        r2 = torch.norm(template) ** 2.0
        return matching_loss + r1 - r2

class MaskedMaxpoolDelta(nn.Module):
    def __init__(self, w):
        super(MaskedMaxpoolDelta, self).__init__()
        self.mp = nn.MaxPool1d(w, stride=1, return_indices=True)
        self.up = nn.MaxUnpool1d(w, stride=1)
        self.w = w

    def forward(self, priority, template, mask):
        val, idx = self.mp(priority + 1e-4)
        peak = self.up(val, idx, output_size=priority.shape)
        r1 = torch.norm(priority - peak) ** 2.0
        r2 = torch.norm(template ** 2.0)
        # priority = priority.unfold(-1, self.w, 1)
        # template = template.unfold(-1, self.w, 1)
        weighted_l2 = template * (template - priority) ** 2
        weighted_l2 = weighted_l2.unfold(-1, self.w, 1)
        matching_loss = torch.sum(mask[None, None, :] * weighted_l2) / self.w
        return matching_loss + r1 - r2

class LossWrapper(nn.Module):
    def __init__(self, loss_fn, den_coeff=1, con_coeff=1,
                 n_mutations=2, align_template=True):
        super(LossWrapper, self).__init__()
        self.loss_fn = loss_fn
        self.den_coeff = den_coeff / (n_mutations + 1)
        self.con_coeff = con_coeff / (n_mutations + 1)
        self.n_mutations = n_mutations
        self.align_template = align_template

    def forward(self, mutation_net, priority_net, template_net, fragments):
        priority = priority_net(fragments)
        if self.align_template:
            template = template_net(fragments)
        else:
            template = priority
        den_loss = 0.0
        con_loss = 0.0
        if self.den_coeff > 0:
            den_loss = self.den_coeff * self.loss_fn(priority, template)

        if self.con_coeff > 0:
            mutations = [
                priority_net(mutation_net(fragments))
                for _ in range(self.n_mutations)
            ]
            con_loss = torch.sum(torch.stack([
                self.con_coeff * self.loss_fn(m, template)
                for m in mutations
            ]))
        return den_loss + con_loss

class ExploitLossWrapper(nn.Module):
    def __init__(self, loss_fn, n_mutations=2):
        super(ExploitLossWrapper, self).__init__()
        self.loss_fn = loss_fn
        self.n_mutations = n_mutations

    def forward(self, mutation_net, priority_net, template_net, fragments):
        priority = priority_net(fragments)
        loss = 0.0
        for _ in range(self.n_mutations):
            mutation = priority_net(mutation_net(fragments))
            loss += torch.sum(priority * (mutation - priority) ** 2) / self.n_mutations
        return loss

class MaskedLossWrapper(nn.Module):
    def __init__(self, loss_fn, den_coeff=1, con_coeff=1,
                 n_mutations=2, align_template=True):
        super(MaskedLossWrapper, self).__init__()
        self.loss_fn = loss_fn
        self.den_coeff = den_coeff / (n_mutations + 1)
        self.con_coeff = con_coeff / (n_mutations + 1)
        self.n_mutations = n_mutations
        self.align_template = align_template

    def forward(self, mutation_net, priority_net, template_net, fragments, mask):
        priority = priority_net(fragments)
        if self.align_template:
            template = template_net(fragments)
        else:
            template = priority
        den_loss = 0.0
        con_loss = 0.0
        if self.den_coeff > 0:
            den_loss = self.den_coeff * self.loss_fn(priority, template, mask)

        if self.con_coeff > 0:
            mutations = [
                priority_net(mutation_net(fragments))
                for _ in range(self.n_mutations)
            ]
            con_loss = torch.sum(torch.stack([
                self.con_coeff * self.loss_fn(m, template, mask)
                for m in mutations
            ]))
        return den_loss + con_loss



# Loss macro
DenLossWrapper = lambda loss_fn, n_mutations: LossWrapper(loss_fn, 1, 0, n_mutations)
ConLossWrapper = lambda loss_fn, n_mutations: LossWrapper(loss_fn, 0, 1, n_mutations)
HbdLossWrapper = lambda loss_fn, n_mutations: LossWrapper(loss_fn, 1, 1, n_mutations)
MaskedDenLossWrapper = lambda loss_fn, n_mutations: MaskedLossWrapper(loss_fn, 1, 0, n_mutations)
MaskedConLossWrapper = lambda loss_fn, n_mutations: MaskedLossWrapper(loss_fn, 0, 1, n_mutations)
MaskedHbdLossWrapper = lambda loss_fn, n_mutations: MaskedLossWrapper(loss_fn, 1, 1, n_mutations)

EptLossWrapper = lambda loss_fn, n_mutations: LossWrapper(loss_fn, 1, 1, n_mutations, align_template=False)
EptLossWrapper_v2 = lambda loss_fn, n_mutations: ExploitLossWrapper(loss_fn, n_mutations)