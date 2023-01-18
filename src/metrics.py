from utils.util import *

# All metric computation is cumulative
class Metrics:
    def __init__(self):
        self.cumulative_metric = 0.0
        self.cumulative_norm = 0.0

    def update(self, minimizer, *args, **kwargs):
        self.cumulative_metric += torch.sum(minimizer)
        self.cumulative_norm += torch.numel(minimizer)

    def tally(self):
        if self.cumulative_norm > 0.0: return self.cumulative_metric / self.cumulative_norm
        else: return torch.tensor(0.0)

class Density(Metrics):
    def __init__(self):
        super(Density, self).__init__()

    def update(self, minimizer, *args, **kwargs):
        super(Density, self).update(minimizer)

class Conservation(Metrics):
    def __init__(self):
        super(Conservation, self).__init__()

    def update(self, minimizer, mutated_minimizer=None, *args, **kwargs):
        total_conserved = torch.sum(minimizer[None, :, :, :] * mutated_minimizer)
        self.cumulative_metric = total_conserved / mutated_minimizer.shape[0]
        self.cumulative_norm = torch.numel(minimizer)


class Coverage(Metrics):
    def __init__(self, w):
        super(Coverage, self).__init__()
        self.w = w

    def update(self, minimizer, *args, **kwargs):
        covered_window = (torch.sum(minimizer.unfold(-1, self.w, 1), dim=-1) > 0).float()
        self.cumulative_metric = torch.sum(covered_window)
        self.cumulative_norm = torch.numel(covered_window)


class GeneralizedSketchScore(Metrics):
    def __init__(self, w):
        super(GeneralizedSketchScore, self).__init__()
        self.cov = Coverage(w)
        self.den = Density()
        self.con = Conservation()

    def update(self, minimizer, mutated_minimizer=None, *args, **kwargs):
        self.cov.update(minimizer)
        self.den.update(minimizer)
        self.con.update(minimizer, mutated_minimizer)

    def tally(self):
        den = self.den.tally()
        if den > 1e-10: return self.cov.tally() * self.con.tally() / den
        else: return torch.tensor(0.0)

    def tally_all(self):
        den = self.den.tally()
        cov = self.cov.tally()
        con = self.con.tally()
        gss = self.cov.tally() * self.con.tally() / den if den > 1e-9 else torch.tensor(0.0)
        return den, cov, con, gss