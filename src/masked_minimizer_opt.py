from src.template_net import *
from src.loss_functions import *
from src.priority_net import *
from src.metrics import *

class MaskedMinimizerOpt(nn.Module):
    def __init__(self, config):
        super(MaskedMinimizerOpt, self).__init__()
        self.priority_net = config['priority_net'].to(device)
        self.template_net = config['template_net'].to(device)
        self.mutation_net = config['mutation_net'].to(device)
        self.w = self.priority_net.w
        self.n_mutations  = config['n_mutations']
        self.loss_function = config['loss_function'].to(device)
        self.loss_wrapper = config['loss_wrapper'](self.loss_function, self.n_mutations).to(device)
        self.batch_size = config['batch_size']
        self.batch_per_epoch = config['batch_per_epoch']
        self.seq_dataset = config['seq_dataset']
        self.history = {
            'epoch': [], 'loss': [],
            'den': [], 'best_den': [],
            'cov': [], 'best_cov': [],
            'con': [], 'best_con': [],
            'gss': [], 'best_gss': [],
            'model': []
        }
        self.mask = config['mask'].to(device)
        self.save_path = config['save_path']

    def init_trainer(self):
        opt = Adam(self.parameters(), lr=2e-3)
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='max')
        self.template_net.apply(init_weights)
        return opt, scheduler

    def train_minimizer(self, n_epochs=1000, eval_interval=10, save_model=False):
        opt, scheduler = self.init_trainer()
        bar = trange(n_epochs)
        for i in bar:
            bar.set_description_str(f'Epoch {i}')
            # Eval every few epochs
            if (i % eval_interval) == 0:

                den, cov, con, gss = self.prune_mask(bar)
                # scheduler.step(gss)
                scheduler.step(-cov)
                self.history['epoch'].append(i)
                self.history['den'].append(den.item())
                self.history['cov'].append(cov.item())
                self.history['con'].append(con.item())
                self.history['gss'].append(gss.item())
                if save_model:
                    self.history['model'].append(self.priority_net.state_dict())
                if i == 0:
                    self.history['best_den'].append(den.item())
                    self.history['best_cov'].append(cov.item())
                    self.history['best_con'].append(con.item())
                    self.history['best_gss'].append(gss.item())
                else:
                    self.history['best_den'].append(
                        min(den.item(), self.history['best_den'][-1]))
                    self.history['best_cov'].append(
                        max(cov.item(), self.history['best_cov'][-1]))
                    self.history['best_con'].append(
                        max(con.item(), self.history['best_con'][-1]))
                    self.history['best_gss'].append(
                        max(gss.item(), self.history['best_gss'][-1]))
            epoch_loss, epoch_fragments = 0.0, 0.0

            # Optimization loop
            for j in range(self.batch_per_epoch):
                opt.zero_grad()
                fragments = self.seq_dataset.sample_batch(self.batch_size).to(device)
                loss = self.loss_wrapper(
                    self.mutation_net, self.priority_net,
                    self.template_net, fragments, self.mask
                )
                loss.backward()
                opt.step()
                bar.set_postfix_str(f'Batch Loss ({j + 1}/{self.batch_per_epoch})={loss.item():.3f}')
                epoch_loss += loss.item()
                epoch_fragments += fragments.shape[0]
                del fragments
                torch.cuda.empty_cache()

            if (i % eval_interval) == 0:
                self.history['loss'].append(epoch_loss / epoch_fragments)
                if self.save_path is not None:
                    torch.save(self.history, self.save_path)

    def prune_mask(self, bar, n_mutations=5):
        # Try minimizer mask
        bar.set_description_str('Try mnz mask')
        result_mnz = self.eval_minimizer(bar, torch.ones(self.w).to(device))
        # Try curr mask
        bar.set_description_str('Try cur mask')
        result_curr = self.eval_minimizer(bar, self.mask)
        if result_mnz[-1] > result_curr[-1]:
            result_best = result_mnz
            mask_best = torch.ones(self.w).to(device)
        else:
            result_best = result_curr
            mask_best = self.mask

        # Try pruning mask
        for i in range(self.w):
            if self.mask[i] == 1:
                bar.set_description_str(f'Try pruning mask at pos {i}')
                mask_try = self.mask * (1. - F.one_hot(torch.tensor([i]), self.w).squeeze()).to(device)
                result_try = self.eval_minimizer(bar, mask_try)
                if result_try[-1] > result_best[-1]:
                    result_best = result_try
                    mask_best = mask_try
            else:
                bar.set_description_str(f'Try filling mask at pos {i}')
                mask_try = self.mask + F.one_hot(torch.tensor([i]), self.w).squeeze().to(device)
                result_try = self.eval_minimizer(bar, mask_try)
                if result_try[-1] > result_best[-1]:
                    result_best = result_try
                    mask_best = mask_try

        self.mask = mask_best
        print(f'\n{mask_best.cpu().tolist()}')
        return result_best


    def eval_minimizer(self, bar, mask, n_mutations=5):
        frag_idx = list(range(0, len(self.seq_dataset), self.seq_dataset.frag_length))
        buffer = []
        metric = GeneralizedSketchScore(self.priority_net.w)
        def eval_buffer(_buffer):
            with torch.no_grad():
                if len(_buffer) > 1:
                    _buffer = torch.stack(_buffer, dim=0).to(device)
                else:
                    _buffer = _buffer[-1].unsqueeze(0).to(device)

                _buffer = F.one_hot(_buffer, len(self.seq_dataset.vocab)).transpose(1, 2).float()
                _, original_mnz = self.priority_net.select(_buffer, mask)
                mutated_mnz = []
                for t in range(n_mutations):
                    _, mutn_mnz = self.priority_net.select(self.mutation_net(_buffer), mask)
                    mutated_mnz.append(mutn_mnz)
                metric.update(original_mnz, torch.stack(mutated_mnz, dim=0))
                den, cov, con, gss = metric.tally_all()
                bar.set_postfix_str(f'Metric (d,v,c,g) = {den:.3f}/{cov:.3f}/{con:.3f}/{gss:.3f}')
                del _buffer, original_mnz, mutated_mnz
                torch.cuda.empty_cache()

        for i, idx in enumerate(frag_idx):
            buffer.append(self.seq_dataset[idx])
            if len(buffer) == self.batch_size:
                eval_buffer(buffer)
                buffer = []
        if len(buffer) > 0:
            eval_buffer(buffer)
        print('')
        return metric.tally_all()