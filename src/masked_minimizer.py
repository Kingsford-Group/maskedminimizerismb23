from src.template_net import *
from src.loss_functions import *
from src.priority_net import *
from src.metrics import *

class MaskedMinimizer(nn.Module):
    def __init__(self, config):
        super(MaskedMinimizer, self).__init__()
        self.priority_net = config['priority_net'].to(device)
        self.template_net = config['template_net'].to(device)
        self.mutation_net = config['mutation_net'].to(device)
        self.w = self.priority_net.w
        if isinstance(config['mask'], torch.Tensor):
            self.mask = config['mask'].to(device)
            self.find_mask = False
        else:
            self.mask = config['mask']
            self.find_mask = True
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
        self.save_path = config['save_path']

    def init_trainer(self):
        opt = Adam(self.parameters(), lr=2e-3)
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='max')
        self.template_net.apply(init_weights)
        return opt, scheduler

    def train_minimizer(self, n_epochs=1000, eval_interval=10, init=True, save_model=False):
        if init: opt, scheduler = self.init_trainer()
        bar = trange(n_epochs)
        for i in bar:
            bar.set_description_str(f'Epoch {i}')
            if (i % eval_interval) == 0:
                if self.find_mask:
                    (den, cov, con, gss), best_mask = self.eval_minimizer_find_mask(bar)
                else:
                    den, cov, con, gss = self.eval_minimizer(bar)
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
            for j in range(self.batch_per_epoch):
                opt.zero_grad()
                fragments = self.seq_dataset.sample_batch(self.batch_size).to(device)
                loss = self.loss_wrapper(self.mutation_net, self.priority_net, self.template_net, fragments)
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

    def eval_minimizer_find_mask(self, bar, n_mutations=5):
        self.mask = torch.ones(self.w).to(device)
        best_result = self.eval_minimizer(bar, n_mutations)
        pruned_list = []
        while True:
            bar.set_description_str(f'Pruning {len(pruned_list) + 1}--')
            current_mask = best_try_mask = deepcopy(self.mask)
            best_improvement, best_i = 0, -1
            best_try_result = (0,0,0,0)
            for i in range(self.w):
                if self.mask[i] == 1:
                    # try pruning mask
                    self.mask[i] = 0
                    try_result = self.eval_minimizer(bar, n_mutations)
                    improvement = try_result[-1] - best_result[-1]
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_try_result = try_result
                        best_try_mask = deepcopy(self.mask)
                        best_i = i
                    # reset mask
                    self.mask = deepcopy(current_mask)
            if best_improvement > 0:
                self.mask = deepcopy(best_try_mask)
                best_result = best_try_result
                pruned_list.append(best_i)
            else:
                bar.set_description_str(f'')
                return best_result, current_mask

    def eval_minimizer(self, bar, n_mutations=5):
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
                _, original_mnz = self.priority_net.select(_buffer, self.mask)
                mutated_mnz = []
                for t in range(n_mutations):
                    _, mutn_mnz = self.priority_net.select(self.mutation_net(_buffer), self.mask)
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