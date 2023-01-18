from utils.util import *


class MinimizersImpl:
    """
    Base class for minimizers; this is actually a random minimizer.
    warning: higher level means higher priority of picking.
    """
    def __init__(self, k):
        self.k = k
        self.maxround = 2
        self.unique_order = False

    def kmer_level(self, kmer):
        """
        Given a k-mer in integer representation return its level.
        """
        return 0

    def stream_kmer_level(self, s):
        """
        Given a string, return the level of its constituent k-mers.
        """
        for km in sequence_mer_iterator(self.k, s):
            yield self.kmer_level(km)

class RandomMinimizers(MinimizersImpl):
    """
    Base class for minimizers; this is actually a random minimizer.
    warning: higher level means higher priority of picking.
    """
    def __init__(self, k):
        super().__init__(k)
        self.modulus = 4 ** k
        self.order = np.random.rand(self.modulus)

    def kmer_level(self, kmer):
        """
        Given a k-mer in integer representation return its level.
        """
        return self.order[kmer]

    def stream_kmer_level(self, s):
        """
        Given a string, return the level of its constituent k-mers.
        """
        for km in sequence_mer_iterator(self.k, s):
            yield self.kmer_level(km)

class Miniception(MinimizersImpl):
    """
    Implements the Miniception.
    the exact rules are:
        (1) there is a smaller minimizer (w0, k0), with w0+k0-1 = k.
        (2) for any k-mer, the smaller random minimizer is applied. if the
                first or the last k0-kmer is selected this k-mer is in the priority class.
    """
    def __init__(self, k, k0):
        super().__init__(k)
        self.k0 = k0
        self.rand_multi = random.randrange(4 ** k0)

    def kmer_level(self, kmer):
        k, k0 = self.k, self.k0
        submod = 4 ** k0
        sub_kmers = []
        cur = kmer
        for i in range(k - k0 + 1):
            sub_kmers.append(((cur % submod) * self.rand_multi) % submod)
            cur = cur // 4
        ss = min(sub_kmers)
        if ss == sub_kmers[0]:
            return 1
        if ss == sub_kmers[-1]:
            return 1
        return 0


class PASHA(MinimizersImpl):
    """
        Load and parse PASHA output
    """
    def __init__(self, k):
        super().__init__(k)
        self.bpd = 64
        self.modulus = 4 ** k
        self.data = [0] * (self.modulus // self.bpd)
        self.pasha_uhs = {
            5: 40, 6: 70, 7: 100, 8: 100,
            9: 100, 10: 100, 11: 90, 12: 100,
            13: 100, 14: 100, 15: 100, 16: 100
        }
        self.url = f'http://pasha.csail.mit.edu/sets_july24/k{k}/PASHA{k}_{self.pasha_uhs[k]}.txt'
        self.decyc_url = f'http://pasha.csail.mit.edu/sets_july24/k{k}/decyc{k}.txt'
        self.fn = f'../artifact/pasha_uhs/PASHA{k}_{self.pasha_uhs[k]}.txt'
        self.decyc_fn = f'../artifact/pasha_uhs/decyc{k}.txt'
        if not os.path.exists('../artifact/pasha_uhs'):
            os.makedirs('../artifact/pasha_uhs')

        if not os.path.exists(self.fn):
            response = wget.download(self.url, self.fn)
            print(response)
        if not os.path.exists(self.decyc_fn):
            response = wget.download(self.decyc_url, self.decyc_fn)
            print(response)
        with open(self.fn) as f:
            for line in tqdm(f):
                s = line.strip()
                self.process_line(s)
        with open(self.decyc_fn) as f:
            for line in tqdm(f):
                s = line.strip()
                self.process_line(s)

    def process_line(self, s):
        assert len(s) == self.k
        cm = 0
        for c in s:
            cm = cm * 4 + chmap[c]
        self.data[cm // self.bpd] |= 2 ** (cm % self.bpd)

    def kmer_level(self, kmer):
        return int((self.data[kmer // self.bpd] & (2 ** (kmer % self.bpd))) > 0)


class MinimizerFromOrder(MinimizersImpl):
    '''
    Implements a minimizer where the ordering comes from an external file.
    Assumes ACGT alphabet.
    '''
    def __init__(self, k, fn):
        super().__init__(k)
        self.unique_order = True
        self.kms = 4 ** k
        self.maxround = self.kms + 1
        self._order = [0] * self.kms
        with open(fn) as f:
            idx = 0
            for line in f:
                s = line.strip()
                if len(s) == 0:
                    continue
                assert len(s) == k
                self._order[kmer_to_int(s)] = self.kms - idx
                idx += 1
        assert idx == self.kms

    def kmer_level(self, kmer):
        return self._order[kmer]


def selected_locs_control(seq, mm: MinimizersImpl):
    '''
    Calculates the density of a minimizer over a sequence.
    Currently assumes the k-mers are in a complete order (no randomness involved).
    @param seq: the actual sequence.
    @param mm: an instance of the minimizer.
    @return: a value indicating (specific) density.
    '''
    w, k = mm.w, mm.k
    #assert mm.maxround >= 4 ** k
    kms = list(0 - mm.kmer_level(x) for x in sequence_mer_iterator(k, seq))
    last = -1
    count = 0
    for i in trange(len(kms) - w + 1):
        sl = kms[i:i+w]
        assert len(sl) == w
        cidx = np.argmin(sl) + i  # current pick
        if cidx != last:
            assert cidx > last
            last = cidx
            count += 1
    return count / len(kms)


def calc_selected_locs(seq, mm: MinimizersImpl, robust_windows=False, ret_array=False):
    """
    Calculates the density of a minimizer over a sequence, with randomized parts
    instantiated using a completely random ordering. Currently only supports k<=15.
    (This might be relaxed in the future using a pseudorandom hash function)
    @param seq: the actual sequence.
    @param mm: an instance of random minimizer.
    @param robust_windows: If using the robust windows technique.
    @param ret_array: If set to true, return if a new k-mer is selected at every context.
    @return: a value if ret_array is False, otherwise see @param ret_array.
    """
    w, k = mm.w, mm.k
    n = len(seq) - (k-1)
    modulus = 4 ** k
    order = []
    if not mm.unique_order:
        assert k <= 16
        for i in trange(modulus):
            order.append(i)
            j = random.randrange(i+1)
            if j != i:
                order[i], order[j] = order[j], order[i]
    #  while (seed % 2) == 0:
    #  seed = random.randrange(modulus)  # ensures results are unique.
    km_it = sequence_mer_iterator(k, seq)
    prio_it = mm.stream_kmer_level(seq)
    def next_val():
        km = next(km_it)
        prio = next(prio_it)
        #  km_sh = (km * seed) % modulus
        if not mm.unique_order:
            km_sh = order[km]
            return km_sh - prio * modulus
        else:
            return 0 - prio
    val_queue = deque()
    ret = 0
    ret_vals = []
    for i in range(w):
        val = next_val()
        val_queue.append(val)
    last_val = min(val_queue)
    last_dist = None
    last_count = 0
    for i in range(w):
        if val_queue[w - 1 - i] == last_val:
            if (last_dist is None) or (not robust_windows):
                last_dist = i
            last_count += 1
    for _ in trange(w, n):
        # new window, doesn't care about contexts
        last_dist += 1
        val = next_val()
        val_queue.append(val)
        pval = val_queue.popleft()
        new_selection = False
        if val == last_val:
            last_count += 1
        if val < last_val:  # new smallest k-mer at last window
            last_dist = 0
            last_val = val
            last_count = 1
            new_selection = True
        elif pval == last_val:  # popping a minimal k-mer
            last_count -= 1
            if last_count == 0:  # brand new minimal k-mer
                last_val = min(val_queue)
                last_dist = None
                last_count = 0
                for j in range(w):
                    if val_queue[w - j - 1] == last_val:
                        if (last_dist is None) or (not robust_windows):
                            last_dist = j
                        last_count += 1
                new_selection = True
            else:  # still the same minimal k-mer, now determine which k-mer to pick
                if last_dist == w:  # the k-mer selected is out of window
                    last_dist = None
                    for j in range(w):
                        if val_queue[w - j - 1] == last_val:
                            if (last_dist is None) or (not robust_windows):
                                last_dist = j
                    new_selection = True
                else:  # the k-mer selected is still in the window, nothing changes
                    assert last_dist < w
                    assert robust_windows
        else:  # no new smallest k-mer, nor
            pass
        ret += int(new_selection)
        if ret_array:
            ret_vals.append(int(new_selection))
    if ret_array:
        return ret_vals
    else:
        return ret / n