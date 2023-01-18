from utils.util import *

class SequenceDataset(Dataset):
    def __init__(self, vocab, path):
        self.vocab = vocab
        self.character_to_idx = {vocab[i]: i for i in range(len(vocab))}
        lines = open(path, 'r').readlines()
        sequence = ''.join([line.strip() for line in lines])
        bar = tqdm(sequence)
        bar.set_description_str('Loading Seq Data')
        self.seq = torch.tensor([
            self.character_to_idx[c]
            for c in bar
        ]).view(1, 1, -1)

    def create_fragments(self, frag_length):
        self.frag_length = frag_length
        self.seq = self.seq.unfold(-1, frag_length, 1).squeeze()

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx):
        return self.seq[idx]

    def sample_batch(self, batch_size):
        if batch_size > self.__len__():
            batch = random.choices(range(self.__len__()), k=batch_size)
        else:
            batch = random.sample(range(self.__len__()), batch_size)
        fragments = torch.stack([
            self.__getitem__(item) for item in batch
        ], dim=0)
        return F.one_hot(fragments, len(self.vocab)).transpose(1, 2).float()

class RawSequenceDataset(Dataset):
    def __init__(self, vocab, path):
        self.vocab = vocab
        self.character_to_idx = {vocab[i]: i for i in range(len(vocab))}
        lines = open(path, 'r').readlines()
        self.sequence = list(''.join([line.strip() for line in lines]))

    def create_fragments(self, frag_length):
        self.frag_length = frag_length
        self.seq = [
            self.sequence[i: min(i + frag_length, len(self.sequence))]
            for i in range(0, len(self.sequence), frag_length)
        ]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return self.seq[idx]
