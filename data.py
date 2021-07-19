import zipfile
import torch

def batchify(data, bsz: int):
    data = data.narrow(0, 0, (len(data) // bsz) * bsz)
    return data.view(bsz, -1).t().contiguous()

class enwik8:
    def __init__(self, bsz, device):
        with zipfile.ZipFile('enwik8.zip') as zf:
            data = zf.read('enwik8')
        tokens = [0] + sorted(set(data) - {ord('\n')})
        self.n_token = len(tokens)
        self.idx2token = {i: t for i, t in enumerate(tokens)}
        self.token2idx = {t: i for i, t in enumerate(tokens)}
        self.token2idx[ord('\n')] = 0
        # self.n_token = 256

        num_test_chars = 5000000
        cut = -2 * num_test_chars
        self.train_data = self.text_batch(data[:cut], bsz).to(device)
        self.valid_data = self.text_batch(data[cut:cut + num_test_chars], bsz).to(device)
        self.test_data  = self.text_batch(data[-num_test_chars:], bsz).to(device)

    def text_batch(self, s: bytes, bsz: int) -> torch.Tensor:
        return batchify(self.encode(s), bsz)

    def encode(self, s: bytes) -> torch.Tensor:
        # return torch.tensor([c if c != b'\n' else 0 for c in b'\n' + s])
        return torch.tensor([self.token2idx[c] for c in b'\n' + s])

    def decode(self, logits: torch.Tensor) -> bytes:
        # return bytes([seq.item() for seq in logits])
        return bytes([self.idx2token[seq.item()] for seq in logits])
