import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def evaluate(model, crit, batches):
    model.eval()
    hidden = mem = None
    with torch.no_grad():
        postfix = {}
        total_loss = 0
        mem = hidden = None
        pbar = tqdm(desc='eval', total=len(batches) // bptt, postfix=postfix)
        for i in range(0, len(batches), bptt):
            chunk = batches[i:i+1+bptt]
            x, target = chunk[:-1], chunk[1:]
            y, mem, hidden = model(x, mem, hidden)
            loss = crit(y.flatten(end_dim=1), target.flatten())
            total_loss += loss.item()
            # progress bar
            pbar.update(1)
            cur_loss = total_loss / pbar.n
            postfix['loss'] = f"{cur_loss:.3f}"
            if cur_loss < 20:
                postfix['ppl'] = f"{math.exp(cur_loss):.3f}"
                postfix['bpc'] = f"{cur_loss / math.log(2):.3f}"
            pbar.set_postfix(postfix)
        pbar.close()
    return total_loss / pbar.n

def train(model, crit, optim, sched, dataset, epochs):
    for i in range(epochs):
        model.train()
        batches = dataset.train_data
        postfix = {'lr': optim.param_groups[0]['lr']}
        total_loss = 0
        pbar = tqdm(desc=f"train[{i+1}]", total=len(batches) // bptt, postfix=postfix)
        while True:
            seq_len = random.randint(bptt - 5, bptt + 5)
            if i + seq_len > len(batches):
                break
            chunk = batches[i:i+1+seq_len]
            x, target = chunk[:-1], chunk[1:]
            i += seq_len

            y, _, _ = model(x)
            loss = crit(y.flatten(end_dim=1), target.flatten())
            # loss = 0
            # for j in range(len(x)):
            #     y, mem, hidden = model.forward(x[j].unsqueeze(0), mem, hidden)
            #     loss += crit(y[-1], target[j])

            if False:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optim) # for clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                scaler.step(optim)
                scaler.update()
            optim.zero_grad()

            total_loss += loss.item()
            # progress bar accounting
            pbar.update(1)
            cur_loss = total_loss / pbar.n
            postfix['loss'] = f"{cur_loss:.3f}"
            if cur_loss < 20:
                postfix['ppl'] = f"{math.exp(cur_loss):.3f}"
                postfix['bpc'] = f"{cur_loss / math.log(2):.3f}"
            pbar.set_postfix(postfix)
        pbar.close()
        val_loss = evaluate(model, crit, dataset.valid_data)
        sched.step(val_loss)
        with open('model.pt', 'wb') as f:
            torch.save(model, f)

if __name__ == '__main__':
    from tqdm import tqdm
    from model import SHARNN
    from data import enwik8

    fresh = True
    cuda = True
    distributed = False
    bsz = 16
    epochs = 40
    bptt = 1024
    device = 'cuda' if cuda else 'cpu'
    if distributed:
        torch.distributed.init_process_group(backend='nccl')
        rank = torch.distributed.get_rank()
        torch.cuda.set_device(rank)

    dataset = enwik8(bsz=bsz, device=device)
    if not fresh:
        with open('model.pt', 'rb') as f:
            model = torch.load(f)
    else:
        model = SHARNN(n_token=dataset.n_token, embed_dim=1024, hidden_dim=4096, n_layers=4, heads=1, max_len=5000, dropout=0.1, tied=True)
    model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, dim=1, find_unused_parameters=True)

    # optim = torch.optim.Adam(model.parameters(), lr=0.002)
    from pytorch_lamb import Lamb
    optim = Lamb(model.parameters(), lr=0.002, min_trust=0.25)

    crit = nn.CrossEntropyLoss().to(device)
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=2)
    scaler = torch.cuda.amp.GradScaler()

    if True:
        train(model, crit, optim, sched, dataset, epochs)

    test_loss = evaluate(model, crit, dataset.test_data)
    print(f"Test | loss {test_loss:.3f} | ppl {math.exp(test_loss):.3f} | bpc {test_loss / math.log(2):.3f}")
    exit()
