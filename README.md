Single Headed Attention RNN
---

This is a new implementation of the [Single Headed Attention RNN: Stop Thinking With Your Head](https://arxiv.org/abs/1911.11423) paper.

The original paper implementation can be found at [Smerity/sha-rnn](https://github.com/Smerity/sha-rnn).

Some notes about my implementation:

- This version uses only builtin Pytorch layers, and supports modern Pytorch and AMP.

- The parameters included are for the 1-head model, not the 4-head model outlined in the paper.

- I used Python 3.6, and installed Pytorch 1.7 and the [Smerity LAMB optimizer](https://github.com/Smerity/pytorch-lamb).

- I trained with `python3 train.py`. The final line of output should be the Test accuracy.

- My implementation appears to be around half as fast as Smerity's, taking 1h per epoch instead of 30 minutes.

- I tested with the Adam optimizer, and only got around 3.0 bpc on the train set. LAMB converged much better.

- The Test set is evaluated automatically at the end of training. Mine looks like this:

    Test | loss 0.772 | ppl 2.163 | bpc 1.113

- This shows 1.113 bpc on Test. It's slightly worse than 1.077 bpc achieved by the original implementation.

- Note that 1.113 bpc is still better than the best non-attention LSTM on [Papers With Code](https://paperswithcode.com/sota/language-modelling-on-enwiki8), which is the Mogrifier LSTM at 1.146 bpc.

- The biggest technical difference between our implementations, barring any bugs, is the Attention layer. I use the Pytorch MultiHeadAttention layer instead of custom Attention. This may be the reason for the speed difference and slightly worse bpc on my version. I also don't hardcode a seed.
