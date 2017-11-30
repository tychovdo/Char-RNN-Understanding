# Understanding Character-level RNN in PyTorch

Repository containing
- PyTorch implementation for the RNN, LSTM and GRU language model training
- Visualization of hidden lstm layer ([Karpathy blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) of Karpathy)
- Analysis using Diagnostic Classifier ([Diagnostic Classifiers Revealing how Neural Networks Process Hierarchical Structure](http://ceur-ws.org/Vol-1773/CoCoNIPS_2016_paper6.pdf))


### Example usage

Train a simple model

```
python main.py data/tiny-shakespeare/train.txt data/tiny-shakespeare/test.txt models/simple.model --n_epochs 2000 --hidden_size 100 --cuda
```

### Requirements

- `python=3.6.3`
- `pytorch=0.2.0`

### Acknowledgements

[Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078) by Karpathy, et. al (2015)
[Diagnostic Classifiers Revealing how Neural Networks Process Hierarchical Structure](http://ceur-ws.org/Vol-1773/CoCoNIPS_2016_paper6.pdf) by Dieuwke Hupkes, et al. (2016)
[Example PyTorch character-level RNN](https://github.com/spro/char-rnn.pytorch) by Sean Robertson
