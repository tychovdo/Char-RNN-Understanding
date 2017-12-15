# Understanding Character-level RNN-LMs in PyTorch

This repository contains a PyTorch implementation for Recurrent Neural Network Language Models (RNN-LMs). Visualization of hidden RNN layers (see [this](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) blog post by Andrej Karpathy). Analysis using Diagnostic Classifier ([Diagnostic Classifiers Revealing how Neural Networks Process Hierarchical Structure](http://ceur-ws.org/Vol-1773/CoCoNIPS_2016_paper6.pdf))

### Example usage

Train a simple model

```
python main.py data/tiny-shakespeare/train.txt data/tiny-shakespeare/test.txt models/simple.model --cuda
```

Example analyis can be found in the iPython Notebooks

### Analysis preview

Some previews of what a hypothesis test by a diagnostic classifier might look like:

![Example analysis 1](https://raw.githubusercontent.com/tychovdo/char-rnn-visualization/master/plots/ex1.png)
![Example analysis 1](https://raw.githubusercontent.com/tychovdo/char-rnn-visualization/master/plots/ex2.png)

### Requirements

- `python=3.6.3`
- `pytorch=0.2.0`

### Acknowledgements

- [Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078) by Karpathy, et. al (2015)
- [Diagnostic Classifiers Revealing how Neural Networks Process Hierarchical Structure](http://ceur-ws.org/Vol-1773/CoCoNIPS_2016_paper6.pdf) by Dieuwke Hupkes, et al. (2016)
- [Example PyTorch character-level RNN](https://github.com/spro/char-rnn.pytorch) by Sean Robertson
