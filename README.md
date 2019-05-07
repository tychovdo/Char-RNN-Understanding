# Understanding Character-level RNN-LMs in PyTorch

This repository contains a PyTorch implementation for Recurrent Neural Network Language Models (RNN-LMs).

For more information, check out this [blogpost](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy and this [paper](https://arxiv.org/abs/1711.10203) about Analysis using Diagnostic Classifier by Dieuwke Hupkes et. al.

### Paper

Paper: [Quantitatively Understanding Recurrent Networks](https://raw.githubusercontent.com/tychovdo/Char-RNN-Understanding/master/Quantitatively_understanding_recurrent_networks.pdf)

### Example Usage

Train a simple model

```
python main.py data/tiny-shakespeare/train.txt data/tiny-shakespeare/test.txt models/simple.model --cuda
```

Example analysis can be found in the iPython Notebooks

### Diagnostic Classifiers

Diagnostic classifiers can be used to verify hypotheses about information in the hidden representations of a recurrent networks.
The figures below show how how a diagnostic classifier verifies that the network captures the position in line and whether characters are part of a comment.

![Example analysis 1](https://raw.githubusercontent.com/tychovdo/char-rnn-visualization/master/plots/ex1.png)
![Example analysis 2](https://raw.githubusercontent.com/tychovdo/char-rnn-visualization/master/plots/ex2.png)

### Most Responsible Neuron

Diagnostic classifiers can also used to automatically find neurons encoded to perform specific subtasks.
The figure below shows a neuron that is active inside quotation marks.

![Most responsible neuron](https://raw.githubusercontent.com/tychovdo/char-rnn-visualization/master/plots/ex4.png)

### Requirements

- `python=3.6.3`
- `pytorch=0.2.0`

### Acknowledgements

- [Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078) by Karpathy, et. al (2015)
- [Visualisation and 'diagnostic classifiers' reveal how recurrent and recursive neural networks process hierarchical structure](https://arxiv.org/abs/1711.10203) by Dieuwke Hupkes, et al. (2016)
- [Example PyTorch character-level RNN](https://github.com/spro/char-rnn.pytorch) by Sean Robertson
