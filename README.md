# Getting started with Adversarial Examples

This repository provides a basic classification model for CIFAR-10 as well as functions for loading the dataset and model. I have also provided some useful utility functions for testing your results.

### Setting Up

1. Install [Anaconda](https://www.anaconda.com/) if you have not already
2. Create a new environment for torch
```python
conda create -n torch
conda activate torch
```
3. Get the correct version of [Torch](https://pytorch.org/get-started/locally/), for me this looks like
```python
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```
4. Install other packages like numpy, tqdm, jupyter, etc
```python
conda install numpy, tqdm, jupyter
```

### Running

I like jupyter notebooks so I have provided [adv.ipynb](code/adv.ipynb), but if you don't want to use it [adv.py](code/adv.py) is the same thing.

### Functions Provided

- `project_lp` - I have provided this function which takes an input and projects it back to an $l_p$-norm of $\xi$
- `accuracy` and `asr` - Accuracy provides accuracy on a batch, ASR computes the adversarial success rate on an entire dataset (or 1 - accuracy)
- `show_attack` - Used to visualize your results

### Useful Reading

- [Intriguing Properties of Neural Networks](https://arxiv.org/pdf/1312.6199.pdf) - The first paper to discuss adversarial examples, a good read to understand basic ideas.
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1312.6199.pdf) - This paper introduces FGSM, has good ideas about Adversarial Examples
- [Towards Deep Learning Models Resistant to Adversrial Attacks](https://arxiv.org/pdf/1706.06083.pdf) - This paper introduces PGD, strong paper that extends previous works
