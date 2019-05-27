# fast-and-flexible

This repository contains the implementation of [The fast and the flexible: training neural networks to learn from small data](https://arxiv.org/abs/1809.06194) paper.

## dataset

### offline-dataset
The folder contains train, dev, and test files of our synthetic generated dataset used during our offline training regime. Train, dev, test contains non-overlapping utterances and block configurations. 

### word-recovery
The folder contains the dataset used in the word recovery experiments. The name of the file indicates the original words that are being replaced. For example, "remove_brown.txt" consists of utterances which contains the perturbed version of the original word of "remove" and "brown". The perturbed version are "rmv" and "braun", respectively.

### wang's
The folder contains the dataset which was initially introduced by [(Wang et.al., 2016)](https://arxiv.org/abs/1606.02447).

## implementation
The implementation mostly borrows from [i-machine-think](https://github.com/i-machine-think) repository with some modification.
Note that this is an ongoing re-implementation of the [original code](https://github.com/rezkaaufar/learning-to-follow-instructions).
Hence some features might not be available yet. Please refer to the original code for more details. 
Use the help command for full details of what you can do with the code.

TO-DO:
* Bidirectional RNN
* Online training