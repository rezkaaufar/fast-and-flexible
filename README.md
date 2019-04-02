# fast-and-flexible

This repository contains the dataset used in [The fast and the flexible: training neural networks to learn from small data](https://arxiv.org/abs/1809.06194).

## offline-dataset
The folder contains train, dev, and test files of our synthetic generated dataset used during our offline training regime. Train, dev, test contains non-overlapping utterances and block configurations. 

## word-recovery
The folder contains the dataset used in the word recovery experiments. The name of the file indicates the original words that are being replaced. For example, "remove_brown.txt" consists of utterances which contains the perturbed version of the original word of "remove" and "brown". The perturbed version are "rmv" and "braun", respectively.

## wang's
The folder contains the dataset which was initially introduced by [(Wang et.al., 2016)](https://arxiv.org/abs/1606.02447).
