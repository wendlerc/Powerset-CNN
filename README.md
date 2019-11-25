# Powerset Convolutional Neural Networks

## Abstract 
We present a novel class of convolutional neural networks (CNNs) for set functions, i.e., data indexed with the powerset of a finite set. The convolutions are derived as linear, shift-equivariant functions for various notions of shifts on set functions. The framework is fundamentally different from graph convolutions based on the Laplacian, as it provides not one but several basic shifts, one for each element in the ground set. Prototypical experiments with several set function classification tasks on synthetic datasets and on datasets derived from real-world hypergraphs demonstrate the potential of our new powerset CNNs. 

## Bibtex

### NIPS (pages will be added asap)

```bibtex
@inproceedings{wendler2019powerset,
  title={Powerset convolutional neural networks},
  author={Wendler, Chris and Alistarh, Dan and P{\"u}schel, Markus},
  booktitle={Advances in Neural Information Processing Systems},
  pages={927--938},
  year={2019}
}
```

## Acknowledgements

The project structure was provided by Max Horn.

## Structure

We separate general code (in the `src` module) from code used to run experiments
(in the `exp` module).
General code represents the definition of models and neural network architectures.
Experiments are sacred scripts with configuration parameters, which then call parts of
the `general code`.

## Installation

The dependecies of the project can be installed using pipenv.
To install all dependecies required for the project run `pipenv install` in
the root project folder (containing the file `Pipfile`).

It is then possible to run commands in the project specific virtual environment using
`pipenv shell`.

Finally, as pipenv does not support multiple dedicated package configurations, it is
possible to change the default tensorflow packages with the correposnding gpu versions
(for example when it is defied to run code on a gpu server).

This can be done using the command `pipenv run install-gpu-packages`.


## Experiments

The experiments for which the results are reported in the paper can be run using the command and exchanging model and
hyperparameters appropriately:

```bash
python -m exp.train_model with model.PCN dataset.AJ
```
To train and test the 2-layer powerset convolutional neural network with shift <a href="https://www.codecogs.com/eqnedit.php?latex=s_A&space;\mapsto&space;s_{A&space;\setminus&space;\{x\}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_A&space;\mapsto&space;s_{A&space;\setminus&space;\{x\}}" title="s_A \mapsto s_{A \setminus \{x\}}" /></a> and
```bash
python -m exp.train_model with model.PCN model.signal_model=4 dataset.AJ
```
to train the 2-layer powerset convolutional neural network with shift <a href="https://www.codecogs.com/eqnedit.php?latex=s_A&space;\mapsto&space;s_{A&space;\cup&space;\{x\}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_A&space;\mapsto&space;s_{A&space;\cup&space;\{x\}}" title="s_A \mapsto s_{A \cup \{x\}}" /></a>.

Remaining model names: model.KIPF (Laplacian based graph convolution on the hypercube), model.ADJ
adjacency based graph convolution on the hypercube), model.MLP (mutli-layer perceptron).

Hyperparameters: model.pooling=pm (for pooling)
model.use_sum=True (for the average aggregation step before the MLP)

Datasets: dataset.AH (Spectral Patterns), dataset.AJ (k-Junta), dataset.AXM (coverage vs. 'almost coverage'),
dataset.DOM4 (four class domain classification), dataset.DOM6 (six class domain classification),
dataset.CON10 (congress-bills open vs. closed hyperedge), dataset.COAUTH10 (coauthorship closed hyperedge vs.
not-closed hyperedge)

In order to store the output of a training run alongside with its configuration,
we can call the sacred script using the `-F` flag which writes all of the runs output into
the folder defined by `-F`:
```bash
python -m exp.train_model with model.PCN dataset.AJ -F target_dir
```


