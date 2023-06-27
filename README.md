#  Abstract Compositional Transformer (ACT)

This repository provides an implementation  Abstract Compositional Transformer (ACT).
It was developed for research purposes by the Neurosymbolic Systems Lab at Poznan University of Technology.

[Live demo](https://act-demo.herokuapp.com/)

![example](images/example.png)
![example_2](images/example_2.png)


# Installation

You need pipenv to install the package.  To install the package, run the following command:

```bash
pipenv install
```

# Usage
1. Train the model
```bash
python train.py
```
2. Evaluate the model
```bash
python evaluate.py --model_path <path_to_model_weights>
```
The weights are available in 
