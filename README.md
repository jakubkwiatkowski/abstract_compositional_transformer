# Abstract Compositional Transformer (ACT)

This repository provides an implementation Abstract Compositional Transformer (ACT).
It was developed for research purposes by the Neurosymbolic Systems Lab at Poznan University of Technology.

[Live demo](https://huggingface.co/spaces/jkwiatkowski/raven)

![example](images/example.png)
![example_2](images/example_2.png)


Code for models and utilities is available in repositories:
- [core_tools](https://github.com/jakubkwiatkowski/core_tools)
- [grid_transformer](https://github.com/jakubkwiatkowski/compositional_transformer)
- [raven_utils](https://github.com/jakubkwiatkowski/raven_tools)

# Installation

You need pipenv to install the package. To install the package, run the following command:

```bash
pipenv install
```

# Usage
Launch a new shell session with `pipenv shell` or configure the environment in your IDE.

1. Property prediction
a. Train first phase - Task tokenizer with random masking

```bash
python main.py pp --phase "train" --tokenizer "task" --masking "random" --data_split "train" --save_weights "model/act_task_random/weights" --epochs 200
```

b. Train second phase - Task tokenizer with last masking

```bash
python main.py pp --phase "train" --tokenizer "task" --masking "last" --data_split "train" --save_weights "model/act_task_both/weights" --load_weights "act_model/task_random/weights" --epochs 20
```

c. Evaluate model 

```bash
python main.py pp --phase "eval" --tokenizer "task" --masking "last" --data_split "test" --load_weights "model/act_task_both/weights"
```

2. Choice maker

a. Evaluate DCM

```bash
python main.py cm --phase "eval" --tokenizer "task" --choice_maker "dcm" --data_split "test" --act_load_weights "model/act_task_both/weights" 
```

b. Train LCM

```bash
python main.py cm --phase "train" --tokenizer "task" --choice_maker "lcm" --data_split "train" --save_weights "model/lcm_task_both/weights" --act_load_weights "model/act_task_both/weights" --epochs 200
```

c. Evaluate LCM

```bash
python main.py cm --phase "eval" --tokenizer "task" --choice_maker "lcm" --data_split "test" --load_weights "model/lcm_task_both/weights" 
```

d. Train CCM

```bash
python main.py cm --phase "train" --tokenizer "task" --choice_maker "ccm" --data_split "train" --save_weights "model/ccm_task_both/weights" --act_load_weights "model/act_task_both/weights"  --epochs 200
```

e. Evaluate CCM

```bash
python main.py cm --phase "eval" --tokenizer "task" --choice_maker "ccm" --data_split "test" --load_weights "model/ccm_task_both/weights" 
```

Weights are available hear: https://drive.google.com/drive/folders/1kxfS_QATWctTJFV3lzM99U83CnKV1zFU
