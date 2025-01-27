# Highway Networks for Character-Level GPT

This repository contains experimental code for applying highway networks to GPT architecture for character-level prediction tasks.

## Overview

The project explores the application of highway networks within the GPT architecture for character-level prediction tasks. The implementation modifies the standard GPT2 architecture by incorporating highway layers.

## Dependencies

- Python 3.10
- PyTorch 2.20
- Weights & Biases (Wandb)

## Model Sources

- GPT2 base model: [placeholder]
- enwiki8 preprocessing code: [placeholder2]

## Setup

Before running the training code, make sure to:
1. Install all required dependencies
2. Add your Wandb API key to the environment

## Code Structure

- `train.py`: Main training script
- `extractor.py`: Contains code for visualizing sigmoid and softmax outputs

## Running the Model

To start training, use:

```bash
python train.py "+wandb.name=test-run" "model=gpt2" "model.block_type=highway"
```

This will initialize a new Wandb run named "test-run" using the GPT2 model with highway blocks.