# Multi-Target Constrained Graph Attack

A framework for training GNN models for fraud detection and evaluating their robustness against adversarial attacks.

## Project Structure

```
project/
├─ data/
├─ models/
├─ results/
├─ src/
│  ├─ attack.py
│  ├─ config.py
│  ├─ dataloader.py
│  ├─ graph.py
│  ├─ models.py
│  ├─ predict.py
│  ├─ train.py
│  └─ utils.py
├─ config.yaml
├─ main.py
└─ README.md
```

## Features

- Train GNN models (GCN, GAT, GATv2, SAGE, Chebyshev) for fraud detection
- Evaluate model performance on test data
- Execute Multi-Target Constrained Graph Attacks (MTCGA) on detected fraud nodes
- Analyze attack success rates and resource usage
- Configurable via config.yaml

## Quick Start

1. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Train models**
   ```bash
   uv run python main.py train --models all --dataset mtcga
   ```

3. **Evaluate models**
   ```bash
   uv run python main.py evaluate --models GCN GAT --split test
   ```

4. **Attack detected fraud nodes**
   ```bash
   uv run python main.py attack --models Chebyshev --split test --limit 10
   ```

5. **Analyze attack results**
   ```bash
   uv run python main.py analyze --results results
   ```

## Commands

- `train` - Train GNN models on fraud detection dataset
- `evaluate` - Evaluate trained models on specified data split
- `attack` - Execute adversarial attacks on detected fraud nodes
- `analyze` - Generate summary statistics from attack results

## Configuration

Key parameters in `config.yaml`:

```yaml
# Model training
lr: 0.005
epochs: 250
hidden_units: 256

# Attack parameters
p_evasion_threshold: 0.5
max_budget_prop: 0.4
max_transformations: 10
gas_penalty_coef: 0.1
value_penalty_coef: 0.01
```
