# CATALux

A simple framework for training and evaluating Graph Neural Network (GNN) models on custom graph datasets.

## Project Structure

```
catalux/
├─ data/
│  ├─ etfd/dataset.csv
│  └─ ammari/dataset.csv
├─ main.py
├─ src/
│  ├─ models.py
│  ├─ utils.py
│  ├─ train.py
│  └─ loader.py
├─ config.yaml
├─ requirements.txt
└─ README.md
```

## Features

- Supports multiple datasets (`ammari`, `etfd`)
- Cross-validation and single-run modes
- Easily configurable via config.yaml
- Results saved to CSV

## Quick Start

1. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

2. **Prepare data**  
   Place your dataset CSVs in data/ammari/dataset.csv or data/etfd/dataset.csv.

3. **Configure settings**  
   Edit config.yaml to adjust model and training parameters.

4. **Run training**  
   ```
   python main.py --dataset ammari
   ```
   or
   ```
   python main.py --dataset etfd
   ```

5. **Results**  
   Results are saved in the `results/` directory as a CSV file.

## Configuration (config.yaml)

Example:
```yaml
num_classes: 2
hidden_units: 128
chebyshev_k: [1, 2]
lr: 9e-3
epochs: 3000
weight_decay: 5e-4
early_stopping_patience: 100
num_folds: 5
device: auto
seed: 42
```

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies
