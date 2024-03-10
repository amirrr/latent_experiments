# latent_experiments

Python package for running experiments on latent spaces of neural networks

## Installation

```bash
pip install latent-experiments
```

## Usage

```python
import latent_experiments as le


# Run experiments on a simple dataset
outcome_diff_df = run_latent_experiments(
    data,
    columns_to_match=columns_to_match,
    outcome_var=outcome_var,
    match_threshold=0.8,
)
```
