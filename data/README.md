# TRI AI Saturdays — Team Hwange
## Final Unified Toxicity Dataset

This package contains the final unified dataset used in the latent probing
toxicity detection project.

### Source datasets

| Dataset | Samples |
|---|---:|
| HateXplain | 19,229 |
| ToxiGen | 9,900 |
| Ubuntu | 9,000 |
| AfriHate | 4,958 |
| **Total** | **43,087** |

### Languages

- English: 33,629
- Amharic: 9,458

### Labels

- Harmful (1): 23,737
- Non-harmful (0): 19,350

### Files

#### canonical.parquet
The unified 43,087-row dataset using a common schema across all four
source datasets.

#### canonical_split.parquet
The same unified dataset with the leakage-safe experimental split assignment.

Split sizes:

- Train: 27,639
- Validation: 2,666
- Test: 3,782
- Ubuntu probe: 9,000

### Sources

- HateXplain
- ToxiGen
- Ubuntu
- AfriHate

The original source datasets remain attributed to their respective
authors/providers. This package contains the unified project-level
representation used for the experiments.

### Project

TRI AI Saturdays — Team Hwange

Project:
Early Detection of Toxic Language through Latent Probing of Large Language Models
