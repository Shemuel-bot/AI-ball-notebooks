# AI Ball Notebooks

AI Ball is a collection of notebooks for experimenting with a **Ball Trap Brain**: a
character-level, discrete symbolic model inspired by balls moving through a ring of
deflectors.

An input string is tokenized and converted into starting positions and directions.
The resulting ball trajectories encounter deflectors that bounce or emit characters.
Training aims to make that discrete behavior produce a desired output string. The
repository explores both the original ball-trap architecture and several variants,
trainers, optimizers, and analysis tools.

> [!IMPORTANT]
> This is an experimental research repository, not a finished machine-learning
> library. Notebook names and APIs vary across iterations, and different notebooks
> contain different copies of the model implementation.

## Highlights

- **Discrete character-level inference:** a toy text-in/text-out system without a
  conventional neural-network forward pass.
- **Multiple training approaches:** genetic algorithms, continuous relaxation,
  gradient-based methods, constructive routing, sparse trace search, and online
  rejection-based training.
- **Checkpoint and resumable-training experiments:** several notebooks include
  model serialization and training progress persistence.
- **Model analysis:** exact structural and SMT-based trainability analysis, sparse
  trainer benchmarks, and Transformer-to-Ball-Trap comparison experiments.
- **Colab-friendly notebooks:** most notebooks include a Google Colab launch badge
  and can be opened directly from GitHub.

## Recommended Notebooks

Start with these notebooks when exploring the repository:

| Notebook | Purpose |
| --- | --- |
| [`ai_ball.ipynb`](ai_ball.ipynb) | Broad introduction to the core `Ball_Trap_Brain`, genetic training, continuous relaxation, and council training. |
| [`Ball_Trap_Brain_constructive.ipynb`](Ball_Trap_Brain_constructive.ipynb) | Small, self-contained example that directly writes answers into deflector holes without search or gradients. |
| [`Ball_Trap_Brain_constructive_adaptive_start_positions.ipynb`](Ball_Trap_Brain_constructive_adaptive_start_positions.ipynb) | Constructive training with adaptive start positions and conflict repair. |
| [`BallTrap_Global_D_Joint_Direction_Trainer.ipynb`](BallTrap_Global_D_Joint_Direction_Trainer.ipynb) | Standalone training notebook with data splitting and global learned directions/start positions. |
| [`Ball_Trap_Sparse_Trainer.ipynb`](Ball_Trap_Sparse_Trainer.ipynb) | Trace-guided discrete search with sparse updates and exact candidate rescoring. |
| [`Ball_Trap_Brain_Online_Soft_Resistance_Rejection.ipynb`](Ball_Trap_Brain_Online_Soft_Resistance_Rejection.ipynb) | Train one example at a time with soft resistance and rejection-based updates. |
| [`Ball_Trap_Trainability_Analyzer_v2.ipynb`](Ball_Trap_Trainability_Analyzer_v2.ipynb) | Determine whether a finite dataset is structurally or exactly feasible under a declared parameter domain. |
| [`ball_trap_transformer_v4_test_notebook.ipynb`](ball_trap_transformer_v4_test_notebook.ipynb) | Compare a Transformer with an equivalent quantized Ball-Trap representation. |

## Notebook Groups

### Core and Genetic Training

The `ai_ball*.ipynb` and `capacity_change_ai_ball_7.ipynb` notebooks evolve or train
the core ball-trap model. These experiments vary:

- genetic-algorithm and batch scheduling behavior;
- continuous and straight-through optimization;
- backpropagation and relaxed-loss implementations;
- low-rank embeddings and class-imbalance handling;
- checkpointing, capacity, speed, and bounce behavior;
- council-style combinations of independently trained brains.

The GA variants under `ai_ball_steps*.ipynb` contain some of the clearest minimal
examples of the genetic-training approach.

### Constructive and Discrete Trainers

- `Ball_Trap_Brain_constructive*.ipynb` writes target characters into the existing
  routing instead of searching for parameters with gradients or a GA.
- `Ball_Trap_Sparse_Trainer.ipynb` searches the original discrete parameters while
  limiting evaluations to the trajectories and tokens affected by a candidate change.
- `Ball_Trap_Brain_Online_Soft_Resistance_Rejection.ipynb` updates the model
  transactionally from individual examples and rejects expensive or incorrect changes.
- `BallTrap_Global_D_Joint_Direction_Trainer.ipynb` learns global directions and
  start positions, rounds the continuous solution, and compiles it into a ball-trap
  representation.

### Hash and Sparse Hash Search

The `Hash_trainer*.ipynb` and `ai_ball_profiled_hash_search*.ipynb` notebooks explore
local, slot-conditioned hash routing. Variants cover memory-efficient execution,
parallel Numba/JIT search, profiled training, Hugging Face datasets, and alignment
patches.

### Analysis and Comparisons

- `Ball_Trap_Trainability_Analyzer_v2.ipynb` distinguishes impossibility, exact
  finite-domain feasibility, and optimizer failure.
- `ball_trap_transformer_v4_test_notebook.ipynb` compares model size and output
  behavior against a TinyGPT-style Transformer.
- `balltrapbrain_snapshot.ipynb` is an implementation snapshot retained for
  experiments or reproducibility.

## Getting Started

### Open in Google Colab

Open one of the recommended notebooks from GitHub and use its **Open in Colab**
badge. Colab launch badges are included throughout the repository.

### Run locally

```bash
git clone https://github.com/Shemuel-bot/AI-ball-notebooks.git
cd AI-ball-notebooks
jupyter lab
```

Open a notebook in JupyterLab and run the cells from top to bottom. Install packages
only as prompted by a notebook; these experiments are not currently exposed through
a shared `requirements.txt`.

The required dependencies vary by notebook. Common packages include:

- `numpy`
- `torch`
- `pandas`
- `matplotlib`
- `scipy`
- `tqdm`
- `transformers` and `datasets`
- `numba`
- `rapidfuzz`
- `z3-solver`

Some notebooks install dependencies themselves, while others assume a Colab or
Kaggle environment is already prepared.

## Common Training Pattern

Most core notebooks follow a similar structure:

1. Define a character alphabet and tokenizer.
2. Initialize a `Ball_Trap_Brain` with a ring capacity and routing parameters.
3. Provide `(input, target_output)` training cases.
4. Choose an optimizer or search strategy.
5. Evaluate exact or relaxed fitness.
6. Test held-out cases.
7. Save a checkpoint or export the learned representation.

The exact classes, settings, and save formats differ between notebook generations.
Use a notebook with a similar objective as the starting point and copy its
self-contained implementation if you need a standalone experiment.

## Repository Notes

- Notebook variants use names such as `_variant_1`, `_fixed`, `_corrected`, or
  `_lowmem`
  indicate iterative or corrected experiments rather than a formal versioned API.
- Ambiguous names such as `Untitled7`, parenthesized suffixes, and numbered
  snapshot duplicates were replaced with descriptive names where the notebook
  purpose could be identified.
- Several notebooks contain copied versions of the model implementation. There is
  not yet one shared Python package for all notebooks.
- Checkpoints, downloaded datasets, temporary caches, and generated files are
  experiment artifacts and may require large local storage.
- A held-out test is a much stronger indication of generalization than success on
  the training set, particularly for constructive trainers that behave as lookup
  tables.
- No license is currently specified in this repository. Review that before copying
  substantial parts of the work into another project.

## Contributing

Contributions can include:

- a new trainer or model variant;
- notebook bug fixes or reproducibility improvements;
- a smaller self-contained benchmark;
- documentation for an existing experiment;
- comparison or trainability analysis.

When adding a notebook, include a short markdown overview and example configuration
near the top, so the repository remains easy to navigate.
