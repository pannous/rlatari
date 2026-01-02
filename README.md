# rl-atari

Modern implementation of reinforcement learning algorithms for Atari games using Gymnasium.

| Algo   | Done         |
|--------|--------------|
| DQN    | ✓            |
| PPO    | ✓            |
| CURL   | ✓            |
| RAD    | ✓            |
| DrQ-v2 | ✓            |
| PUPG   | ✓            |

## Installation

```bash
# Install dependencies
pip install -e .

# Install with development tools
pip install -e ".[dev]"
pre-commit install
```

## Run

```bash
# Simple training
python train.py algo=dqn env=Alien-v5

# Training without saving models
python train.py algo=dqn env=Alien-v5 model=False

# Change other parameters
python train.py algo=dqn env=CrazyClimber-v5 algo.max_steps=2005000
```

## Development

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Clean build artifacts
make clean
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Gymnasium with Atari ROMs
