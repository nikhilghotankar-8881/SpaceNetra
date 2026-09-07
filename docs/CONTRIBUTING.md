# 🤝 Contributing to SpaceNetra

> Guidelines for contributing to the Satellite Intelligence Engine project

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Branch Strategy](#branch-strategy)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (for training)
- Docker & Docker Compose (for deployment)
- Node.js 18+ (for frontend)
- Git

### Development Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd SpaceNetra

# 2. Create Python virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Download LEVIR-CD dataset (for model development)
python scripts/download_levir_cd.py --output data/levir_cd/

# 5. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Code Style

### Python

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use docstrings for all public functions/classes

```python
def detect_changes(
    t1_image: torch.Tensor,
    t2_image: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Detect changes between two temporal satellite images.

    Args:
        t1_image: Tensor of shape (B, C, H, W) for time T1.
        t2_image: Tensor of shape (B, C, H, W) for time T2.
        threshold: Probability threshold for binary classification.

    Returns:
        Binary change mask of shape (B, 1, H, W).
    """
    ...
```

### JavaScript/React

- Use functional components with hooks
- Use ESLint + Prettier
- Prefer named exports

---

## Branch Strategy

```
main          ──────────────────────────────────── (stable releases)
                │           │           │
develop       ──┼───────────┼───────────┼──────── (integration)
                │     │     │     │     │
feature/*     ──┘     │     └─────┘     │
                      │                 │
bugfix/*      ────────┘                 │
                                        │
hotfix/*      ──────────────────────────┘
```

| Branch | Purpose |
|--------|---------|
| `main` | Stable, deployable code |
| `develop` | Integration branch |
| `feature/<name>` | New features |
| `bugfix/<name>` | Bug fixes |
| `hotfix/<name>` | Critical production fixes |

---

## Commit Messages

Use conventional commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Use |
|------|-----|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `refactor` | Code restructuring |
| `test` | Test additions/changes |
| `perf` | Performance improvement |
| `chore` | Build/tooling changes |

### Examples

```
feat(model): add ChangeFormer architecture
fix(data): handle corrupted LEVIR-CD images gracefully
docs(api): add search endpoint documentation
test(training): add loss function unit tests
perf(inference): optimize batch processing for 2x speedup
```

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_model.py

# With coverage
pytest --cov=src tests/

# Quick smoke test
pytest tests/ -m "not slow"
```

### Test Categories

| Category | Directory | Markers |
|----------|----------|---------|
| Unit tests | `tests/test_*.py` | default |
| Integration | `tests/integration/` | `@pytest.mark.integration` |
| Slow tests | Various | `@pytest.mark.slow` |
| GPU tests | Various | `@pytest.mark.gpu` |

### What to Test

- [ ] Model forward pass with expected input/output shapes
- [ ] Loss function behavior with edge cases
- [ ] Dataset loading and augmentation
- [ ] API endpoint responses
- [ ] Metric calculations
- [ ] Preprocessing steps

---

## Documentation

### Where to Document

| What | Where |
|------|-------|
| Architecture decisions | `docs/ARCHITECTURE.md` |
| API endpoints | `docs/API_REFERENCE.md` |
| Training procedures | `docs/MODEL_TRAINING.md` |
| Data pipeline | `docs/DATA_PIPELINE.md` |
| Deployment | `docs/DEPLOYMENT.md` |
| Security | `docs/SECURITY.md` |
| Code documentation | Inline docstrings |

### Documentation Requirements

- All public functions must have docstrings
- All API endpoints must be documented
- Architecture changes must update `ARCHITECTURE.md`
- Model changes must update `MODEL_TRAINING.md`
- New dependencies must be documented

---

## Project Priorities

When contributing, align with the current development milestone:

1. 🔴 **Milestone 1**: Change Detection AI (highest priority)
2. 🟠 **Milestone 2**: Real-World Satellite Data
3. 🟡 **Milestone 3**: Semantic Search
4. 🟢 **Milestone 4**: Intelligence Engine
5. 🔵 **Milestone 5**: Full Application

See [docs/ROADMAP.md](ROADMAP.md) for detailed phase descriptions.
