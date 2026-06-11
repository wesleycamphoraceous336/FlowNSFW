# Contributing to FlowNSFW

Thank you for your interest in contributing! This document provides guidelines for contributing to FlowNSFW.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- Be respectful and considerate
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other contributors

### Unacceptable Behavior

- Harassment, discrimination, or trolling
- Publishing private information without consent
- Any conduct that could be considered unprofessional

---

## Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub
git clone https://github.com/YOUR_USERNAME/FlowNSFW.git
cd FlowNSFW

# Add upstream remote
git remote add upstream https://github.com/vmoranv/FlowNSFW.git
```

### Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install ruff pytest

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

---

## Development Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `perf/` - Performance improvements
- `refactor/` - Code refactoring

### 2. Make Changes

- Write clear, concise commit messages
- Keep commits focused (one logical change per commit)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Lint
ruff check src/

# Run smoke test
python -c "
import sys
sys.path.insert(0, 'src')
from flow_nsfw import FlowNSFW
model = FlowNSFW(dim=64, num_heads=2, num_temporal_layers=1)
print('OK')
"
```

---

## Pull Request Process

### Before Submitting

- [ ] Code passes lint checks (`ruff check src/`)
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] Branch is up-to-date with main

### Commit Message Format

```
type(scope): subject

body (optional)

footer (optional)
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

**Examples**:
```
feat(model): add INT8 quantization support

- Implement quantization-aware training
- Add post-training quantization script
- Update README with quantization guide

Closes #42
```

```
fix(flow): resolve correlation memory leak

The correlation layer was not releasing GPU memory properly.
Added explicit tensor cleanup after each forward pass.

Fixes #38
```

### Submitting

1. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request on GitHub

3. Fill out the PR template completely

4. Wait for review and address feedback

### PR Review Criteria

- Code quality and style
- Test coverage
- Documentation completeness
- Performance impact
- Breaking changes (major version bump required)

---

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints where possible
- Max line length: 100 characters
- Use `ruff` for linting

```python
# Good
def compute_flow(
    frames: Tensor,
    radius: int = 4,
) -> tuple[Tensor, Tensor]:
    """Compute optical flow between consecutive frames.
    
    Args:
        frames: (B, T, C, H, W) video frames
        radius: Correlation search radius
        
    Returns:
        flow_fwd: (B, T-1, 2, H, W) forward flow
        flow_bwd: (B, T-1, 2, H, W) backward flow
    """
    pass
```

### Documentation

- All public functions need docstrings
- Use Google-style docstrings
- Include examples for complex functions

```python
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 30,
) -> dict[str, float]:
    """Train FlowNSFW model.
    
    Args:
        model: FlowNSFW model instance
        dataloader: Training data loader
        epochs: Number of training epochs
        
    Returns:
        Training metrics dict with keys:
        - 'loss': Final training loss
        - 'accuracy': Final accuracy
        
    Example:
        >>> model = FlowNSFW(dim=128)
        >>> loader = DataLoader(dataset, batch_size=2)
        >>> metrics = train_model(model, loader, epochs=30)
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    """
    pass
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=src/flow_nsfw tests/
```

### Writing Tests

```python
# tests/test_model.py
import torch
from flow_nsfw import FlowNSFW

def test_model_forward():
    """Test model forward pass."""
    model = FlowNSFW(dim=64, num_heads=2, num_temporal_layers=1)
    frames = torch.randn(1, 4, 3, 160, 160)
    
    with torch.no_grad():
        out = model(frames)
    
    assert "video_cls" in out
    assert out["video_cls"].shape == (1, 3)
    assert len(out["decoded"]) == 4  # 4 scales
```

---

## Documentation

### What to Document

- **README.md**: High-level overview, quick start
- **ARCHITECTURE.md**: Model architecture details
- **THEORY.md**: Mathematical foundations
- **Docstrings**: All public APIs

### Documentation Updates Required

- New features → Update README and relevant guides
- API changes → Update docstrings
- Performance improvements → Update benchmarks
- Bug fixes → Update CHANGELOG.md

---

## Release Process

(For maintainers only)

1. Update version in `src/flow_nsfw/__init__.py`
2. Update CHANGELOG.md
3. Create release branch: `git checkout -b release/v1.1.0`
4. Tag release: `git tag -a v1.1.0 -m "Release v1.1.0"`
5. Push: `git push origin v1.1.0`
6. Create GitHub Release with notes
7. Upload model weights to release assets

---

## Questions?

- **General Questions**: Open a [Discussion](https://github.com/vmoranv/FlowNSFW/discussions)
- **Bug Reports**: Open an [Issue](https://github.com/vmoranv/FlowNSFW/issues/new?template=bug_report.yml)
- **Feature Requests**: Open an [Issue](https://github.com/vmoranv/FlowNSFW/issues/new?template=feature_request.yml)

---

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Paper acknowledgments (if applicable)

Thank you for contributing to FlowNSFW! 🚀
