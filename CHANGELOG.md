# Changelog

All notable changes to FlowNSFW will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-11

### Added

- Initial release of FlowNSFW
- Core model architecture:
  - UNetEncoder for RGB feature extraction
  - FlowNet with optimized correlation layer
  - Mamba SSM temporal aggregation (O(N) complexity)
  - Multi-scale detection head (stride 1/2/4/8)
  - Video-level classifier
- Three-tier SSM backend fallback:
  1. `mamba-ssm` CUDA kernels (fastest)
  2. HuggingFace Mamba2 (PyTorch)
  3. Fallback SSM (pure PyTorch, always available)
- Training features:
  - Multi-scale training with random resolution sampling
  - Balanced batch sampler for class balance
  - Five loss functions: detection, video_cls, temporal_smooth, flow_consistency, flow_smoothness
  - BF16 mixed precision training
- Inference features:
  - Sliding window inference (8-frame clip, 4-frame stride)
  - Automatic resolution adjustment based on GPU VRAM
  - Native resolution support (no forced resize)
- Scripts:
  - `scripts/train.py`: Training script with full hyperparameter control
  - `scripts/infer.py`: Inference on video frame directories
  - `scripts/eval_multi_res.py`: Multi-resolution evaluation
  - `scripts/bench_full.py`: 4-model comparison benchmark
- Documentation:
  - README.md with quick start and architecture overview
  - ARCHITECTURE.md with detailed module explanations
  - THEORY.md with mathematical foundations
  - QUICKSTART.md with one-liner setup

### Performance

- 96.4% accuracy on 224-video test set
- 98.3% NSFW recall (only 2 false negatives)
- 94.0% SFW accuracy (only 6 false positives)
- 411ms average inference time per video
- 5.22M parameters (83.7MB FP32)

### Benchmarks

vs. YOLOv11 v16_s: +26.4% accuracy (70.0% → 96.4%)
vs. Traditional ML (SVM+HOG): +41.0% accuracy (55.4% → 96.4%)

## [Unreleased]

### Planned

- INT8 quantization (target: 20MB model, 2× speedup)
- TensorRT export for edge deployment
- ONNX export for cross-platform deployment
- Extended clip length support (8 → 32 frames)
- Audio+Video multi-modal detection
- Weakly-supervised learning pipeline

---

**Legend**:
- `Added` - New features
- `Changed` - Changes in existing functionality
- `Deprecated` - Soon-to-be removed features
- `Removed` - Removed features
- `Fixed` - Bug fixes
- `Security` - Security fixes

[1.0.0]: https://github.com/vmoranv/FlowNSFW/releases/tag/v1.0.0
