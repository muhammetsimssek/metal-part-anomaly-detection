# Clean-environment inference validation

Validated on **13 October 2026** using a new virtual environment on Windows x64 and Python 3.12.14. The environment does not inherit system site packages. Pretrained weights were downloaded into a separate empty Torch cache.

[Setup](SETUP.md) · [Project overview](../README.md) · [Pinned dependencies](../requirements-lock.txt)

## Executed checks

Both actual pretrained models built reference memory banks, performed inference on two test photographs, and saved two readable overlay images each. The input subset consisted of **two reference and two test photographs** copied from the original local project into an isolated workspace. Original source files and input photographs remained unchanged.

For every image, the anomaly map matched the input dimensions and contained finite values; the image score was finite. These checks confirm that the model pipelines execute in this environment. They do not measure defect-detection accuracy.

| Pipeline | References | Tests / readable outputs | Memory-bank shape | Result |
|---|---:|---:|---|---|
| full-memory | 2 | 2 / 2 | 4608 × 1536 | Passed |
| dinov2 | 2 | 2 / 2 | 10240 × 2304 | Passed |

## Dependency versions

| Package | Validated version |
|---|---|
| torch | 2.14.1+cpu |
| torchvision | 0.29.1+cpu |
| numpy | 2.5.2 |
| opencv-python | 5.0.0.93 |
| Pillow | 12.3.0 |
| tqdm | 4.70.1 |

Direct and transitive package versions are pinned in `requirements-lock.txt`. The root `requirements.txt` includes that lockfile. CPU PyTorch wheels are obtained from the official PyTorch wheel index.

The published `requirements.txt` was then installed into a second empty virtual environment. All 16 pinned package versions matched, model/image libraries imported successfully, and `pip check` reported no broken requirements. SHA-256 checks also confirmed that all 469 previously recorded original project files remained unchanged.

## Model-source reproducibility

DINOv2 Hub code is pinned to commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` of `facebookresearch/dinov2`. TorchVision's WideResNet50-2 experiment explicitly selects `IMAGENET1K_V1` pretrained weights. Model weights are downloaded at runtime and are not redistributed in this repository.

The publication copy adds the DINOv2 source pin; feature extraction, memory-bank construction, distance scoring and visualization remain unchanged. The original local experiment files were preserved.

## Scope

This is a small integration check, not a repeat of all historical experiments, a speed benchmark or a labelled quality evaluation. The historical visual comparisons were generated with different reference collections; their score values should not be compared with this subset. Acceptance/rejection calibration and production validation remain outside the demo.

The lockfile and inference were validated on Windows/Python 3.12/CPU. Other operating systems, Python versions and CUDA environments are not claimed as verified.
