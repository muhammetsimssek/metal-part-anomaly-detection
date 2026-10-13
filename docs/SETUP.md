# Setup and execution

[Project overview](../README.md) · [Türkçe açıklama](../README_TR.md)

The following CPU setup was validated in a clean Windows x64 virtual environment with Python 3.12.14. Both pretrained models completed reference-memory construction and inference on a two-reference/two-test integration subset. See [VALIDATION.md](VALIDATION.md) for the exact scope. Package versions are pinned in `requirements-lock.txt`; the original historical environment is unknown.

## 1. Create an isolated Python environment

Use Python 3.12 for the validated environment. The lockfile installs the matching CPU PyTorch/TorchVision pair and all remaining packages. For a different Python/CUDA configuration consult the [PyTorch installer](https://pytorch.org/get-started/locally/); that configuration is outside this validation.

```powershell
git clone https://github.com/muhammetsimssek/metal-part-anomaly-detection.git
cd metal-part-anomaly-detection
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For Linux/macOS use `.venv/bin/python` instead. Both experiment scripts currently select `torch.device("cpu")` regardless of CUDA availability.

## 2. Supply your own images

Create these directories in the repository root:

```text
metal-part-anomaly-detection/
  train_data/                 # reference photographs of acceptable parts
  test_data/                  # new photographs to inspect
  experiments/               # original experiment scripts
  run.py                     # input checks and experiment launcher
```

Supported file extensions: `.jpg`, `.jpeg`, `.png`, `.bmp` and their fully uppercase equivalents. Rename mixed-case extensions such as `.Jpg` before running the original experiments. Use one consistent part type and camera setup, with separate reference/test captures. Start with a small number of readable RGB photographs. Both folders must contain images before running a script. Do not use the comparison boards as input: they already contain text and heatmap overlays.

The factory's full reference/test dataset and inspection booklet are intentionally excluded. The repository provides selected visual results for reviewing the work.

## 3. Run an experiment from the repository root

```powershell
.venv\Scripts\python.exe run.py full-memory
.venv\Scripts\python.exe run.py dinov2
```

Run either command separately to compare approaches. Initial execution needs internet access: TorchVision downloads pretrained WideResNet50-2 weights; DINOv2 is loaded from the official `facebookresearch/dinov2` PyTorch Hub repository and downloads its weights. The published DINOv2 script pins the Hub source to commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.

| Script | Output directory | Filename prefix |
|---|---|---|
| `patchcore_full_memory.py` | `artifacts/full_memory_output_New/` | `FULL_MEM_` |
| `dinov2_patchcore_v4.py` | `artifacts/dinov2_v4_reg_rot_pclip_22/` | `DINOV2_V4_` |

The suffix `_22` is a historical directory name, not a restriction on input count. Repeated runs overwrite matching output filenames. The output images display an overlay and an image score; no acceptance/rejection decision is produced.

## Interpretation and practical limits

- Reference memory is rebuilt on every run; pretrained backbone weights are not fine-tuned.
- Full Memory uses a **48×48** feature grid. Older comments mentioning 56 or 64 do not describe the current numeric setting.
- DINOv2 concatenates intermediate layers 5, 8 and 11 and uses a **32×32** grid with five aligned views.
- Full reference memory and pairwise comparisons can use substantial RAM, especially with DINOv2's five views. Runtime/memory benchmarks are not available.
- The launcher rejects missing/empty image directories before importing model dependencies. The experiment scripts still assume readable images; unreadable files can cause errors.
- Scores are relative distance-based anomaly measures, not probabilities or accuracy percentages. Calibrate a threshold on representative labelled validation data before using them for part rejection.

## Türkçe kısa kullanım

Depoyu klonlayın, sanal ortam oluşturup bağımlılıkları yükleyin. Kendi sağlam referans fotoğraflarınızı `train_data`, incelenecek fotoğraflarınızı `test_data` klasörüne koyun. Depo kökünden `python run.py full-memory` veya `python run.py dinov2` komutunu kullanın. Çıktılar `artifacts` altında oluşur; aynı isimli eski çıktılar üzerine yazılır. Hazır veri seti ve model ağırlıkları depoya dahil değildir. Python 3.12/Windows/CPU temiz ortamında iki referans ve iki test fotoğrafıyla çıkarım doğrulandı; kapsam `VALIDATION.md` dosyasında açıklanır.
