import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import glob

# ============================================================
# CONFIG - DINOv2 + PatchCore v4
#   v3'ten farklar:
#     - backbone = dinov2_vitb14_reg   (register tokens -> temiz feature)
#     - TTA = identity + hflip + rot90/180/270  (rotasyon dostu)
# ============================================================
def get_image_files(pattern_or_folder):
    files = []
    if os.path.isdir(pattern_or_folder):
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            files.extend(glob.glob(os.path.join(pattern_or_folder, ext)))
            files.extend(glob.glob(os.path.join(pattern_or_folder, ext.upper())))
    else:
        files = glob.glob(pattern_or_folder)
    return sorted(list(set(files)))

TRAIN_FILES = get_image_files("train_data") if os.path.isdir("train_data") else get_image_files("ref*.jpg")
TEST_FILES = get_image_files("test_data") if os.path.isdir("test_data") else get_image_files("test*.jpg")

print(f"Eğitim Görselleri ({len(TRAIN_FILES)} adet): {TRAIN_FILES}")
print(f"Test Görselleri ({len(TEST_FILES)} adet): {TEST_FILES}")

ARTIFACTS_DIR = Path("artifacts")
OUTPUT_DIR = ARTIFACTS_DIR / "dinov2_v4_reg_rot_pclip_22"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Heatmap kontrast stretch: tek outlier pikselin skalayi yutmasini engelle
HEATMAP_PCT_LO = 1.0
HEATMAP_PCT_HI = 99.0

# ---- Tunables ----
IMAGE_SIZE = 448
PATCH_SIZE = 14
GRID = IMAGE_SIZE // PATCH_SIZE

DINO_BLOCKS = [5, 8, 11]

NEIGHBORHOOD_KERNEL = 1
TOP_K = 1
STD_WEIGHT = 0.0

# D4 grup alt kumesi: identity + hflip + 3 rotasyon
TTA_VARIANTS = ["identity", "hflip", "rot90", "rot180", "rot270"]
TTA_REDUCE = "mean"
BLUR_KERNEL = 3

DEVICE = torch.device("cpu")

# ============================================================
class DinoV2PatchCoreV4:
    def __init__(self, model_name="dinov2_vitb14_reg"):
        self.model_name = model_name
        self.backbone = None
        self.memory_bank = None

        self.transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self):
        print(f"Model yükleniyor: {self.model_name}")
        self.backbone = torch.hub.load('facebookresearch/dinov2', self.model_name)
        self.backbone.to(DEVICE).eval()

    @torch.no_grad()
    def _forward_features(self, img_tensor):
        feats = self.backbone.get_intermediate_layers(
            img_tensor.to(DEVICE),
            n=DINO_BLOCKS,
            reshape=True,
            norm=True,
        )
        x = torch.cat(list(feats), dim=1)

        if NEIGHBORHOOD_KERNEL > 1:
            pad = NEIGHBORHOOD_KERNEL // 2
            x = F.avg_pool2d(x, kernel_size=NEIGHBORHOOD_KERNEL, stride=1, padding=pad)
        return x

    def extract_patches(self, img_tensor):
        x = self._forward_features(img_tensor)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1).reshape(-1, c)
        x = F.normalize(x, dim=1)
        return x.cpu().numpy()

    @staticmethod
    def _apply_tta(img_t, variant):
        if variant == "identity":
            return img_t
        if variant == "hflip":
            return torch.flip(img_t, dims=[-1])
        if variant == "rot90":
            return torch.rot90(img_t, k=1, dims=[-2, -1])
        if variant == "rot180":
            return torch.rot90(img_t, k=2, dims=[-2, -1])
        if variant == "rot270":
            return torch.rot90(img_t, k=3, dims=[-2, -1])
        return img_t

    @staticmethod
    def _invert_map(amap, variant):
        # amap: (GRID, GRID) numpy. Orijinale geri dondur.
        if variant == "identity":
            return amap
        if variant == "hflip":
            return amap[:, ::-1].copy()
        if variant == "rot90":
            return np.rot90(amap, k=-1).copy()
        if variant == "rot180":
            return np.rot90(amap, k=-2).copy()
        if variant == "rot270":
            return np.rot90(amap, k=-3).copy()
        return amap

    def fit(self, train_files):
        print("\nEğitim: DINOv2 memory bank (v4, reg+rot)...")
        self.load_model()

        existing = [f for f in train_files if os.path.exists(f)]
        if not existing:
            print("Eğitim dosyaları bulunamadı!")
            return

        all_features = []
        for p in tqdm(existing, desc="Features"):
            img = Image.open(p).convert("RGB")
            img_t = self.transform(img).unsqueeze(0)
            for v in TTA_VARIANTS:
                t = self._apply_tta(img_t, v)
                feats = self.extract_patches(t)
                all_features.append(feats)

        self.memory_bank = np.vstack(all_features).astype(np.float32)
        print(f"Memory Bank: {self.memory_bank.shape}")

    def _score_one(self, img_t):
        test_feats = self.extract_patches(img_t)
        qry = torch.from_numpy(test_feats)
        mem = torch.from_numpy(self.memory_bank)

        out = []
        chunk = 1024
        k = min(TOP_K, mem.shape[0])

        for i in range(0, qry.shape[0], chunk):
            sims = qry[i:i+chunk] @ mem.T
            topk_sims, _ = torch.topk(sims, k=k, dim=1)
            topk_dists = 1.0 - topk_sims
            mean_d = topk_dists.mean(dim=1)
            if k > 1 and STD_WEIGHT > 0.0:
                std_d = topk_dists.std(dim=1)
                score = mean_d + STD_WEIGHT * std_d
            else:
                score = mean_d
            out.append(score.cpu().numpy())

        d = np.concatenate(out)
        return d.reshape(GRID, GRID)

    def predict(self, test_files):
        print(f"\nTest (top-{TOP_K}, TTA {TTA_REDUCE} x{len(TTA_VARIANTS)}, neigh_k={NEIGHBORHOOD_KERNEL}, blur={BLUR_KERNEL})...")
        existing = [f for f in test_files if os.path.exists(f)]

        for p in tqdm(existing, desc="Testing"):
            img_orig = cv2.imread(p)
            img_pil = Image.fromarray(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
            img_t = self.transform(img_pil).unsqueeze(0)

            maps = []
            for v in TTA_VARIANTS:
                t = self._apply_tta(img_t, v)
                amap = self._score_one(t)
                amap = self._invert_map(amap, v)
                maps.append(amap)

            stack = np.stack(maps, axis=0)
            if TTA_REDUCE == "max":
                anomaly_map = stack.max(axis=0)
            else:
                anomaly_map = stack.mean(axis=0)

            if BLUR_KERNEL >= 3 and BLUR_KERNEL % 2 == 1:
                anomaly_map = cv2.GaussianBlur(anomaly_map, (BLUR_KERNEL, BLUR_KERNEL), 0)

            H, W = img_orig.shape[:2]
            anomaly_map_resized = cv2.resize(anomaly_map, (W, H))

            image_score = float(np.max(anomaly_map_resized))
            self.visualize(p, img_orig, anomaly_map_resized, image_score)

    def visualize(self, file_path, img_orig, anomaly_map, score):
        am_lo = np.percentile(anomaly_map, HEATMAP_PCT_LO)
        am_hi = np.percentile(anomaly_map, HEATMAP_PCT_HI)
        norm_map = np.clip((anomaly_map - am_lo) / (am_hi - am_lo + 1e-8), 0.0, 1.0)
        heatmap = (norm_map * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_orig, 0.5, heatmap_color, 0.5, 0)

        base_name = os.path.basename(file_path)
        cv2.putText(overlay, f"Score: {score:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        save_path = OUTPUT_DIR / f"DINOV2_V4_{base_name}"
        cv2.imwrite(str(save_path), overlay)


if __name__ == "__main__":
    print("=" * 60)
    print("DINOV2 + PATCHCORE v4  (vitb14_reg + 5-way TTA)")
    print("=" * 60)

    det = DinoV2PatchCoreV4(model_name="dinov2_vitb14_reg")
    det.fit(TRAIN_FILES)
    det.predict(TEST_FILES)

    print("\nTamamlandı.")
    print(f"Çıktılar: {OUTPUT_DIR.absolute()}")
