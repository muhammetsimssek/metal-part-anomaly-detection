import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import glob

# ============================================================
# CONFIG - MAXIMIZED FOR SMALL DATA
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
OUTPUT_DIR = ARTIFACTS_DIR / "full_memory_output_New"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Geliştirilmiş Ayarlar
IMAGE_SIZE = 320        # Daha yüksek çözünürlük
BATCH_SIZE = 32

# Sampling YOK! Tüm patchleri saklayacağız.
# CORESET_SIZE ayarı gerekmiyor çünkü hepsini tutuyoruz.

LAYERS_TO_EXTRACT = ['layer2', 'layer3']

DEVICE = torch.device("cpu")

# ============================================================
# HIGH DENSITY PATCHCORE CLASS
# ============================================================
class FullMemoryPatchCore:
    def __init__(self, backbone_name="wide_resnet50_2"): # Better backbone "Model"
        self.backbone_name = backbone_name
        self.backbone = None
        self.memory_bank = None
        self.outputs = {}
        
        # Transform işlemi
        self.transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def load_model(self):
        print(f"Model yükleniyor: {self.backbone_name}")
        if self.backbone_name == "wide_resnet50_2":
            self.backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
        self.backbone.to(DEVICE)
        self.backbone.eval()
        
        for name, module in self.backbone.named_children():
            if name in LAYERS_TO_EXTRACT:
                module.register_forward_hook(self._hook_fn(name))

    def extract_features(self, img_tensor):
        self.outputs = {}
        with torch.no_grad():
            _ = self.backbone(img_tensor.to(DEVICE))
            
        features = []
        for layer_name in LAYERS_TO_EXTRACT:
            feat_map = self.outputs[layer_name]
            
            # YOĞUNLUK ARTIRMA (Density Increase)
            # Daha önceki scriptte 32x32 idi.
            # Az veride detayı kaçırmamak için kritik.
            target_dim = 48 # 56 denedim sonra 64 e çektim 64 çok çok az da olsa daha iyi sonuç verdi...
            
            pooled = torch.nn.functional.adaptive_avg_pool2d(feat_map, (target_dim, target_dim))
            features.append(pooled)

        # Embeddingleri birleştir
        features = torch.cat(features, dim=1) # [B, C_total, H, W]
        
        # Patch vector haline getir
        b, c, h, w = features.shape
        features = features.view(b, c, -1).permute(0, 2, 1) # [B, N_patches, C]
        
        return features.reshape(-1, c).cpu().numpy()

    def fit(self, train_files):
        print("\n Eğitim: FULL MEMORY modu (Hiçbir patch silinmiyor)...")
        self.load_model()
        
        all_features = []
        
        existing_files = [f for f in train_files if os.path.exists(f)]
        if not existing_files:
            print("Eğitim dosyaları bulunamadı!")
            return

        for p in tqdm(existing_files, desc="Features"):
            img = Image.open(p).convert("RGB")
            img_t = self.transform(img).unsqueeze(0)
            feats = self.extract_features(img_t)
            all_features.append(feats)
            
        # Sampling yapmıyoruz! Hepsini tutuyoruz.
        self.memory_bank = np.vstack(all_features)
        
        print(f"Toplam Özellik Sayısı: {self.memory_bank.shape[0]}")
        print(f"Memory Bank Hazır (Boyut: {self.memory_bank.shape})")

    def predict(self, test_files):
        print("\nTest Aşaması (Full Precision)...")
        existing_files = [f for f in test_files if os.path.exists(f)]
        
        mem_bank_tensor = torch.from_numpy(self.memory_bank).to(DEVICE)
        
        for p in tqdm(existing_files, desc="Testing"):
            img_orig = cv2.imread(p)
            img_pil = Image.fromarray(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
            img_t = self.transform(img_pil).unsqueeze(0)
            
            test_feats = self.extract_features(img_t)
            test_feats_tensor = torch.from_numpy(test_feats).to(DEVICE)
            
            dists = []
            
            # Bellek yönetimi için chunk size
            # Memory bank artık daha büyük olabilir, chunk'ı ufak tutalım
            chunk_size = 512 
            
            for i in range(0, test_feats_tensor.shape[0], chunk_size):
                test_chunk = test_feats_tensor[i:i+chunk_size]
                
                # Pairwise distance
                # Full memory ile karşılaştırma
                # Eğer memory çok büyükse (örn 100k patch) burası yavaşlayabilir
                # Ama 3-4 resim için (yaklaşık 12k patch) sorun olmaz.
                dist_matrix = torch.cdist(test_chunk, mem_bank_tensor, p=2)
                
                min_val, _ = torch.min(dist_matrix, dim=1)
                dists.extend(min_val.cpu().numpy())
            
            dists = np.array(dists)
            
            # Reshape to 56x56
            grid_size = int(np.sqrt(len(dists))) # Should be 56
            anomaly_map = dists.reshape(grid_size, grid_size)
            
            # Smoothing - Grid büyüdüğü için kernel'i de hafif büyütebiliriz
            anomaly_map = cv2.GaussianBlur(anomaly_map, (11, 11), 0)
            
            # Upsample
            anomaly_map_resized = cv2.resize(anomaly_map, (img_orig.shape[1], img_orig.shape[0]))
            
            image_score = np.max(anomaly_map_resized)
            self.visualize(p, img_orig, anomaly_map_resized, image_score)

    def visualize(self, file_path, img_orig, anomaly_map, score):
        am_min = anomaly_map.min()
        am_max = anomaly_map.max()
        norm_map = (anomaly_map - am_min) / (am_max - am_min + 1e-8)
        heatmap = (norm_map * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(img_orig, 0.5, heatmap_color, 0.5, 0)
        
        base_name = os.path.basename(file_path)
        cv2.putText(overlay, f"Score: {score:.3f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        save_path = OUTPUT_DIR / f"FULL_MEM_{base_name}"
        cv2.imwrite(str(save_path), overlay)

if __name__ == "__main__":
    print("="*60)
    print("PATCHCORE 'FULL MEMORY' (Small Data Optimized)")
    print("="*60)
    
    detector = FullMemoryPatchCore(backbone_name="wide_resnet50_2")
    detector.fit(TRAIN_FILES)
    detector.predict(TEST_FILES)
    
    print("\nTamamlandı chf.")
    print(f"Çıktılar: {OUTPUT_DIR.absolute()}")
