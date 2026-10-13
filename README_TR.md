# Metal Parçalarda Görsel Anomali Analizi

**Python · PyTorch · OpenCV · PatchCore · DINOv2**

Bir fabrikanın metal parça kontrolü ihtiyacı için geliştirdiğim bilgisayarlı görü prototipi. Hazır özellik çıkarıcıları ve referans özellik belleği kullanarak yeni görüntülerdeki farklı bölgeleri anomali skoru ve ısı haritalarıyla gösterir.

[English](README.md) · [Kurulum](docs/SETUP.md) · [Yöntem ve sonuçlar](docs/METHODS.md) · [Sunum PDF](docs/Fabrika_Portfoy_TR.pdf)

![Orijinal görüntü, Full Memory PatchCore ve DINOv2 v4 sonuçları](assets/ornek_test.jpg)

*Geçmişte kaydedilmiş görsel karşılaştırma; ölçülmüş doğruluk sonucu değildir.*

## Geliştirdiğim çalışma

- WideResNet/PatchCore ve DINOv2 tabanlı yaklaşımları sınırlı görüntü verisi üzerinde karşılaştırdım.
- Referans fotoğraflardan bölgesel özellik belleği oluşturdum.
- Test görüntülerini en yakın referans özellikleriyle karşılaştırıp anomali skorları ve ısı haritaları ürettim.
- Yüzey izleri, şekil farklılıkları ve kontrol işaretlerine verilen tepkileri inceledim.

Anomali uygulaması ve deneyler kişisel katkımdır. Ekip arkadaşımın ayrı delik çapı ölçümü çalışması bu depoda bulunmaz. Hazır araştırma modellerinin geliştirilmesi benim katkım olarak sunulmaz.

## Çalıştırma

[Kurulum yönergelerini](docs/SETUP.md) izleyip kendi sağlam referans fotoğraflarınızı `train_data`, incelenecek fotoğraflarınızı `test_data` klasörüne yerleştirin. Depo kökünden:

```bash
python run.py full-memory
python run.py dinov2
```

Başlatıcı eksik veya boş veri klasörlerini model yüklemeden kontrol eder. İki deney scripti de CPU üzerinde çalışır; ilk kullanımda hazır modeller indirilir. Referans özellik belleği yüksek RAM kullanabilir. Geçmiş bağımlılık sürümleri kaydedilmedi; bu yayımlama sırasında temiz ortamda tam çıkarım yeniden doğrulanmadı.

## Sonuç ve kapsam

Karşılaştırılan koşullarda **22 referans, 13 test görüntüsü ve iki yönteme ait 26 kayıtlı çıktı** bulunuyor. Bunlar projenin bütün veri setinin toplam sayısı değildir. Dört seçilmiş örnek ve yorumları [yöntem dokümanında](docs/METHODS.md) yer alır.

Çalışma demo aşamasında kaldı. Gerçek kontrol ortamından ek görüntüler istendi; veri akışı ilerlemedi. Otomatik kabul/ret eşiği, etiketli sayısal performans değerlendirmesi, hız ölçümü ve üretim kurulumu tamamlanmadı.

Skor kusur olasılığı veya başarı yüzdesi değildir. İki yöntemin skor ölçekleri farklıdır; renkler görüntü başına normalize edilir. Isı haritasında bir bölgenin vurgulanması, o bölgenin kesin kusur olduğunu göstermez.

## İçerik

- `experiments/`: iki özgün deney scripti.
- `assets/`: seçilmiş kayıtlı karşılaştırmalar.
- `docs/`: kurulum, yöntem açıklaması ve PDF sunum.
- `run.py`: veri kontrolü ve deney başlatma.
- `.github/workflows/`: kaynak sözdizimi ve belge bağlantısı kontrolü.

Tam fabrika veri seti, kontrol kitapçığı, model ağırlıkları ve üçüncü taraf Dinomaly deposu paylaşılmıyor. Orijinal yerel proje değiştirilmedi.

## Araştırma kaynakları

[DINOv2](https://github.com/facebookresearch/dinov2) · [TorchVision](https://github.com/pytorch/vision) · [PatchCore](https://github.com/amazon-science/patchcore-inspection)
