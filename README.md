# Türk Plaka Tespiti — YOLOv8 (Aşama 1)

Türk araç plakalarını okuyan iki aşamalı bir sistemin **1. aşaması**: görüntüdeki
plakanın **yerini** tespit eden bir YOLOv8n modeli. Bu aşama yalnızca tespit
(lokalizasyon) yapar; tespit edilen plaka bölgesi kırpılıp kaydedilir ve
2. aşamadaki karakter okuma (OCR) modülünün girdisi olur.

> **Aşama 2 — OCR entegrasyonu planlanıyor.** Bu repo şimdilik yalnızca tespit
> modelini içerir.

## Repo yapısı

```
├── download_data.py   # Roboflow'dan veri indirme (API anahtarı env'den)
├── train.py           # Yerel eğitim + değerlendirme
├── train.ipynb        # Google Colab notebook — önerilen eğitim yöntemi
├── predict.py         # Çıkarım: kutu çizme + plaka kırpma
└── requirements.txt
```

## Kurulum

Python 3.10+ gerekir.

```bash
git clone <bu-repo>
cd turkish-plate-reader
pip install -r requirements.txt
```

## Veri indirme

Veri seti Roboflow Universe'te barındırılıyor ve **repoya dahil değildir**
(bkz. `.gitignore`). Kendiniz indirmeniz gerekir:

1. Ücretsiz bir Roboflow hesabı açın ve API anahtarınızı alın:
   https://app.roboflow.com/settings/api
2. Anahtarı **ortam değişkeni** olarak tanımlayın (koda/dosyaya gömmeyin):

   ```bash
   # Linux / macOS
   export ROBOFLOW_API_KEY="anahtariniz"

   # Windows PowerShell
   $env:ROBOFLOW_API_KEY="anahtariniz"
   ```

   Alternatif olarak proje kökünde bir `.env` dosyasına
   `ROBOFLOW_API_KEY=...` yazabilirsiniz (`.env` gitignore'dadır).

3. İndirin:

   ```bash
   python download_data.py
   ```

Veri `datasets/turkish-plates/` altına YOLOv8 formatında iner
(`data.yaml` dahil). Roboflow'un verdiği train/val bölünmesi olduğu gibi
kullanılır; YOLOv8'in eğitim-anı augmentation'ı ayrıca varsayılan ayarlarla
açıktır.

**Veri seti:** [License Plates of Vehicles in Turkey](https://universe.roboflow.com/kemalkilicaslan-gzpvq/license-plates-of-vehicles-in-turkey-s3tbj)
— 3.501 görüntü, tek sınıf (`license plate`), lisans: **CC BY 4.0** (atıf için
aşağıya bakın).

## Eğitim

### Yöntem 1 — Google Colab (önerilen)

Eğitim GPU gerektirir; en kolay yol ücretsiz Colab GPU'su kullanmaktır.

1. `train.ipynb` dosyasını [Google Colab](https://colab.research.google.com)'da açın.
2. **Runtime → Change runtime type → T4 GPU** seçin.
3. Roboflow API anahtarınızı Colab **Secrets** paneline `ROBOFLOW_API_KEY`
   adıyla ekleyin (ya da notebook sorduğunda girin).
4. Hücreleri sırayla çalıştırın. Notebook veriyi indirir, modeli eğitir,
   metrikleri raporlar ve `best.pt`'yi bilgisayarınıza indirir.
5. İnen `best.pt`'yi yerel repoda `weights/` klasörüne koyun.

### Yöntem 2 — Yerel (GPU'lu makine)

```bash
python download_data.py
python train.py                      # varsayılan: epochs=50 imgsz=640 batch=16
python train.py --batch 8            # GPU belleği yetmezse batch'i düşürün
```

Eğitim sonunda en iyi ağırlık `weights/best.pt` olarak kopyalanır,
val metrikleri `results/metrics.md`'ye, örnek tahmin görüntüleri
`results/val_predictions/` altına yazılır.

> **Not:** Colab'ın ücretsiz T4 GPU'sunda da bellek yetmezliği (CUDA out of
> memory) görürseniz `batch=8` deneyin.

## Değerlendirme sonuçları

Aşağıdaki tablo, eğitim tamamlandıktan sonra `results/metrics.md`'den (veya
Colab notebook'unun 6. hücresinin çıktısından) doldurulacaktır:

| Metrik | Değer |
|---|---|
| mAP@0.5 | _eğitim sonrası doldurulacak_ |
| mAP@0.5:0.95 | _eğitim sonrası doldurulacak_ |
| Precision | _eğitim sonrası doldurulacak_ |
| Recall | _eğitim sonrası doldurulacak_ |

Örnek val tahminleri (kutular çizilmiş) `results/val_predictions/` klasöründe.

## Çıkarım (predict.py)

```bash
# Tek görüntü
python predict.py --source ornek.jpg

# Klasör + özel güven eşiği ve çıktı klasörü
python predict.py --source foto_klasoru/ --conf 0.4 --out predictions
```

Çıktılar:

- `predictions/annotated/` — tespit kutuları çizilmiş görüntüler
- `predictions/crops/` — **kırpılmış plaka bölgeleri** (her tespit ayrı dosya;
  bunlar Aşama 2'deki OCR'ın girdisi olacak)

Varsayılan güven eşiği `--conf 0.25`'tir.

## Lisans ve atıf

- Kod: [MIT](LICENSE)
- Veri seti: **License Plates of Vehicles in Turkey**, Kemal Kılıçaslan,
  Roboflow Universe — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

  > Kılıçaslan, K. *License Plates of Vehicles in Turkey* [Veri seti].
  > Roboflow Universe.
  > https://universe.roboflow.com/kemalkilicaslan-gzpvq/license-plates-of-vehicles-in-turkey-s3tbj

Veri seti ve eğitilmiş ağırlıklar bu repoda **dağıtılmaz**; veriyi yukarıdaki
adımlarla kendiniz indirin.
