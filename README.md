# MRI Görüntü Analizi ve Qwen Asistanı

Bu ödev projesi üç parçayı bir araya getirir:

1. Hazır ağırlıklı U-Net ile FLAIR anormallik segmentasyonu.
2. **Fine-tune edilmeden**, ImageNet-1K pretrained EfficientNet-B0 ile 1280 boyutlu genel görüntü özelliği çıkarımı.
3. Ollama üzerinde yerel `qwen2.5vl:3b` ile görüntü hakkında İngilizce, eğitim amaçlı açıklama.

EfficientNet ImageNet üzerinde eğitildiği için ürettiği özellikler tümör tanısı veya hastalık olasılığı değildir. Qwen katmanı kesin tanı, reçete, ilaç/doz başlatma-bırakma talimatı vermez; görüntüde betimlenebilenleri, belirsizlikleri, doktorla görüşülebilecek genel yaklaşımı ve acil uyarı işaretlerini açıklar.

## Veri Kümesi

Bu proje, Kaggle üzerindeki [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) veri kümesini kullanır. Veri kümesi yaklaşık 1 GB ve 7.860 dosyadan oluştuğu için GitHub deposuna dahil edilmemiştir. Ayrıca veri kümesi **CC BY-NC-SA 4.0** lisansı altında ayrı olarak dağıtılmaktadır.

Projeyi çalıştırmadan önce veri kümesini Kaggle'dan indirin. İndirdiğiniz arşivdeki `kaggle_3m` klasörünü proje kök dizinine yerleştirin:

```text
medical_assistant/
├── app.py
├── requirements.txt
├── kaggle_3m/
│   ├── TCGA_CS_4941_19960909/
│   ├── TCGA_CS_4942_19970222/
│   └── ...
└── ...
```

`kaggle_3m/` klasörü boyut ve lisans nedenleriyle `.gitignore` içinde tutulur. Model ağırlıkları da depoya eklenmez; uygulama ilk çalıştırmada gerekli U-Net ağırlığını otomatik indirir.

## Kurulum

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama list
.\.venv\Scripts\python.exe -m streamlit run app.py
```

`ollama list` çıktısında `qwen2.5vl:3b` bulunmalıdır. Ollama arka planda çalışmıyorsa `ollama serve` komutunu çalıştırın. İlk EfficientNet kullanımında torchvision'ın resmi ImageNet ağırlıkları indirilebilir.

## Akış

```text
MRI TIFF ──┬──> U-Net ──> segmentasyon maskesi ve işaretli alan oranı
           ├──> EfficientNet-B0 (frozen) ──> 1280 boyutlu genel özellik
           └──> Qwen2.5-VL + kullanıcı sorusu ──> güvenli eğitim açıklaması
```

Qwen görüntünün kendisini görür. U-Net'in işaretlediği oran ek bağlam olarak gönderilir; EfficientNet embedding'i dil modeline aktarılmaz çünkü bunun tıbbi anlamı fine-tune/eğitim olmadan tanımlı değildir.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

> Eğitim prototipidir; klinik kullanım için doğrulanmamıştır. Sonuçlar uzman değerlendirmesi gerektirir.
