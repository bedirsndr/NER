# Türkçe Varlık İsmi Tanıma (NER): Sıfırdan Transformer vs. BERTurk

Türkçe cümlelerde **PERSON**, **LOCATION** ve **ORGANIZATION** varlıklarını token seviyesinde (BIO etiketleme) tanıyan iki yaklaşımın karşılaştırması:

1. **Baseline:** Sıfırdan eğitilen, kelime seviyesinde sözlüğe sahip bir `nn.Transformer` modeli
2. **BERTurk:** `dbmdz/bert-base-turkish-cased` modelinin token sınıflandırma için ince ayarı (fine-tuning)

Tüm kod tek bir notebook içindedir: [`NER.ipynb`](NER.ipynb)

---

## Sonuçlar (test seti, %20)

| Model | Test doğruluğu (token) | Macro Precision | Macro Recall | Macro F1 |
|---|:---:|:---:|:---:|:---:|
| Transformer (sıfırdan, 10 epoch) | 0.958 | 0.77 | 0.69 | 0.73 |
| **BERTurk (fine-tune, 5 epoch)** | **0.994** | **0.96** | **0.96** | **0.96** |

Sınıf bazında F1 skorları:

| Etiket | Transformer | BERTurk |
|---|:---:|:---:|
| O | 0.99 | 1.00 |
| B-PERSON | 0.77 | 0.97 |
| I-PERSON | 0.51 | 0.97 |
| B-ORGANIZATION | 0.76 | 0.96 |
| I-ORGANIZATION | 0.73 | 0.95 |
| B-LOCATION | 0.81 | 0.97 |
| I-LOCATION | 0.53 | 0.89 |

Baseline model özellikle çok kelimeli varlıkların devam token'larında (`I-PERSON`, `I-LOCATION`) zayıf kalırken, BERTurk tüm sınıflarda dengeli bir performans gösteriyor.

---

## Veri Seti

`nerdata.txt` dosyası, varlıkların `<b_enamex TYPE="...">...<e_enamex>` etiketleriyle işaretlendiği bir Türkçe NER veri setidir. Notebook bu formatı regex ile ayrıştırıp kelime + BIO etiketi çiftlerine dönüştürür.

Örnek biçim (açıklayıcı):

```
<b_enamex TYPE="PERSON">Ali Yılmaz<e_enamex> dün <b_enamex TYPE="LOCATION">İzmir<e_enamex>'e gitti .
```

Token bazında etiket dağılımı (toplam 492.233 token):

| Etiket | Adet | Oran |
|---|---:|---:|
| O | 438.976 | %89,18 |
| B-PERSON | 16.293 | %3,31 |
| I-PERSON | 7.585 | %1,54 |
| B-LOCATION | 10.889 | %2,21 |
| I-LOCATION | 1.559 | %0,32 |
| B-ORGANIZATION | 10.031 | %2,04 |
| I-ORGANIZATION | 6.900 | %1,40 |

Veri, `train_test_split(test_size=0.2, random_state=42)` ile %80 eğitim / %20 test olarak bölünmüştür.

> `nerdata.txt` dosyası notebook ile aynı dizinde bulunmalıdır.

---

## Yaklaşım

```
nerdata.txt
    │  regex ile ayrıştırma (<b_enamex ...>)
    ▼
kelimeler + BIO etiketleri ──► %80 eğitim / %20 test
    │
    ├──────────────────────────────┐
    ▼                              ▼
[A] Baseline                   [B] BERTurk
 kelime → id (<PAD>,<UNK>)      WordPiece tokenizer
 Embedding (256)                is_split_into_words=True
 nn.Transformer (4+4 katman)    ilk alt-kelime → etiket, diğerleri -100
 Linear → 8 sınıf               BertForTokenClassification
    │                              │
    └──────────────┬───────────────┘
                   ▼
   Loss/Accuracy eğrileri + classification_report
```

### Model A: Transformer (sıfırdan)

| Parametre | Değer |
|---|---|
| Sözlük | Kelime seviyesi (`<PAD>`, `<UNK>` + veri setindeki kelimeler) |
| d_model / nhead | 256 / 8 |
| Encoder / Decoder katmanı | 4 / 4 |
| Dropout | 0.3 |
| Optimizer / LR | Adam / 3e-4 |
| Batch size / Epoch | 16 / 10 |
| Maksimum uzunluk | 128 |
| Loss | CrossEntropy (`ignore_index=0`) |

### Model B: BERTurk

| Parametre | Değer |
|---|---|
| Model | `dbmdz/bert-base-turkish-cased` |
| Sınıf sayısı | 8 (7 etiket + PAD) |
| Learning rate | 2e-5 |
| Batch size / Epoch | 16 / 5 |
| Weight decay | 0.01 |
| Maksimum uzunluk | 128 (alt-kelime) |
| Eğitim süresi | ~15,3 dk (6.890 adım) |

Alt-kelimelere bölünen kelimelerde yalnızca ilk alt-kelime etiketlenir, diğerleri `-100` ile loss dışında bırakılır.

---

## Kurulum ve Çalıştırma

Notebook Google Colab (GPU) üzerinde çalıştırılmıştır.

```bash
pip install torch "transformers>=4.41" accelerate scikit-learn matplotlib tqdm numpy
```

1. `nerdata.txt` dosyasını notebook ile aynı dizine koyun.
2. `NER.ipynb` dosyasını açın ve hücreleri sırayla çalıştırın.
3. İlk bölüm baseline Transformer'ı, ikinci bölüm BERTurk'ü eğitir.

> `TrainingArguments(eval_strategy=...)` parametresi nedeniyle `transformers` sürümü 4.41 veya üzeri olmalıdır.

---

## Notlar ve Sınırlılıklar

- **Metrikler token seviyesindedir.** `O` sınıfı verinin ~%89'unu oluşturduğu için genel doğruluk (accuracy) şişkindir. Model kalitesini asıl yansıtan değerler sınıf bazlı precision/recall/F1'dir. Varlık seviyesinde F1 (ör. `seqeval`) hesaplanmamıştır.
- **Ayrı bir doğrulama seti yoktur.** %20'lik test bölümü eğitim sırasında izleme için de kullanılmıştır. Model seçimi veya erken durdurma yapılmadığı (`save_strategy="no"`) ve hiperparametre araması yapılmadığı için bu bir seçim sızıntısı yaratmaz; yine de temiz bir doğrulama/test ayrımı daha güvenilir olurdu.
- **Baseline sadeleştirilmiş bir referanstır.** Encoder ve decoder'a aynı girdi verilmiş, padding/causal maske kullanılmamış ve konumsal kodlama (positional encoding) eklenmemiştir. Bu tercihler baseline skorunu düşürüyor olabilir.
- Baseline'ın kelime sözlüğü tüm cümlelerden (test dahil) oluşturulmuştur.
- İki modelin `support` değerleri küçük farklarla ayrışır: BERTurk'te 128 alt-kelimelik kırpma uygulanır ve yalnızca ilk alt-kelimeler değerlendirilir.
- `compute_metrics` fonksiyonu tanımlıdır ancak `Trainer`'a verilmemiştir; doğruluk değerleri `AccuracyCallback` ve eğitim sonrası `classification_report` ile hesaplanır.

## Olası Geliştirmeler

- `seqeval` ile varlık seviyesinde precision/recall/F1
- Transformer baseline'a positional encoding ve maskeler eklemek
- Ayrı doğrulama seti ve birden fazla tohum (seed) ile tekrar
- CRF katmanı veya farklı Türkçe BERT varyantları ile karşılaştırma

## Repo Yapısı

```
ner-turkish-bert/
├── NER.ipynb
├── nerdata.txt        # (veri seti; notebook ile aynı dizinde olmalı)
└── README.md
```
