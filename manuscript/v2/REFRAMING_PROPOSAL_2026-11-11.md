# Yeniden çerçeveleme önerisi (taslak, 2026-11-11)

Durum: **uygulandı (2026-11-11, yazar onayıyla)**; ayrıntı REVISION_PLAN kaydında. Aşağıdaki metin öneri hâliyle korunmuştur; bölüm 5'teki hedef uzunluk (7.000–8.000 kelime) yazar kararıyla geri çekildi; kısaltma dergi seçildiğinde yapılacak. Yeni deney, yeniden
eğitim veya yeni veri gerektirmez; bütün sayılar `results/v2_2026-11` altındaki kayıtlı çıktılardan gelir. Araştırma sorusunun
vurgusu değiştiği için uygulanması yazar onayına bağlıdır (CLAUDE.md §3).

## 1. Neden

Mevcut metin "donmuş temel-model temsilleri göreve özgü sınıflandırıcılardan daha iyi aktarılıyor mu?" sorusunu soruyor ve cevabı
d1-3B probuna bağlıyor. Denetimden sonra bu cevabın dayanakları zayıf:

- Probun CinC 2011'deki anlamlı üstünlüğü 73 düz derivasyonlu kayda bağlı (onlarsız p = 0,069; Tablo S40).
- PTB-XL'de prob zayıf (0,579–0,592) ve HuBERT-ECG ile LightGBM'in anlamlı olarak altında.
- "Hiçbir zaman en iyinin anlamlı olarak gerisinde değil" ifadesi dört kümenin üçünde fark bulunamamasına dayanıyor.

Aynı veriler başka bir iddiayı daha sağlam destekliyor: **alan-içi karşılaştırma, aktarılan modeli güvenilir biçimde göstermiyor ve
harici sıralamalar veri seti artefaktlarına duyarlı.** Bu iddia tek bir modele değil 11 modelin tamamına dayanıyor.

## 2. Önerilen ana mesaj (tek cümle)

> In-domain benchmarks of ECG usability models did not reliably identify the models that transfer, and external rankings and
> operating points depended on flat leads, recording length, aggregation and local prevalence.

Temel-model sorusu kaybolmuyor; ikinci amaç olarak kalıyor ("no approach, foundation model or not, was best throughout").

## 3. Başlık seçenekleri

1. **In-domain performance does not identify ECG usability models that transfer: a patient-independent benchmark of eleven
   approaches across four datasets** (önerilen)
2. External validity of ECG signal-quality models: a multi-dataset benchmark of task-specific classifiers and foundation-model
   representations
3. What changes the ranking of ECG usability models under dataset shift? A patient-independent, multi-dataset benchmark

## 4. Özet taslağı (İngilizce)

**Objective.** To test whether in-domain performance of electrocardiogram (ECG) usability models predicts performance on other
datasets, and which evaluation choices change the conclusions.

**Methods.** Eleven approaches (signal-quality-index models; raw-signal, image and multimodal networks; a rule heuristic;
zero-shot LiquidAI d1-3B; probes on frozen d1-3B and HuBERT-ECG) were developed on BUT QDB (988 10-s windows, 15 subjects;
subject-grouped cross-validation) and applied unchanged to CinC 2011 (998 records), CinC 2017 (8,512 recordings, 3.3% noisy)
and, post hoc, PTB-XL.

**Results.** In-domain, eight approaches reached AUROC 0.938–0.985, none significantly below the best. In-domain AUROC was
moderately related to AUROC on CinC 2011 (Spearman ρ = 0.61) and unrelated on CinC 2017 (ρ = −0.02): the raw-signal network fell
from second (0.983) to tenth (0.768), and the HuBERT-ECG probe rose from eighth to first (0.906) while near chance on CinC 2011
(0.562). No approach was best throughout; zero-shot d1-3B was weak (0.641–0.828). Artefacts mattered: 73 flat-lead CinC 2011
records decided whether the two leading models differed (Holm-adjusted p = 0.003 with them; p = 0.069 without), and on CinC
2017 the number of windows alone gave AUROC 0.698. On PTB-XL no model exceeded 0.736. Thresholds set for 5% false rejection on
BUT QDB gave 3–58% on CinC 2017; re-set on 100 local recordings, median sensitivity stayed below 30%.

**Conclusions.** In-domain benchmarks did not reliably identify the models that transferred, and external rankings were
sensitive to flat leads, recording length and aggregation. ECG quality models should be compared externally, with dataset
artefacts checked and thresholds set locally.

Not: 250 kelime sınırı için d1-3B probunun "hiçbir kümede en iyinin anlamlı olarak gerisinde değil" bulgusu özetten çıkarıldı;
ana metinde (3.5) kalır. PTB-XL'in "sonradan eklenen, gürültü vekili" niteliği Yöntemler'de açıklanır.

Sayıların kaynağı:

| İfade | Kaynak | Durum |
|---|---|---|
| 0,938–0,985; 0,983; 0,768; 0,906; 0,562; 0,641–0,828 | `model_cis.csv` | Makalede var |
| Sıralar (2. → 10.; 8. → 1.) | `model_cis.csv` AUROC sıralaması | **Yeni; tabloya girmeli** |
| Spearman ρ = 0,61 (CinC 2011), −0,02 (CinC 2017); n = 11 model | `model_cis.csv` | **Yeni; betikle üretilmeli** |
| p = 0,003 ve 0,069 | `paired_comparisons.csv`, `cinc2011_nonconstant_paired.csv` | Makalede var (Tablo S40) |
| 0,698 | `cinc2017_length_stratified.csv` | Makalede var (Tablo S42) |
| PTB-XL ≤ 0,736 (maksimum kuralı; ortalama kuralında en yüksek 0,694) | `ptbxl_frozen_summary.csv` | Makalede var (Tablo S38) |
| %3–58 (BUT QDB'de YPO ≤ %5 eşiği, CinC 2017'de gerçekleşen: %3,1–%58,1) | `threshold_transfer_summary.csv` | Tablo S4'te var; metne aralık olarak girmeli |
| Medyan duyarlılık < %30 (n = 100, CinC 2017) | `local_recalibration_summary.csv` | Makalede var |

Dikkat edilecekler:
- ρ yalnızca 11 modele dayanıyor; güven aralığı geniştir. CinC 2011 için ρ = 0,61 (p = 0,047) "ilişkisiz" değildir. Bu yüzden
  iddia "öngörmüyor" değil, "güvenilir biçimde belirlemiyor" olarak yazıldı. Betimsel ve post hoc olarak etiketlenmeli.
- "Sekiz yaklaşımdan biri elle ayarlı" notu özetten çıkarılırsa Bölüm 3.1'de kalmalı.

## 5. Bölüm yapısı

Giriş, son paragraf — şu anki soru: "do frozen foundation-model representations hold up better than classifiers built for the
task?" Önerilen:

> This study asks two questions. First, does in-domain performance under a patient-independent protocol identify the ECG
> usability models that perform best on other datasets? Second, do frozen foundation-model representations hold up under
> dataset shift better than classifiers built for the task?

Sonuçlar:

| Yeni bölüm | İçerik | Kaynak (şu anki yer) |
|---|---|---|
| 3.1 In-domain performance does not separate the models | 11 model, güç sınırı, denek bazında AUROC | 3.1, Tablo S41 |
| 3.2 In-domain rank does not identify external rank | Üç kümede AUROC ve sıra tablosu, ρ; 1D CNN ve HuBERT örnekleri | 3.2 (yeni tablo) |
| 3.3 What moves external rankings | Düz derivasyonlar (S26, S40), tek derivasyon / 12 derivasyon etiketi (S5, S35), kayıt uzunluğu ve toplama kuralı (S42), SQI ölçek artefaktı (S36) | 3.2 |
| 3.4 Operating points and calibration do not transfer | Aktarılan eşikler, kalibrasyon, yerel yeniden ayarlama | 3.4 |
| 3.5 Foundation-model representations | Sıfır-örnekli d1-3B, iki prob, HuBERT-ECG, PTB-XL (kısaltılmış) | 3.1, 3.2, 3.6 |
| 3.6 Multi-source training | Bir paragraf; Tablo 2 kalır | 3.3 |

Eke taşınacaklar (metinde birer cümle kalır): NSTDB gürültü testi, gecikme ölçümleri (3.5), kalite eleme (3.7), istem
varyantları, raster çizici, keşifsel flatline, resmi bölünme, Sınıf 2 duyarlılığı, HuBERT ince ayarı.

Tartışma: (1) alan-içi karşılaştırmanın sınırı; (2) sıralamayı değiştiren üç artefakt ve bunlar için önerilen kontroller (sabit
sinyal kontrolü, pencere sayısına göre tabakalama, etiket–girdi uyumu); (3) temel modeller: tek tip sonuç yok; (4) eşikler.

Hedef uzunluk: ana metin şu an tablolarla birlikte yaklaşık 14.000 kelime; taşımalarla yaklaşık 7.000–8.000.

## 6. Yapılması gerekenler (onay sonrası)

1. `evaluation/audit_sensitivity.py`'ye sıra/ρ çıktısı ve tablo üreticiye bir ana metin tablosu eklemek (kayıtlı AUROC'lardan).
2. Şablonda başlık, özet, giriş son paragrafı, sonuç bölümlerinin sırası ve tartışmayı düzenlemek; taşınan bölümleri eke almak.
3. Bölüm 2.6'ya bir madde: çerçeve ve birincil soru, tüm sonuçlar görüldükten sonra değiştirildi.
4. `verify_v2_prose.py` kontrollerini yeni cümlelere göre güncellemek; TRIPOD madde 1–2.

## 7. Riskler

- Çerçeve sonuçlardan sonra değişiyor; bu açıkça beyan edilmezse seçici raporlama izlenimi doğar. Madde 6.3 bunun için.
- Ana iddia artık "olumsuz" bir bulgu. Karşılaştırma çalışmalarına açık dergiler için uygun, "yeni yöntem" bekleyenler için değil.
- Geliştirme kohortunun küçüklüğü (15 denek, 7'sinde pozitif) bu çerçevede de temel sınırlılık olarak kalır.
