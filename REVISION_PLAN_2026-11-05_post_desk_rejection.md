# Revizyon Planı — PMEA masa başı reddi sonrası (2026-11-05)

Kaynak: dış ajanın hazırladığı "EKG Makalesi Geliştirme Projesi" rehberi (16 sayfa, 5 iş paketi).
Bu belge rehberin her maddesini depodaki kod ve sonuçlara karşı kontrol eder ve
**kabul / değiştirilerek kabul / reddedildi** kararını gerekçesiyle kaydeder.
Rehber bir LLM çıktısıdır (metinde `[span_5](start_span)` gibi kalıntılar var); iddiaları doğrulanmadan uygulanmaz.

## 0. Rehber iddialarının doğrulanması (2026-11-05)

| İddia | Durum | Kanıt |
|---|---|---|
| Tablo S30'da sütun kayması var (LightGBM satırında BUT_QDB=0,880 vb.) | **YANLIŞ** | `results/model_cis.csv`, `manuscript/supplement_q1.md` satır 535–541 ve gönderilen PDF (sayfa 22): LightGBM BUT 0,964 / CinC2011 0,754 / CinC2017 0,880 / dengeli 0,880 — doğru sütunlarda. `table_1g()` sütunları veri seti adına göre eşliyor. Düzeltilecek bir hata yok. |
| d1-3B, LFM2.5-VL-3B tabanlı; görsel kodlayıcı SigLIP2 NaFlex shape-optimized 400M | Doğru | HF model kartı `LiquidAI/d1-3B` |
| d1-2B, LFM3-VL-2B tabanlı ve d1-3B'den daha iyi | Doğru (üretici beyanı) | HF model kartı `LiquidAI/d1-2B`. "3 kat daha az hesaplama" iddiası doğrulanmadı → kullanılmayacak. |
| CinC 2011 Set A'da 998 etiketli kayıttan yalnızca 565'i skorlandı | Doğru | `results/model_cis.csv` (565 kayıt; ilk-75 dengeli küme + 5 rastgele çekiliş birleşimi) |
| 105001'in dışlanan bölgesindeki 59 pencere ana analizde | Doğru | F1a yalnızca değerlendirmeden çıkarıyor; modeller 1.047 pencereyle eğitildi (`results/butqdb_official_split_summary.csv`) |
| window_duration ve sampling_rate sabit girdiler | Doğrulanacak | `features/sqi_extractor.py` satır 27, 149 |
| "Van Calster 2019: güvenilir kalibrasyon eğrisi için ≥200 olay ve ≥200 olay-dışı" | Doğrulanmadı | Bu öneri büyük olasılıkla Van Calster ve ark. 2016 (J Clin Epidemiol, kalibrasyon hiyerarşisi) kaynağından; kaynak doğrulanmadan metne girmeyecek. |
| İP-2 tablosundaki ~35–40 kaynak | **Her biri ayrı doğrulanacak** | Bazıları şüpheli (ör. "Zhu ve ark. 2020 Physiol Meas 41 105001" — 105001 aynı zamanda BUT QDB kayıt numarası; "Mondal ve ark. 2024 IEEE Sens Lett 8 1–4" eksik künye). Doğrulanamayan kaynak kaynakçaya girmez. |

## 1. İş paketleri — karar ve uygulama

### İP-3 Deneysel temizlik (önce yapılır; metin bunlara bağlı)

| # | Rehber önerisi | Karar | Gerekçe / bilimsel not |
|---|---|---|---|
| 3.1 | Birincil kohortu resmi protokole uydur: 59 pencereyi çıkar, tüm modelleri 988 pencerede **yeniden eğit**; F1b'yi (TabPFN dahil) ikincil analiz yap | **Kabul** | Gerekçe veri setinin yayımlanmış protokolüdür, sonuçlara bakarak seçilmemiştir. Ancak birincil kohortun sonuçlar görüldükten sonra değiştiği makalede bir cümleyle beyan edilecek (CLAUDE.md §4). Eski 1.047 pencere sonuçları silinmez; ek materyalde duyarlılık analizi olarak kalır. Kat (fold) ataması değişmeden korunacak. |
| 3.2 | CinC 2011 Set A'nın tamamını (998 kayıt) tüm modellerle skorla | **Kabul** | Saf harici değerlendirme; ayarlama yok. İlk-75 / 565 alt küme tabloları kaldırılır. Modeller yalnızca BUT QDB ile eğitildiği için sızıntı riski yok. LODO analizinde CinC 2011 eğitim kaynağı olduğundan orada da 998 kayıt kullanılacak. |
| 3.3a | Sabit iki girdiyi (window_duration, sampling_rate) tablosal modellerden çıkar | **Kabul** | Sayısal etkisi ~0 olmalı (standartlaştırmada sabit sütun sıfırlanır); yeniden çalıştırmada doğrulanacak. d1-3B sıfır-örnekli istemi **değiştirilmeyecek** (istem değişikliği yeni bir model demektir ve ~10 saatlik yeniden skorlama gerektirir); istem metni makalede olduğu gibi raporlanacak. |
| 3.3b | Kuantizasyon-duyarlı flatline tanımını kodla ve CinC 2011 transferini iyileştirdiğini göster | **Değiştirilerek kabul** | Bu düzeltme CinC 2011 sonuçları görüldükten sonra tasarlanıyor → **keşifsel (post-hoc)** olarak etiketlenecek, birincil sonuçların yerine geçmeyecek. Tolerans ve minimum süre CinC verisine bakılarak ayarlanmayacak; kayıt başlığındaki ADC çözünürlüğünden ve önceden yazılmış sabit bir süreden (ör. ≥ 200 ms) türetilecek. "Transfer çöküşünü çözer" iddiası ancak sonuç bunu gösterirse yazılır. |
| 3.4 | Kural tabanlı sezgiseldeki sızıntı şerhini kaldır (katsayıları eğitim katlarında öğren) | **Kabul — mevcut analiz öne alınacak** | `experiments/heuristic_refit.py` bunu zaten yapıyor (Tablo S22). Ana tabloda kat-içi yeniden uyarlanmış sürüm birincil, el ayarlı sürüm ikincil olacak. Rehberin "iç içe CV ile 7 katsayıyı optimize et" önerisi gereksiz: sabit C=1 ile ayar yapılmadığından iç içe CV'ye gerek yok. |
| 3.5 | Gecikme ölçümlerini tek betikte birleştir; PIL raster çizicisiyle 2D CNN ve d1 probe AUROC'unun korunduğunu göster | **Kabul** | Raster çizici farklı bir girdi dağılımı üretir; 2D CNN'in PIL görüntüleriyle **yeniden eğitilmesi** gerekir (aksi hâlde alan kayması ölçülür). Bu ikincil analizdir. Ölçümler tek oturumda, ısınma çağrılarından sonra, sabit sırayla yapılacak. |

### İP-4 Raporlama

| # | Öneri | Karar | Not |
|---|---|---|---|
| 4.1a | Karar düzeyi metriklerine (Sens., FRR, PPV, NPV) %95 bootstrap GA | **Kabul** | BUT QDB'de denek düzeyinde, harici setlerde kayıt düzeyinde yeniden örnekleme. |
| 4.1b | Eğitilmiş ağırlıkları GitHub'a koy (TRIPOD 22) | **Kabul, lisans kontrolüyle** | HuBERT-ECG CC BY-NC 4.0, d1-3B LFM1.0 lisanslı; yalnızca kendi eğittiğimiz ağırlıklar (CNN'ler, MLP, LR probları) paylaşılır. Push için ayrıca onay alınacak. |
| 4.1c | Hedef kullanıcılar, kullanıcı etkileşimi, alt grup farklılıkları paragrafları (3b, 27b, 3c, 14) | **Kabul** | Alt grup analizi (vücut yapısı, aktivite) veri setlerinde yok → yalnızca sınırlılık olarak yazılır, analiz yapılmış gibi sunulmaz. |
| 4.2 | Kalibrasyon kesim noktası (CITL) ve eğimi | **Kabul** | Kaydedilmiş tahminlerden hesaplanır; logit(p) üzerinde lojistik regresyon. 0/1'e kırpılmış olasılıklar (LR ve MLP'de çok sayıda tam 0/1 var) için logit kırpma eşiği önceden sabitlenip raporlanacak. |
| 4.3 | S30 hatasını düzelt; ek tabloları ~14'e indir | **Kısmen** | S30 hatası yok (bkz. §0). Sadeleştirme kabul: 3.2 sonrası alt küme tabloları kalkar. |
| 4.4 | GenAI beyanı | **Değiştirilerek kabul** | Rehberin önerdiği "the author independently executed all analyses" ifadesi **doğru değil** (analiz kodları yapay zekâ desteğiyle yazıldı ve çalıştırıldı). Beyan kısa ve standart olacak, ama gerçekte ne yapıldığını doğru söyleyecek. "LLM-as-a-judge" ifadesi kaldırılabilir; kullanım "taslak üzerinde dil/tutarlılık kontrolü" olarak doğru biçimde tanımlanır. Son metin yazarın onayına sunulacak. |
| 4.5 | Şekil 1: veri akışı + üç modlu mimari diyagramı | **Kabul** | |

### İP-1 Yeniden konumlandırma ve İP-2 literatür

- Başlık ve özet tek ürün yerine "alan kayması altında temel model temsilleri vs. göreve özgü sınıflandırıcılar" çerçevesine taşınır. Başlık seçimi yazara aittir.
- SigLIP2 NaFlex 400M Yöntemler'de açıkça adlandırılır (doğrulandı).
- Kronolojik "revizyon günlüğü" dili (Bölüm 2.1, 2.3, 2.5, 4.1) temiz protokol diline çevrilir. **Ancak** sonuçlar görüldükten sonra eklenen analizler (d1 probu, flatline düzeltmesi, kohort değişikliği) tek ve kısa bir "analiz planından sapmalar" paragrafında dürüstçe beyan edilir; gizlenmez (CLAUDE.md §4, §5.4). HuBERT-ECG'nin ilk ön-işlemesi "duyarlılık analizi" olarak sunulmaz; ek materyale "uygulama farkı" notu olarak iner veya çıkarılır.
- Kaynakça hedefi ~55–60 hakemli yayın; her kaynak scite/PubMed üzerinden künye ve iddia desteği açısından ayrı doğrulanır. Doğrulanamayan aday kaynakça dışında tutulur.

### İP-5 Hedef dergi

Yazarın kararı. Deneyler dergiden bağımsız; format işi en sona kalır.

## 2. Uygulama sırası

1. İP-3.3a + 3.1: özellik matrisi temizliği ve 988 pencerede birincil CV'nin yeniden çalıştırılması (tüm modeller, aynı katlar, aynı tohumlar).
2. İP-3.2: CinC 2011 998 kayıt çıkarımı ve skorlama (d1-3B dahil).
3. İP-3.4: yeniden uyarlanmış sezgiselin birincil hâle getirilmesi.
4. Bağımlı analizler: model GA'ları, ikili karşılaştırmalar (Holm), LODO, yerel kalibrasyon, CinC 2017 transferi.
5. İP-4.1a + 4.2: karar metrikleri GA'ları, kalibrasyon kesim noktası/eğimi.
6. İP-3.5: birleşik gecikme betiği; PIL raster ikincil analizi.
7. İP-3.3b: keşifsel flatline analizi.
8. İP-2 literatür doğrulama ve İP-1/İP-4 metin yeniden yazımı; Şekil 1.
9. TRIPOD+AI kontrol listesi güncellemesi; hedef dergi formatı.

Eski sonuç dosyaları üzerine yazılmaz; yeni çıktılar `results/v2_2026-11/` altına gider.

## 3. İlerleme kaydı

### 2026-11-05
Yazar kararları: yapay zekâ beyanı kısa ve doğru olacak; hedef dergi sonra seçilecek; uzun işler arka planda çalışacak.

Kod değişiklikleri (commit edilmedi; v1 commit 55095ed'de yeniden üretilebilir):
- `config.py`: `RESULTS_DIR` artık `ECG_RESULTS_DIR` ile değiştirilebilir (varsayılan `results/`, v1 davranışı aynı); `CINC2011_LEADII_FULL` eklendi.
- `features/sqi_extractor.py`: `MODEL_INPUT_FEATURE_NAMES` (11 SQI). d1-3B istemi değişmedi (testle doğrulandı).
- `models/tabular_baselines.py`, `models/multimodal_fusion.py`: girdi 11 özellik; tablosal modeller yanlış genişlikte hata veriyor.
- Deney/değerlendirme betikleri: sonuç yolları config üzerinden; model girdisi 11 özellik; CinC 2011 kaynağı 998 kayıt
  (`lodo_multisource`, `hubert_ecg_probe`, `d1_linear_probe`, `tabpfn_runner`, `external_validation`, `heuristic_refit`,
  `paired_comparisons.families`).
- Yeni: `experiments/cinc2011_full_extraction.py`, `experiments/v2_setup.py`, `experiments/feature_cache.py`
  (öznitelikleri segment_id ile eşler, konumsal eşleme yok), `experiments/v2_queue.sh`,
  `evaluation/decision_calibration.py` (karar metrikleri GA + kalibrasyon kesim noktası/eğimi), `tests/test_v2.py`.
- `d1_cinc2017_full.py --rescore`: önbelleğe yazmadan BUT-eğitimli probu yeniden uyarlar
  (eski `FULL_PRED` sabit olarak `results/` altına yazıyordu; düzeltildi).

Veri kontrolleri:
- CinC 2011: 998 kayıt (773/225). Önbellekteki 565 kaydın sinyali yerel PhysioNet kopyasıyla, görüntüsü yeniden çizimle,
  13 SQI'ı ve etiketi eski dosyayla birebir aynı (fark 0). Derivasyon II ADC kazancı 200/mV (5 µV adım).
- v2 BUT QDB kohortu: 988 pencere, 238 kullanılamaz, 15 denek; katlar v1 ile aynı. **Çıkarılan 59 pencerenin tamamı
  kullanılamaz sınıfında** (prevalans %28,4 → %24,1); v1 AUROC'u bu kolay pozitiflerle hafifçe yükselmiş olabilir.
- Testler: 11 → 22 test, hepsi geçti (kalibrasyon kodu scipy ile bağımsız olarak doğrulandı, 4 ondalık).

Çalışan / sırada:
- Birincil CV (988 pencere) arka planda; ardından `v2_queue.sh` (konsolidasyon, d1 ve HuBERT problarında 433 yeni kayıt,
  harici doğrulama, TabPFN, LODO, ikili karşılaştırmalar, GA'lar, karar/kalibrasyon).
- Henüz v2'ye taşınmayanlar: yuvalanmış eşikler, yerel kalibrasyon, F1b resmi bölünme, NSTDB, tohum tekrarları,
  sınıf-2 duyarlılığı, 12 derivasyonlu analizin 998'e genişletilmesi, gecikme birleştirme (tek başına), PIL raster
  analizi, keşifsel flatline düzeltmesi, tablo/şekil üreticileri.

### 2026-11-06
- Kuyruk LODO aşamasında oturum kapanınca kesildi; `START_AT=lodo` ile devam ettirildi (kesilen günlük
  `logs/lodo_interrupted_2026-11-06.log`). Önceki aşamalar tamamlanmıştı.
- d1-3B kontrolü: BUT QDB'de yeniden hesaplanan sıfır-örnekli skorlar saklı skorlarla aynı (maks. fark 0,0000);
  CinC 2017 probu yeniden skorlandığında d1_linear_probe ile maks. fark 3e-8.
- **Multimodal ağ CinC 2011'de 998 kaydın hepsine 1,0 veriyor (AUROC 0,500, bağ).** Hata değil: CinC 2011'de
  flatline_fraction, BUT QDB dağılımının medyanda ~2.286 SD dışında (5 µV kuantizasyonu; CinC 2017'de 4,2 SD).
  Doyma yönü başlatmaya bağlı: v1'de kayıtların %88'i 0'a, v2'de hepsi 1'e gitti. LR (%92 tam 1) ve MLP (%93 tam 0)
  aynı durumda. Sonuç: SQI tabanlı modellerin CinC 2011 AUROC'ları bu artefakt tarafından belirleniyor ve tohumdan
  tohuma kararsız; makalede böyle raporlanmalı. Bu, 3.3b keşifsel analizinin gerekçesini güçlendiriyor.

### 2026-11-06 — 3.3b keşifsel analiz: önceden yazılmış tanım (CinC sonuçlarına bakılmadan önce)
Amaç: kuantizasyona duyarsız flatline ve doygunluk ölçüleriyle SQI tabanlı modellerin CinC 2011 transferinin
değişip değişmediğini görmek. **Keşifsel**: sorun CinC 2011 sonuçlarından sonra fark edildi; birincil analizin yerine
geçmez, makalede böyle etiketlenir.

Tanımlar (eşikler fizyolojik gerekçeyle sabitlendi, hiçbir veri setinde ayarlanmayacak; sinyal mV, 500 Hz):
- `flatline_fraction_robust`: tepe-tepe genliği ≤ 0,02 mV olan, en az 1,0 s uzunluğundaki herhangi bir pencerenin
  içinde kalan örneklerin oranı. Gerekçe: kalp hızı ≥ 60/dk olan bir EKG'de 1 s içinde en az bir QRS (≫ 0,02 mV) olur;
  0,02 mV, 5 µV'luk ADC adımının 4 katıdır, bu yüzden kuantizasyon tek başına bir pencereyi düz gösteremez.
- `saturation_fraction_robust`: sinyalin min veya max değerinin aralığın %0,5'i içinde kalan ve en az 50 ms (25 örnek)
  süren kesintisiz koşulara düşen örneklerin oranı. Gerekçe: ADC kırpması rayda süren bir plato yaratır; normal R tepesi
  bu bantta 50 ms'den kısa kalır.

Plan: 11 model girdisinde iki eski ölçünün yerine bu ikisi konur; LR, LightGBM ve SQI MLP aynı ayarlarla
(1) BUT QDB 988 pencere 5-katlı CV, (2) BUT-yalnız eğitim → CinC 2011 (998) ve CinC 2017 (8.512, kayıt maks.)
çalıştırılır. Ayrıca her iki ölçünün veri seti ve sınıf başına dağılımı ile tek değişkenli AUROC'u raporlanır.
Karşılaştırma v2 birincil sonuçlarıyla, aynı birimlerde. Başka eşik denenmeyecek.
- **Değişiklik (aynı gün, CinC verisi skorlanmadan önce):** `saturation_fraction_robust` sentetik testte başarısız oldu
  (0,3 Hz'lik yavaş taban çizgisi çukuru temiz bir sinyalde 0,149 verdi: çukur 50 ms boyunca %0,5 bandında kalıyor).
  Eşik oynatılmayacak; ölçü analizden çıkarıldı. Keşifsel analiz tek değişkene indirildi: `flatline_fraction` yerine
  `flatline_fraction_robust`; `saturation_fraction` dahil diğer 10 girdi aynı kalır. Sentetik kontroller:
  kuantize (5 µV) temiz sinyal eski 0,972 → yeni 0,000; 5 s elektrot kopukluğu → 0,500.

### 2026-11-06 — 3.3b keşifsel sonuçlar (results/v2_2026-11/flatline_robust_summary.csv)
AUROC, özgün flatline → kuantizasyona duyarsız flatline (aynı modeller, ayarlar, tohum, birimler):
- CinC 2011 (998): SQI MLP 0,675 → 0,737; LR 0,322 → 0,431; LightGBM 0,752 → 0,759. Farklı skor sayısı MLP'de 15 → 909,
  LR'de 28 → 925 (doyma kalktı).
- BUT QDB (988): MLP 0,904 → 0,916; LR 0,899 → 0,903; LightGBM 0,957 → 0,956. CinC 2017: değişim ≤ 0,011.
- Güven aralığı ve eşleştirilmiş test hesaplanmadı (tek keşifsel karşılaştırma); fark büyüklükleri bu haliyle raporlanmalı.

Yorum için zorunlu not:
- Yeni ölçü **BUT QDB'nin 988 penceresinin hepsinde 0** (düz sinyalli tek pencereler çıkarılan 59 sinyal-kaybı penceresiydi).
  Eğitimde sabit olduğu için modele bilgi katmıyor; görülen iyileşme kuantizasyon artefaktının girdiden çıkmasından
  geliyor, yani etkisi özgün özelliği düşürmeye eşdeğer. Rehberin "matematiksel düzeltme transfer çöküşünü çözer"
  iddiası desteklenmiyor: LR hâlâ 0,5'in altında, LightGBM değişmedi.
- CinC 2011'de yeni ölçü çok seçici: kabul edilemez 225 kaydın 86'sında > 0, kabul edilebilir 773 kaydın hiçbirinde
  değil (tek değişkenli AUROC 0,691). Bu bilgi BUT QDB'den aktarılamıyor çünkü eğitim verisinde örneği yok.
- LR'nin 0,5 altında kalması için kalan aday: saturation_fraction da kuantizasyona duyarlı (CinC 2011'de medyan 22 SD).
  Bu bir hipotezdir; yeni bir varyant denenmedi (önceden yazılmış plan tek değişiklikle sınırlıydı).

### 2026-11-06 — v2 kuyruğu tamamlandı (results/v2_2026-11/)
Doğrulama: model_cis AUROC'ları ham tahmin dosyalarından sklearn ile yeniden hesaplandı (2D CNN, LightGBM; BUT QDB ve
CinC 2011), 4 ondalıkta aynı. HuBERT CinC 2011 AUROC'unun v1 ile aynı çıkması (0,562) tesadüf: olasılıklar farklı
(aynı 565 kayıtta v1 0,562, v2 0,571).

Ana bulgular (AUROC [%95 GA]; Holm düzeltmeli en iyiye karşı test, α = 0,05):
- BUT QDB 988: en iyi 2D CNN 0,985; 1D CNN, kural, d1 probu, multimodal, LightGBM, TabPFN, HuBERT en iyiden ayırt
  edilemiyor. v1'e göre tüm modeller düşük (LR 0,938 → 0,899, HuBERT 0,972 → 0,938 en büyük düşüşler).
- CinC 2011 998: en iyi d1 image-token probu 0,824 [0,789–0,856]; **diğer 10 modelin hepsi anlamlı olarak düşük**.
  Multimodal ağ 0,500 (tüm skorlar 1,0; doyma), LR 0,322.
- CinC 2017 8.512: en iyi HuBERT probu 0,906; LightGBM 0,891, d1 probu 0,894, kural 0,887 ayırt edilemiyor.
- Kalibrasyon (CinC 2017, prevalans %3,3): tüm modellerde kesim noktası negatif (−2,5 ile −14,7) → BUT QDB'den
  (prevalans %24,1) aktarımda risk sistematik olarak aşırı tahmin ediliyor. Eğim CNN'lerde çok küçük (1D 0,10, 2D 0,35:
  aşırı uç olasılıklar), HuBERT probunda 0,93. BUT QDB CV'de kesim noktası yorumlanmayacak (iç doğrulama; Van Calster 2019).
- LODO: d1 probu CinC 2017'de çok kaynaklı eğitimle iyileşiyor (0,894 → 0,922, p < 0,001); CinC 2011'de değişmiyor.

Henüz v2'de yapılmayanlar: yuvalanmış eşikler, yerel kalibrasyon, F1b resmi bölünme, NSTDB, tohum tekrarları, sınıf-2,
12 derivasyonun 998'e genişletilmesi, gecikme birleştirme + PIL raster, tablo/şekil üreticileri, metin.

### 2026-11-07 — Devir notu madde 1: kalan analizlerin v2'ye taşınması
Kod değişiklikleri (commit edilmedi):
- `butqdb_official_split.py`: **hata düzeltildi** — d1 ve HuBERT prob öznitelikleri eski 1.047 satırlık önbellekten
  konumsal okunuyordu; 988 pencerede yanlış eşleşme olurdu. Artık `feature_cache.load_features` ile segment_id'ye göre.
  TabPFN F1b'ye eklendi (`--split-only` → `tabpfn_runner.py --official-split` → tam çalıştırma). v2 kohortunda resmi
  aralık dışında pencere kalmadığından F1a atlanıyor (birincil analizle aynı olurdu).
- `exp4_revised_nstdb.py`: **hata düzeltildi** — v2 değişikliği d1-3B istemine 13 yerine 11 SQI veriyordu. Metin
  varsayılanlar (10 s, 500 Hz) sayesinde aynı kalırdı, yine de istem girdisi v1'deki gibi 13 ada döndürüldü.
  Şekil artık `ECG_FIGURES_DIR` altına yazılıyor (`config.py`'ye eklendi; varsayılan `figures/`, v1 aynı).
- `nested_thresholds.py` transfer kısmı: 565/rastgele çekiliş dosyaları yerine `paired_comparisons.families()`
  (CinC 2011 998, CinC 2017 8.512; TabPFN, HuBERT ve d1 probu dahil). İç içe CV kısmı değişmedi.
- `local_recalibration.py`, `cinc2011_constant_lead_check.py`: aile adı 998 kayda; `seed_stability.py`: CinC 2011
  150 kayıt → 998; `prevalence_subjects_features.py`: kota kohort satırı manifestten, resmi aralıklar içi doğal
  prevalans eklendi, öznitelik kayması CinC 2011 998; `seed_repeats_transfer.py`, `butqdb_class2_sensitivity.py`:
  yalnız belge metni (veri yolları zaten v2).
- Yeni: `experiments/v2_queue_part2.sh`; `tests/test_v2.py` +3 test (27 test, hepsi geçti).

Eğitimsiz analizler tamamlandı (`results/v2_2026-11/`):
- Sabit derivasyon: CinC 2011 998'de **73** kayıt sabit derivasyon II, hepsi kabul edilemez (v1 565'te 65).
  Bunlar çıkarılınca d1 probu 0,824 → 0,740, LR 0,322 → 0,477, HuBERT 0,562 → 0,684.
- Doğal prevalans: resmi aralıklar içindeki 28.276 aday pencerede Sınıf 3 oranı **%2,14** (tüm kayıtlarda %16,0;
  fark 105001'in dışlanan bölgesinden). v2 kota kohortu %24,1. Prevalansa göre yeniden ağırlıklandırılmış metrikler
  bu değeri de içeriyor; makalede PPV için hangi prevalansın birincil olacağına karar verilmeli.
- Eşik transferi ve yerel kalibrasyon: v2 paydaları (988/998/8.512) testle doğrulandı; LightGBM ve 2D CNN için
  Hedef-1 duyarlılık/YPO bağımsız bir uygulamayla 4 ondalıkta aynı.
- Resmi bölünme: eğitim 546 pencere (146 kullanılamaz, 8 denek), test 442 (92, 7 denek); denekler ayrık.

Eğitimli kuyruk (`logs/v2_queue_part2.log`), 2026-11-07 20:07 itibarıyla 5/6 aşama tamamlandı, hata yok:
- F1b resmi bölünme (test 442 pencere, 7 denek), AUROC: 1D CNN 1,000, HuBERT probu 1,000, TabPFN 0,995,
  kural 0,993, LightGBM 0,992, d1 probu 0,991, LR 0,980, MLP 0,973, 2D CNN 0,961, multimodal 0,935, d1-3B 0,723.
  En iyiye göre Holm p'lerinin hepsi 1,0 (7 denek; güç düşük). 11 AUROC tahmin dosyasından bağımsız olarak
  yeniden hesaplandı, aynı.
- NSTDB: d1-3B skorları ve SQI'lar v1 ile birebir aynı (en büyük fark 0); eğitilen modeller v2 kohortunda yeniden
  eğitildiği için farklı (beklenen).
- Sınıf-2 duyarlılığı: kullanılamaz 238 → 431 (v1 297 → 490); 2D CNN 0,985 / 0,985, d1-3B 0,828 / 0,760.
- İç içe eşikler + transfer: tamamlandı (`threshold_transfer_summary.csv`; TabPFN, HuBERT ve d1 probu dahil).
- Tohum kararlılığı (1–4): BUT CV 1D/2D 0,978–0,985; CinC 2011 transferi 1D 0,68–0,71, 2D 0,775–0,787
  (tohum 42: 0,695 / 0,787). Multimodal CinC 2011: 0,501 / 0,503 / **0,671** / 0,546 (tohum 42: 0,500), yani
  multimodal modelin "0,500" sonucu tohuma bağlı; makalede böyle belirtilmeli.
- Tohum tekrarları (LODO, tohum 42 + 1–4): tamamlandı 22:30, kuyruk exit 0. Paydalar: CinC 2011 998 kayıt,
  CinC 2017 8.512 kayıt (23.985 pencere), BUT QDB 988 pencere (17 kayıt, 15 denek). Özet AUROC'lar ham tahminlerden
  bağımsız olarak yeniden hesaplandı (en büyük fark 1e-16). Tutarlılık kuralı (5 tohumda aynı işaret ve
  |medyan Δ| ≥ 0,02; Δ = LODO − yalnız BUT) karşılananlar:
  - CinC 2011: LR +0,076, LightGBM −0,115, MLP −0,186, 1D CNN +0,078; 2D CNN (−0,019) ve multimodal (−0,002,
    aralık −0,129…+0,201) karşılamıyor.
  - CinC 2017: 1D CNN +0,116, 2D CNN −0,037, multimodal +0,053; LR (+0,020, sınırda), LightGBM ve MLP karşılamıyor.
  - LR ve LightGBM tohumdan bağımsız (sabit random_state); onlarda "5 tohumda tutarlı" bilgi taşımaz.
  - LODO'da BUT QDB testinde multimodal 0,736–0,970 aralığında, yani tohuma çok duyarlı.

### 2026-11-08 — Devir notu madde 2–5 (devam ediyor)
- Madde 2: `cinc2011_twelve_lead_extraction.py` 998 kayda genişletildi (yeni dosya `cinc2011_twelve_lead_windows_full.csv`, 11.976 pencere; eski 565 kaydın
  SQI'ları ve etiketleri birebir aynı, fark 0). `experiments/d1_twelve_lead_full.py` çalışıyor (9.328 yeni pencere; 30 pencerelik yeniden üretim kontrolü fark 0).
  `external_validation.py --twelve-lead`: skorlar ayrı dosyaya gider, birincil tahminler yeniden yazılmaz.
- Madde 3: `experiments/latency_all.sh` (tek giriş noktası, tek başına çalışır), `experiments/pil_raster_2dcnn.py` (2D CNN raster şeritlerle yeniden eğitilir).
  Hata düzeltmeleri: `latency_foundation_probes.py` önbelleği satır sırasıyla okuyordu (segment_id'ye geçti); `latency_tabpfn.py` 13 girdi kullanıyordu (11'e indi).
- Madde 4: `build_manuscript_tables.py`, `generate_q1_figures.py`, `sync_manuscript_tables.py`, `assemble_q1_manuscript.py` v2'ye bağlandı. Alt küme tabloları (2d, 6) kaldırıldı;
  yeni tablolar 1h (karar metrikleri + kalibrasyon), 14 (raster), 15 (keşifsel flatline); yeni akış şeması. Tablo 7'nin CinC sütunu 150 kayıtlık alt kümede kaldı (d1 istem varyantları 998'e genişletilmedi).
- Sapma (plan 3.4): el ayarlı sezgisel Tablo 9'daki ikili karşılaştırma ailesinde kaldı; kat-içi yeniden uyarlanmış sürüm (BUT 0,905) metinde sızıntısız iç tahmin olarak verilecek.
- Kaynaklar: `references_v2_candidates_2026-11-06.md` sonuna özet okumasına dayalı iddia desteği eklendi (24 aday kullanılacak, 6'sı dışarıda).
- Kuyruklar: `v2_queue_part3.sh` (d1 → 12 derivasyon modelleri → raster → gecikme), `v2_queue_part4.sh` (ölçek duyarlılığı, HuBERT ince ayar, tablolar).

### 2026-11-09 — HuBERT-ECG probu 12 derivasyonda ve ek analizler

`experiments/hubert_probe_twelve_lead.py` (post hoc, ikincil): her derivasyon lead II yuvasında, BUT QDB'de eğitilen prob. AUROC: lead II 0,562; maksimum 0,431; ortalama 0,510; gerçek 12 derivasyonlu giriş 0,267 (prob tek derivasyonlu girişle eğitildi; derivasyon sırası varsayım). İlk çalıştırma sıra kontrolünde durdu: CinC 2011 dosyalarında aVF, aVL'den önce geliyor (`hubert_probe_twelve_lead_failed_lead_order.log`). Lead II'deki 998 pencere önbellekten geldiği için lead II eşleşme kontrolü bu pencerelerde döngüsel. Ek analizlerin planı: `ANALYSIS_PLAN_2026-11-09_additional_analyses.md`.

### 2026-11-08 (gece) — Yazar kararları ve d1 probu 12 derivasyonda

Başlık, birincil PPV prevalansı (%2,14) ve AI beyanı yazar tarafından taslaktaki hâliyle onaylandı. Hedef dergi açık. `experiments/d1_probe_twelve_lead.py` eklendi (post hoc, ikincil): BUT QDB'de eğitilen image-token probu, önbellekteki öznitelikler üzerinden 998 kaydın 12 derivasyonunda skorlandı. Toplama kuralları Tablo S5 ile aynı; dört DeLong testi çalıştırmadan önce sabitlendi. Lead II skorları ana analizle eşleşti (en büyük fark 2,9e-08). AUROC: maksimum 0,951, ortalama 0,944, lead II 0,824. Sabit derivasyonlu 135 kayıt olmadan (betimsel, sonuçlar görüldükten sonra eklendi): 0,902 ve 0,912. HuBERT-ECG probu ve TabPFN 12 derivasyonda skorlanmadı. Çıktılar: `results/v2_2026-11/d1_probe_twelve_lead_*.csv`; Tablo S5 ve yeni Tablo S35.

### 2026-11-08 (akşam) — Madde 2–3 sonuçları ve madde 5 taslağı
- 12 derivasyon (998 kayıt, `cinc2011_twelve_lead_summary.csv`): yeniden uyum ile birincil lead II farkı 0; AUROC'lar bağımsız sıra istatistiğiyle yeniden hesaplandı (en büyük fark 2e-16).
  Ortalama: 2D CNN 0,882, LightGBM 0,854, MLP 0,840; maksimum: kural 0,842, MLP 0,838, d1 sıfır-örnekli 0,813. Problar ve TabPFN 12 derivasyonda skorlanmadı; bu yüzden "d1 probu CinC 2011'de en iyi" iddiası yalnız tek derivasyon için geçerli (metinde yazıldı).
- Raster 2D CNN (`raster_2dcnn_summary.csv`): BUT 0,980 / 0,985 (Δ −0,005), CinC 2011 0,798 / 0,787 (Δ +0,012, p_Holm 0,025), CinC 2017 0,852 / 0,850.
- Gecikme: **hata** `empirical_latency_benchmark.py` LR ve MLP parametre sayılarını elle 13 girdiye göre yazıyordu (14, 3.201 → 12, 3.073). Düzeltildi; zamanlama etkilenmez ama tablo üretilmiş çıktıdan gelsin diye oturum `v2_queue_part5.sh` ile tekrarlanıyor (ilk oturum `latency_first_session/`).
- Metin: `manuscript/v2/` (ana metin, ek, kaynaklar, TRIPOD). `assemble_q1_manuscript.py` artık v2 dosyalarına yazıyor, v1 `main_q1.md` korunuyor. Kaynaklar `[@anahtar]` ile yazılıyor, ilk geçiş sırasıyla numaralanıyor; kaynakta bulunmayan/atıfsız anahtar hata veriyor.
  `evaluation/verify_v2_prose.py`: metindeki sayıları CSV'lerden yeniden üretip arar.
- Crossref ile düzeltilen künye: Zhu 2020 başlığı "...flexible model integration" (aday tablosunda "selection" yazıyordu); ECGFounder başlığı kısa haliyle.

### 2026-11-09 (akşam) — Ek analizler sonuçlandı (`ANALYSIS_PLAN_2026-11-09_additional_analyses.md`)

- **PTB-XL dondurulmuş test** (`experiments/ptbxl_frozen_test.py`; Tablo S38–S39; `--analyse` bir kez çalıştırıldı). Birincil: global gürültü - temiz, 600 kayıt. d1 probu 0,579 (maks.) ve 0,592 (ort.), 2D CNN 0,501 ve 0,521; fark +0,078 ve +0,071, ikisinde de p_Holm = 0,001: plandaki hipotez desteklendi. Ancak probun mutlak ayrımı zayıf; HuBERT-ECG (0,736/0,694) ve LightGBM (0,689/0,694) probun anlamlı olarak üstünde (p_Holm < 0,001). Sonuç özet, tartışma ve sonuca bu haliyle yazıldı; "en tutarlı model" iddiası kullanılabilirlik etiketleriyle sınırlandı. Çıktı bir vekil (annotated noise). AUROC'lar sıra tabanlı ayrı hesapla örtüştü (fark ≈ 1e-16), bootstrap aralıkları DeLong ile uyumlu (çapraz kontrol, bağımsız hakem değil).
- **Kod hatası (sonuçlardan önce):** `--score-others` başındaki yeniden üretim kontrolü `lead_idx` eksikliğinden durdu; yalnızca kontrol düzeltildi, sonra kontrol 0,000000 farkla geçti. Plan dosyasına işlendi.
- **Kalite eleme, CinC 2017** (`experiments/cinc2017_quality_gating.py`; Tablo S37). Elemesiz makro-F1 0,719; %10 elemede rastgeleye göre artış 0,001–0,008 (d1 probu 0,008, 95% GA 0,005–0,013). Etki küçük; çok gürültülü kayıtlar analize giremedi.
- **Multimodal doygunluk** (`experiments/multimodal_saturation_diagnosis.py`; Tablo S36). 0,500 değeri flatline girdisinin ölçek artefaktı: medyan z = 2.286; yalnız görüntü akışı 0,801; flatline değişince 0,719.
- Kontroller: `verify_v2_prose.py` 110 kontrol 0 hata; 27 birim testi geçti; özet 249 kelime. Düzeltme: HuBERT ortalama değeri 0,510'dur (sohbette önce 0,511 yazılmıştı). Commit yapılmadı.

### 2026-11-11 — Bağımsız kod denetimi ve kod revizyonu

Denetim raporu: `REVIEW_2026-11-11_code_audit_v2.md` (salt-okunur; sızıntı, etiket/hizalama veya metrik hatası bulunmadı;
kayıtlı AUROC'lar bağımsız hesapla aynı; durum: MAJOR REVISION REQUIRED). Bulgulara göre yapılan kod değişiklikleri (commit edilmedi):

- **CODE-001** `evaluation/audit_sensitivity.py` (yeni): CinC 2011'de sabit derivasyon II'li 73 kayıt olmadan (925 kayıt, 152
  kabul edilemez) ikili DeLong ailesi. d1 probu 0,740, 2D CNN 0,716; fark 0,024 (%95 GA −0,002…0,050), p = 0,069,
  p_Holm (55 çift) = 0,833, en iyiye karşı aile p_Holm = 0,069. Diğer dokuz model probun anlamlı olarak altında.
  **Sonuç: "CinC 2011'de en iyi (p = 0,003)" ifadesi bu 73 kayda bağlı; özet ve Bölüm 3.2 buna göre yeniden yazılmalı (yapılmadı).**
- **CODE-003** `paired_comparisons.cluster_bootstrap_paired`: p artık 1/geçerli-yeniden-örnekleme sayısının altına inmez
  (önce p = 0 dönebiliyordu). Yalnız sıfırlar değişir. **Kayıtlı `paired_comparisons.csv` ve türevleri yeniden üretilmedi**
  (BUT QDB ailesindeki 4 karşılaştırmada p = 0 duruyor; yeniden üretilince 0,0005 olur, Holm sonrası 0,0275, anlam değişmez).
  Ayrıca denek bazında AUROC tablosu: `butqdb_per_subject_auroc.csv`.
- **CODE-004** CinC 2017, kayıt başına pencere sayısına göre AUROC (`cinc2017_length_stratified.csv`): pencere sayısı tek
  başına 0,698; 3 pencereli kayıtlarda (7.561; 143 gürültülü) maksimum kuralıyla HuBERT 0,953, kural 0,951, LightGBM 0,923,
  d1 probu 0,924. Betimsel. `paired_comparisons.cinc2017_windows()` `families()` içinden ayrıldı (sonuç aynı; testle doğrulandı).
- **CODE-008** `run_supervised_cohort_cv.py`: belge metni düzeltildi (iç doğrulama eşikleri kaydedilmiyor ve kullanılmıyor). Kod yolu aynı.
- **CODE-009** `external_validation.score_windows`: eski 565 kayıtlık 12 derivasyon tablosu artık `external_window_predictions.csv`'ye
  yazılmıyor. Mevcut dosya değiştirilmedi (6.780 eski satır duruyor; `dataset` ile süzmeye devam).
- **CODE-011** PTB-XL örneği: 900 kayıt, 900 farklı hasta (`ptbxl_frozen_sample_patient_check.csv`; yalnız üst veri okundu,
  `--analyse` yeniden çalıştırılmadı). HuBERT-ECG ön-eğitim örtüşmesi hâlâ doğrulanmadı.
- Testler: 27 → 35, hepsi geçti. Çalıştırma kaydı: `logs/audit_sensitivity.log`, `audit_sensitivity_provenance.json`.

Yapılmayanlar (yazar kararı / onayı gerekir): CODE-002 (kohort tanımı metni; 105001'in resmi aralık içinde 54 Sınıf 3 penceresi var,
kohortta 1 — yeniden örnekleme tüm modellerin yeniden eğitilmesini gerektirir), CODE-005 (commit, göreli yollar), CODE-006 ve
CODE-007 (özet ve Bölüm 2.6 metni), yeni tabloların makaleye eklenmesi, `paired_comparisons.py` ve tablo zincirinin yeniden üretilmesi.

### 2026-11-11 (devam) — Denetim sonuçlarının makaleye işlenmesi (yazar onayıyla)

- `paired_comparisons.py` ve `paired_vs_best.py` p tabanıyla yeniden üretildi. Eski çıktılar `results/v2_2026-11/before_pvalue_floor_2026-11-11/`
  altında. Fark: yalnız BUT QDB ailesinde p = 0 olan 4 satır (p 0 → 0,0005; p_Holm 0 → 0,0275); en iyiye karşı tabloda LR ve MLP
  p_Holm 0,028 (55 çift) / 0,005 (10 karşılaştırma). Başka hiçbir sayı değişmedi (1e-12 toleransla doğrulandı).
- `audit_sensitivity.py`: 5. çıktı eklendi (`butqdb_quota_candidates_by_subject.csv`). 105 numaralı denek: tüm aralıklarda 4.734, resmi
  aralık içinde 54 Sınıf 3 aday penceresi; kohortta 1. Kota resmi aralık içinde uygulansaydı 291 kullanılamaz pencere olurdu (şimdi 238).
- Yeni ek tablolar: S40 (CinC 2011, sabit derivasyonsuz ikili testler), S41 (denek bazında AUROC), S42 (CinC 2017, pencere sayısına göre),
  S43 (kota ve resmi aralıklar). Üretici: `build_manuscript_tables.py` tablo 21–24.
- Metin (`manuscript/v2/main_v2_template.md`): özet (246 kelime, aynı sayaçla önce 245): "best on CinC 2011" yerine 925 kayıtlık sonuçla
  nitelenmiş ifade; "eight approaches (one hand-set)"; Bölüm 2.1 kohort tanımı (kota kısıtlamadan önce uygulandı); 2.5 (p tabanı, yeni
  analizler); 2.6 madde 4 (prob temsilinin seçimi önceden belirlenmemişti; yazar onayıyla yalnız belgelenebilir olgu yazıldı: iki temsil aynı çalıştırmada, commit 8417435) ve yeni madde 8; 3.1, 3.2,
  tartışma, sınırlılıklar ve sonuç.
- Kontroller: `verify_v2_prose.py` 121 kontrol, 0 hata (11 yeni kontrol); 35 birim testi geçti; üretilen tablo dosyasında eski tablolardaki
  tek fark Tablo 9 ve 9b'deki dört p hücresi.
- Yapılmadı: TRIPOD+AI kontrol listesi, docx/pdf dışa aktarımı, CODE-002 için yeniden örnekleme, CODE-005 (commit). Commit yapılmadı.

### 2026-11-11 (akşam) — Yeniden çerçeveleme (yazar onayıyla; `manuscript/v2/REFRAMING_PROPOSAL_2026-11-11.md`)

- Birincil soru değişti: "alan-içi başarı, aktarılan modelleri belirliyor mu?" birinci, temel-model sorusu ikinci. Bölüm 2.6'ya madde 9
  olarak beyan edildi (çerçeve tüm sonuçlar görüldükten sonra değişti). Yeni deney veya yeniden eğitim yok.
- Yeni başlık ve özet (246 kelime). Yeni betimsel çıktı: `rank_agreement.csv`, `rank_agreement_spearman.csv`
  (`audit_sensitivity.py` 6. analiz; Tablo S44). Spearman ρ (11 model): BUT QDB–CinC 2011 0,61 (p = 0,047), BUT QDB–CinC 2017 −0,02,
  CinC 2011–CinC 2017 0,19. n = 11 olduğu için kesinliği düşük; iddia "güvenilir biçimde belirlemiyor" olarak yazıldı.
- Sonuçlar yeniden sıralandı (3.1 alan-içi, 3.2 sıra uyumu, 3.3 sıralamayı değiştiren etkenler, 3.4 çok kaynaklı eğitim, 3.5 eşikler,
  3.6 PTB-XL). Yedi paragraf değiştirilmeden eke taşındı (Ek Sonuçlar S-R1–S-R7: Sınıf 2 ve resmi bölünme, HuBERT ince ayarı, keşifsel
  flatline, sezgisel–LR, NSTDB, hesaplama maliyeti, kalite eleme). Ana metin Tablo 5 (gecikme) → Tablo S45.
- Tartışma ve sonuç yeni birinci soruyla açılıyor; harici değerlendirme için üç kontrol önerisi eklendi.
- `verify_v2_prose.py`: taşınan cümleler diğer dosyada da kabul ediliyor (sayı yine kontrol edilir); 129 kontrol, 0 hata, 21'i ekte bulundu.
  36 birim testi geçti.
- Uzunluk: ana metin şablonu 10.871 → 10.560 kelime. Öneri dosyasındaki 7.000–8.000 kelime hedefi yazar kararıyla geri çekildi
  (hedef dergi belli değil); kısaltma dergi seçildiğinde, onun sınırlarına göre yapılacak. Eke taşınan paragraflar odak için taşındı.
- Özetten çıkanlar (ana metinde duruyor): d1 probunun "en iyinin anlamlı olarak gerisinde değil" bulgusu, sezgiselin elle ayarlı olduğu notu,
  PTB-XL'in vekil sonuç olduğu notu. Commit yapılmadı.

### 2026-11-11 — Bağımsız bilimsel hakem raporu (PAPER-001…012) ve uyumlaştırılmış boru hattı analizi: ÖN TANIM

Hakem raporu (yazarın ilettiği "Independent scientific review", manuscript v2): 12 bulgu depoyla karşılaştırıldı. PAPER-007 yanlış
(0,736 doğru: herhangi bir derivasyonu sabit 135 kayıt olmadan HuBERT derivasyon II; cümle belirsiz), diğerleri geçerli veya kısmen
geçerli. Yazar kararı (2026-11-11): metin düzeltmeleri, kaynak doğrulaması, dalın push edilmesi, gecikmenin eşit sınırlarla tekrarı ve
uyumlaştırılmış boru hattı karşılaştırması yapılacak.

**PAPER-002 için ön tanım (hiçbir uyumlaştırılmış skor hesaplanmadan önce yazıldı; post hoc, ikincil, birincil sonuçların yerine
geçmez).** Amaç: bilinen girdi-ölçek/kuantizasyon uyumsuzluğu giderildiğinde model sıralamasının ve sıra uyumunun ne kadar değiştiğini
göstermek. Üç önlem; hiçbiri hedef veri setlerinin etiketlerine veya başarımına bakılarak ayarlanmaz:

- **G1 Sabit girdi kuralı.** Standart sapması 0 olan pencere, modelden önce "kullanılamaz" (skor 1) sayılır. Gerekçe: değişimi olmayan
  sinyalde QRS olamaz. 11 modelin tümüne uygulanır (yeniden eğitilmeyenlerde kayıtlı skorlara son-işlem olarak).
- **G2 Kuantizasyona duyarsız flatline.** `flatline_fraction` yerine 2026-11-06'da önceden tanımlanan `flatline_fraction_robust`
  (1,0 s, 0,02 mV; değiştirilmeden). Tablosal girdisi olan modeller: LR, LightGBM, SQI MLP, TabPFN, multimodal ağ (SQI kolu) ve
  elle ayarlı sezgisel (yalnız bu girdi; katsayılar aynı).
- **G3 Standartlaştırılmış girdilerin kırpılması.** Eğitim katının ortalama ve SS'iyle z-skoru, sonra [−3, 3] aralığına kırpma. Sınır,
  v1'den beri Tablo S15'teki "Z-clip" ölçekleyicisinin sınırıdır; yeni seçilmedi. LR, LightGBM, SQI MLP, TabPFN ve multimodal ağa
  uygulanır (sezgisel ham değer ve kendi tavanlarını kullandığı için uygulanmaz).

Diğer her şey birincil analizle aynı: aynı katlar, hiperparametreler, tohum 42; MLP ve multimodal ağ ayrıca tohum 1–4 (yalnız BUT-yalnız
aktarım). 1D CNN, 2D CNN, sıfır-örnekli d1-3B ve iki prob yeniden eğitilmez (tablosal girdileri yok); yalnız G1 uygulanır.
Değerlendirme: BUT QDB 5 katlı OOF (988), CinC 2011 derivasyon II (998), CinC 2017 (8.512, kayıt maksimumu).
Raporlanacak: model ve veri seti başına özgün ve uyumlaştırılmış AUROC, eşli fark ve %95 GA (BUT: denek küme bootstrap; harici: DeLong;
çoklu karşılaştırma düzeltmesi yok, betimsel), farklı skor sayısı, |z| > 3 olan standartlaştırılmış girdi oranı (veri seti ve özellik
başına), tohum aralığı, ve uyumlaştırılmış AUROC'larla Spearman sıra uyumu. Eşik, kalibrasyon ve LODO yeniden yapılmaz.
Yorum kuralı: sonuç ne çıkarsa çıksın raporlanır; uyumlaştırma sıralamayı değiştirirse ana metindeki iddialar "test edilen boru hatları"
ile sınırlandırılır, değiştirmezse bu da yazılır. Başka sınır veya tanım denenmeyecek.

**PAPER-009 için:** `empirical_latency_benchmark.py` sezgisel dalı, CNN dallarıyla aynı sınırdan (kayıtlı 500 Hz pencere → karar)
ölçülecek: görüntü dosyadan açılmayacak, `render_ecg_image` ile çizilecek. Oturum `latency_all.sh` ile tek başına tekrarlanır; önceki
çıktılar `latency_before_equal_boundaries_2026-11-11/` altında saklanır.

### 2026-11-11 (gece) — Hakem raporu PAPER-001…012: yapılanlar ve sonuçlar

- **PAPER-002 uyumlaştırılmış boru hattı** (ön tanım commit 999c419; `experiments/harmonised_pipeline.py`, `harmonised_tabpfn.py`,
  `harmonised_inputs.py`; Tablo S46, S47). CinC 2011 AUROC, özgün → uyumlaştırılmış: LR 0,322 → 0,766; multimodal 0,500 → 0,731
  (tohumlar 0,706–0,768); MLP 0,675 → 0,761 (0,686–0,776); LightGBM 0,752 → 0,779; TabPFN 0,731 → 0,741; sezgisel 0,678 → 0,738;
  yalnız sabit-girdi kuralıyla HuBERT 0,562 → 0,787, 1D CNN 0,695 → 0,770, 2D CNN 0,787 → 0,808; d1 probu 0,824 ve sıfır-örnekli
  0,641 değişmedi. Sıfır-örnekli dışındaki on modelin aralığı 0,322–0,824'ten 0,731–0,824'e daraldı. BUT QDB'de en büyük değişim 0,024,
  CinC 2017'de 0,058 (multimodal 0,799 → 0,858). Spearman (11 model): BUT–CinC 2011 0,61 → 0,49; BUT–CinC 2017 −0,02 → −0,19;
  CinC 2011–CinC 2017 0,19 → 0,43. **Yorum (ön tanımdaki kurala göre):** CinC 2011'deki düşük AUROC'ların çoğu girdi boru hattı
  hatasıydı; alan-içi ile harici sıra arasındaki zayıf ilişki ise uyumlaştırmayla düzelmedi. Model ailelerine ilişkin ifadeler "test
  edilen boru hatları" ile sınırlandı. Bu karşılaştırmada çoklu test düzeltmesi ve prob–2D CNN eşli testi yok (betimsel).
- **PAPER-009 gecikme:** sezgisel artık şeridi çiziyor (önbellekten açmıyor). Oturum `latency_all.sh` ile tek başına tekrarlandı
  (17:23–17:31); önceki çıktılar `latency_before_equal_boundaries_2026-11-11/`. Sezgisel toplam 2,53 → 62,70 ms; diğerleri: 2D CNN
  63,62 → 65,92; 1D CNN 1,28 → 1,31; d1-3B 894 → 896; HuBERT probu 33,1 → 28,0 ms. Metindeki tüm gecikme sayıları güncellendi ve
  artık `verify_v2_prose.py` tarafından kontrol ediliyor (önce yalnız bir kez doldurulmuş, kontrolsüzdü). S-M5'teki "25 pencerenin
  19'unda" önbellek açıklaması yeniden doğrulanmadı.
- **Metin:** PAPER-003 (prevalans "projeksiyon" olarak yeniden adlandırıldı; %16,0 yorumlanmıyor), PAPER-004 (başlık, özet, giriş,
  2.1'e heterojenlik paragrafı, tartışma, sonuç: kapsam "incelenen heterojen ortamlar"), PAPER-006 (özette tek aile: 0,003 / 0,833),
  PAPER-007 (0,736 doğruydu; cümle hangi kayıtların çıkarıldığını söylüyor), PAPER-008, PAPER-010, PAPER-011, PAPER-012, TabPFN yorumu.
  Özet 245 kelime; PTB-XL ve yerel yeniden ayarlama dışındaki ayrıntılar için yer açmak üzere PTB-XL özetten çıkarıldı (ana metinde 3.6).
  Başlık değişti; son karar yazarın.
- **PAPER-005 kaynaklar:** ST-MEM (ICLR 2024), MERL (PMLR 235:31949–31963), K-MERL (Findings of EMNLP 2025, s. 7298–7316,
  doi:10.18653/v1/2025.findings-emnlp.385) resmi kayıtlardan doğrulanıp değiştirildi. PULSE, ECG-Chat, TabPFN-3 ve HuBERT-ECG için
  nihai yayın bulunamadı; ön baskı/teknik rapor olarak işaretli. Diğer kaynakların tek tek iddia desteği bu turda yeniden denetlenmedi.
- **PAPER-001:** dal uzak depoya push edildi (aşağıdaki commit).
- Kontroller: `verify_v2_prose.py` 145 kontrol, 0 hata; 39 birim testi geçti.

### 2026-11-12 — Hakemin ikinci raporu (PAPER-003/005/007/012 kalanları, yeni PAPER-013…015); yazar onayıyla

Yeni deney yok; yalnız metin, kaynakça, tablo üretici ve kontrol listesi. Hakemin yedi iddiası depoda doğrulandı.
- **PAPER-013:** Bölüm 2.4, 2.6 (10), 3.3 tartışma cümlesi, özet ve Tablo S46 başlığı: uyumlaştırılmış analiz artık "sonuçlardan
  haberdar duyarlılık analizi" olarak tanımlı. Önlemler hedef etiketi kullanmıyor ve skorlardan önce commit edildi (999c419), ancak
  önceki harici sonuçlar bilinerek seçildi (sabit derivasyonlar, flatline ölçüsü, S15'teki [−3, 3] sınırı). Özette "Pre-stated" →
  "Post hoc".
- **PAPER-014:** sıfır-örnekli d1-3B istemde sekiz SQI'yi metin olarak alır; bu değerler değiştirilmedi (yalnız G1). Metin ve
  `experiments/harmonised_pipeline.py` belge dizgisi düzeltildi; kod davranışı değişmedi.
- **PAPER-015:** "indistinguishable" iki yerde "no significant difference was detected" oldu.
- **PAPER-003:** sınırlılıklarda "re-weighted" → koşullu projeksiyon; Tablo S3 sütun başlıkları "projected".
- **PAPER-007:** Tablo S35'e derivasyon II sütunları eklendi (aynı kayıtlı CSV'lerden); 0,736 artık tabloda görünüyor.
- **PAPER-012:** kontrol listesi S1–S47; madde 2 için özet, TRIPOD+AI for Abstracts'ın 13 maddesine göre tek tek işaretlendi
  (kaynak: bildirimin Tablo 3'ü, PMC11019967); 12b, 13, 16, 18c, 23a, 23b güncellendi.
- **PAPER-005:** ECG-Chat yayımlanmış sürümüyle değiştirildi (IEEE ICME 2025, doi:10.1109/ICME59968.2025.11209476; Crossref ve
  yazarların deposundan doğrulandı). HuBERT-ECG (medRxiv), PULSE (arXiv) ve TabPFN-3 (arXiv teknik rapor) için hakemli sürüm
  bulunamadı; ön baskı olarak işaretli kaldı. Bunların kaynakçada kalıp kalmayacağı yazar kararı.
- Kontroller: `verify_v2_prose.py` 145 kontrol, 0 hata; 39 birim testi geçti. Commit edilmedi.

### 2026-11-12 (devam) — Hakemin üçüncü notu; görsel denetim ve bulunan tablo eksikliği

- **PAPER-013 kalıntısı:** Bölüm 3.3'teki "(Table S46; post hoc, pre-stated)" düzeltildi. `verify_v2_prose.py`'ye 16 ifade kontrolü
  eklendi (7 yasak eski ifade, 9 zorunlu yeni ifade).
- **Görsel denetim:** üç PDF'in bütün sayfaları (26 + 38 + 6; yeniden üretimden sonra ek 39) küçültülmüş sayfa görüntüleriyle
  incelendi (PDFKit ile; pdftoppm yok). Kırpılan tablo veya taşan metin görülmedi; şekiller yerinde. Küçültülmüş görüntüde hücre
  düzeyinde okuma yapılmadı.
- **Bulunan hata (kendi tespitimiz):** `sync_manuscript_tables.table_blocks` çok bölümlü tabloların yalnız ilk bölümünü alıyordu.
  Bu yüzden Tablo S36 (alt: yeniden skorlama), S37 (alt: eşli farklar), S44 (Spearman bölümü) ve S47 (orta: tohum aralığı; alt:
  girdi tanıları) ekte eksikti; başlıkları bu bölümleri vaat ediyordu. Commit 0dc93f4'teki ek bu eksikle push edildi. Sayılar
  metinde doğruydu (kontroller CSV'den), yalnız tablolar gösterilmiyordu. İşlev düzeltildi, birim testi eklendi (40 test).
- **Kaynakça politikası (yazar kararı, 2026-11-12):** PULSE (arXiv ön baskısı; yalnız girişte ilgili çalışma) kaynakçadan ve
  giriş cümlesinden çıkarıldı; cümle kalan yayımlanmış kaynakların desteklediği kapsama daraltıldı ("reading ECG images" →
  "ECG report generation"). HuBERT-ECG (medRxiv) ve TabPFN-3 (arXiv teknik rapor) bu çalışmada kullanılan modellerin birincil
  teknik tanımlarıdır; hakemli sürümleri bulunamadı ve hakem değerlendirmesinden geçmemiş kaynak olarak açıkça işaretli biçimde
  kaynakçada tutuluyor (istisna). PAPER-005 bu nedenle "kabul edilmiş sınırlılık" olarak kalır, "düzeltildi" değil. ECG-Chat ICME 2025 kaydı Crossref API yanıtıyla doğrulandı (10.1109/ICME59968.2025.11209476).
- Kontroller: 145 sayı + 16 ifade kontrolü, 0 hata; 40 birim testi geçti. Commit edilmedi.

### 2026-11-12 (son) — Dışa aktarım sürümü ve görsel denetim

Altı dışa aktarım (izlenmeyen `manuscript/v2/export/`) aynı üç kaynak dosyadan tek komutla, 2026-11-12 09:27'de üretildi
(pandoc + başsız Chrome, `evaluation/pmea_print.css`). SHA-256:

```
  2c194a3e8a012c881f8460fe7f89ef5a9d9698b3e3e47a6c0a50e156f3e84560  manuscript/v2/main_v2.md
  0a504b0108a7176e0ce3e57947887519091204fad33d209f98e778ff4fd6aa09  manuscript/v2/supplement_v2.md
  d203a95823aacff7bf6adb896e188645eecdede1fcf0e24222be2eba9e0568ff  manuscript/v2/tripod_ai_checklist_v2.md
  e0b50b424b3d5ec7b921c398f515f76751884b12cd4ce629083587886da94dcd  manuscript/v2/export/main_v2.docx
  303ce834d033d8b356f8666e0f41b03b676434c67e191d399a3ceac03b4b9079  manuscript/v2/export/main_v2.pdf
  a8a4db323cebed0fde1c0609841c68e22aac3d4a4fd5bd5c104b094b84372370  manuscript/v2/export/supplement_v2.docx
  49c65befb6b311f6c9f28ddf19dfc49d778965c14861034fe903e94361f2cc2c  manuscript/v2/export/supplement_v2.pdf
  905753282fbd47b8fd774d3fe0ecde4859e84bf443b9cd492d782a0e36bcb50b  manuscript/v2/export/tripod_ai_checklist_v2.docx
  def7deda5908eccf1ab4f30aaf586f728eba999ce74b3fa16a1512d13ccae0ab  manuscript/v2/export/tripod_ai_checklist_v2.pdf
```

PDF'ler yeniden üretimde zaman damgası taşıdığı için özetleri her üretimde değişir; kaynak `.md` özetleri belirleyicidir.
- Görsel denetim (bu sürüm): ana metin 26 sayfanın tamamı; ek 39 sayfanın 25–39. sayfaları (S31 sonu–S47; değişen bölge). Ekin
  1–24. sayfaları bir önceki üretimde incelendi ve kaynağı değişmedi. Kontrol listesi (6 sayfa) bir önceki üretimde incelendi,
  kaynağı değişmedi. Kırpılan tablo, taşan metin veya eksik şekil görülmedi; hücre düzeyinde okuma yapılmadı.
- Sonuç dosyaları: `git status results/` yalnız `manuscript_tables_2026-10-12.md` (üretilen tablo metni: S3 başlıkları, S35
  sütunları) değişikliğini gösteriyor; tahmin ve özet CSV'lerinin hiçbiri değişmedi.
- PAPER-005: belgelenmiş politika istisnası; hedef derginin kabulü doğrulanmadı (dergi seçilmedi).

### 2026-11-12 — Kayıtlı çıktılardan çapraz kontrol (PAPER-001/002/009 için kanıt; yazar onayıyla depoya eklendi)

`evaluation/crosscheck_saved_outputs.py` (proje modülü içe aktarmaz; salt-okunur; model eğitmez, süre ölçmez). Sonuç (c12de60
üzerindeki çıktılarla): Tablo 1'in 33 AUROC'u `model_cis.csv` ile en çok 1,1e-16 farkla; 33 özgün ve 33 harmonize AUROC
`harmonised_summary.csv` ile en çok 1,1e-16 farkla (sabit-girdi kuralı betikte ayrıca uygulanarak; CinC 2011'de 73 sabit kayıt,
hepsi kabul edilemez); Spearman 0,6091/0,4909, −0,0182/−0,1909, 0,1909/0,4273; on modelin CinC 2011 aralığı 0,322–0,824 →
0,731–0,824; tohum aralıkları MLP 0,686–0,776, çok kipli ağ 0,706–0,768; Tablo S45'in 24 medyan (p95) hücresi üç ham ölçüm
dosyasından aynen. Gecikme kodunda sekiz modelin hepsi için sinyal zamanlayıcıdan önce yükleniyor, süre SQI/çizimle başlıyor.
Sınır: bu ikinci bir hesaplamadır, bağımsız hakem doğrulaması değildir; harmonize eğitimler ve gecikme ölçümü yeniden
çalıştırılmadı.

### 2026-11-13 — Hedef dergi: Scientific Reports (yazar kararı); dergiye özel sürüm

Kılavuzlar Chrome'dan okundu: CMPB (ana metin normalde ≤ 3.500 kelime), BSPC (~5.000, yeni teknik beklentisi), MBEC (≤ 18 sayfa;
"var olan yöntemleri açık veri setlerine uygulayan" çalışmalar masada elenir), JBI ve AI in Medicine (yeni yöntem şartı),
Computers in Biology and Medicine (17 Kasım 2025'te Web of Science'tan çıkarıldı). Yazar Scientific Reports'u seçti: ana metin
≤ 4.500 kelime (yöntem hariç), başlık ≤ 20 kelime, özet ≤ 200 kelime başlıksız, ≤ 8 görsel öge, tablo ≤ 1 sayfa, ≤ 60 kaynak,
yayın ücreti £2.290 / $2.850 / €2.490 + KDV. Çeyreklik JCR'den doğrulanmadı.
- `manuscript/srep/`: ana metin 3.608 kelime (giriş 505, sonuçlar 1.779, tartışma 1.324), başlık 16, özet 197 kelime; 3 tablo,
  4 şekil; eski Tablo 4 → Ek Tablo S48 (Şekil 4 aynı bulguyu gösteriyor); CinC 2017 tablosu Tablo 2, LODO Tablo 3 (atıf sırası).
  Yöntem sona taşındı; tartışma alt başlıksız (sınırlılıklar ve sonuç içinde). İstatistik bölümüne "tüm testler çift yönlü,
  anlamlılık p_Holm < 0,05" eklendi (kodla doğrulandı: DeLong 2·sf, bootstrap 2·min). Yapay zekâ beyanı ve etik Yöntem'de.
  Etik: yalnız BUT QDB için yazılı onam belgede var (PhysioNet sayfası); PTB-XL için etik kurul yayın onayı (PTB-2020-1);
  CinC 2011, CinC 2017, NSTDB belgelerinde onam bilgisi yok, metin bunu söylüyor.
- Ek: numarasız ek yöntem/sonuç başlıkları, "Supplementary Table Sn" etiketleri, eski sürümün tam Sonuçlar metni
  "Supplementary Note: extended results" olarak (yalnız çapraz atıflar uyarlandı). Hiçbir analiz eklenmedi ya da çıkarılmadı.
- Kaynakça Nature biçimine çevrildi (50 kaynak). TRIPOD+AI listesi bu sürüme uyarlandı (özet maddesi yeniden değerlendirildi).
  Başvuru mektubu taslağı: önerilen/istenmeyen hakemler ve tarih yazar tarafından doldurulacak.
- Betikler: `assemble_q1_manuscript.py --variant srep`; `verify_v2_prose.py --variant srep` (srep'e özgü ifade kontrolleri ve sayı
  kaynağı kontrolü). Sonuç: 137 sayı + 15 ifade kontrolü 0 hata; ana metindeki 280 sayının hepsi doğrulanmış kaynaklarda; v2
  sürümü değişmedi (145 + 16, 0 hata); 40 birim testi ve çapraz kontrol geçti. Ana metin PDF'i 22 sayfanın tamamı gözle incelendi.
- Commit edilmedi.

### 2026-11-13 — Üçüncü hakem raporu (PAPER-001–012, Scientific Reports sürümü üzerinde); yazar "Evet, uygula"

Bulguların hepsi kod ve metinle doğrulandı. Hiçbir model yeniden eğitilmedi; yeni analizler kayıtlı çıktılardan.
- PAPER-003: `decision_calibration.py` kesim noktası artık skor denkleminin kökü (Brent), eğim adım yarılamalı Newton; durum
  sütunları eklendi. Eski çözücü sabit skorda NaN veriyordu (998 kayıt, 225 pozitif, p = 1−1e-6: analitik −15.0497). Eski çıktı
  `decision_calibration_pre_PAPER003_2026-11-13.csv` olarak saklandı. Değişen: CinC 2011'de 4 kesim noktası (LR, MLP, multimodal,
  zero-shot d1-3B) ve CinC 2017 1D CNN'in kesim noktası aralığı (önceden bazı bootstrap örnekleri NaN). Eğim: multimodal sabit
  skor → tanımsız; LR ve MLP yarı-tam ayrışma (74 kayıt alt sınırda, hepsi pozitif) → yakınsamaz (Nelder–Mead ile teyit).
  2 yeni birim testi (analitik sabit skor, ayrışma).
- PAPER-001/002: `evaluation/rank_uncertainty.py` → `rank_uncertainty.csv` (2.000 eşleştirilmiş bootstrap, seed 42; BUT QDB
  denek, dış veri kayıt). ρ: 0,61 [0,06–0,75], −0,02 [−0,35–0,32], 0,19 [0,04–0,31]; iç alan lideri dış veride lider: %0,5 ve
  %0,0; seçim kaybı 0,037 ve 0,057 AUROC. Sezgisel yöntem çıkarılınca ρ 0,62 ve −0,05. Nokta tahminleri rank_agreement ile
  birebir. Ek Tablo S44'e ikinci bileşen tablo. Özet ve sonuçlarda sezgisel yöntem nitelendi; sekiz model arasındaki 28 çiftin
  hepsi p_Holm ≥ 0,615 (paired_comparisons.csv) — mektuptaki "birbirinden farklı değil" ifadesi buna dayandırıldı.
- PAPER-006: `evaluation/local_threshold_triviality.py` → `local_threshold_triviality.csv`; Ek Tablo S22/S48 (v2 Tablo 4) iki sütun.
  n = 100: CinC 2017'de sıfır ret %17–95, yararlı kontrol %0–70; CinC 2011'de LR ve multimodal %100, 1D CNN %60.
- PAPER-004: BUT QDB 15 denek: 9 kadın, 6 erkek, 21–83 yaş (medyan 37) (subject-info.csv). v2'deki "10 kadın 8 erkek kayıt"
  kayıt düzeyiydi; denek düzeyine çevrildi.
- PAPER-005: istatistik envanteri düzeltildi (12 derivasyon prob testleri: Holm dört test; S26/S35 betimsel).
- PAPER-007: "matters more" ve "removes the tuning question" yapılandırmalarla sınırlandı; uyarlama bütçesi farkı ve tek-derivasyon
  HuBERT girişi sınırlılıklarda.
- PAPER-008: iki ön baskı "preprint" etiketiyle tutuldu (dergi izin veriyor; yazara bırakıldı).
- PAPER-009: README güncellendi; veri erişimine dal adı ve [COMMIT HASH AND ARCHIVE DOI] yer tutucusu (yazar dolduracak).
- PAPER-010: BenchECG/xECG npj Digit. Med. 2026 (doi:10.1038/s41746-026-03196-y; erken kabul sürümü, yayıncı sayfası okundu)
  eklendi; katkı cümlesi sınırlandı. Kaynak sayısı srep 51.
- PAPER-011/012: TRIPOD (madde 2, 19, 20b, çift önek), Etik'e hasta-kamu katılımı cümlesi; mektup ifadesi.
- Sapma maddesi eklendi (v2 (11), srep (12)). Özet 199 kelime; srep ana metin 4.031 kelime.
- Doğrulama: v2 154 sayı + 23 ifade, srep 146 + 22, 0 hata; srep 291 sayının hepsi kaynaklı; 42 test ve çapraz kontrol geçti.
  Değişen tablolar ve v2 Tablo 4 PDF'te gözle incelendi.
- Commit edilmedi.

### 2026-11-13 — V3: üçüncü hakemin takip yanıtı ("Bu yanıt.docx"); sürüm numaralandırması başladı

Yazar kararı: her düzeltme turu sürümü 1 artırır (V1 = PMEA gönderimi, V2 = bu tarihe kadarki durum, V3 = bu tur).
Kaynak klasörler değişmez; dışa aktarımlar `manuscript/<variant>/export/v3/*_v3.*`, önceki dışa aktarımlar V2 kaydı olarak
yerinde. Yeni model eğitimi veya yeni deney yok; yalnız kayıtlı çıktılar.

- PAPER-003 (kapanış kanıtı): `decision_calibration.py`'ye `separation()` eklendi. Bu fonksiyon tam veya yarı-tam ayrışmayı skor–etiket yapısından doğrudan, optimizasyondan bağımsız olarak saptar (Albert & Anderson 1984).
  - Durumlar: "not identifiable (constant score)", "not estimable (complete|quasi-complete separation)" ve "not converged (optimisation)"; sonuncusu yalnız ayrışma yokken çözücü başarısız olursa kullanılır.
  - Nokta tahminlerinde ve 2.000 yeniden örneklemenin hiçbirinde optimizasyon hatası çıkmadı. BUT QDB'de eğimi olmayan 1–16 yeniden örneklemenin hepsi saptanmış ayrışma.
  - CinC 2011'de LR ve MLP: bir kırpma sınırındaki 74 kaydın hepsi kullanılamaz, yani yarı-tam ayrışma. Profil olabilirliği |b| ile monoton artıyor (LR b → −∞, MLP b → +∞).
  - Sayısal değerler önceki çıktıyla birebir aynı; yalnız durum etiketleri değişti. Yedek: `decision_calibration_pre_V3_2026-11-13.csv`.
  - Testler: kaba kuvvet tanımıyla 20.000 rastgele örnekte uyum; scikit-learn'ün cezasız fit'iyle normal, aşırı ve kırpılmış skorlarda karşılaştırma; yarı-tam ayrışma yapısı. Toplam 44 test.
  - S32'de, bazı yeniden örneklemelerde tahmin yoksa tahminli örnekleme sayısı gösteriliyor.
- PAPER-007: "mattered more than the model class" kaldırıldı; yapılandırma bağımlılığı bildiriliyor.
- Sıralama cümlesi: betimsel sıra ile Holm-destekli üstünlük ayrıldı.
- S44 yorumu:
  - CinC 2011 aralığı sıfırı dışlıyor ama geniş.
  - "0.0%" yerine "2.000 yeniden örneklemenin hiçbirinde (gözlenmedi; olasılık sıfır demek değil)".
- PAPER-011: TRIPOD v2 madde 2 yalnız S28; iki listede madde 14 notu ana metinle uyumlu (demografi var, 7 denek yetersiz, dış veriler demografik olarak analiz edilmedi).
- PAPER-008 (yazar kararı: yazılım atfı):
  - HuBERT-ECG medRxiv ve TabPFN-3 arXiv kaynakça girdileri kaldırıldı.
  - Yerlerine kullanılan sürümler kondu: HF Edoardo-Coppola/hubert-ecg-base rev. 8ffa1ad…; HF Prior-Labs/tabpfn_3 rev. 24a16a8…, checkpoint tabpfn-v3-classifier-v3_default.ckpt, tabpfn 9.0.0. Revizyonlar yerel indirme metaverisiyle eşleşti.
  - Sapma listesindeki ön baskı cümlesi güncellendi.
- Kaynak doğrulaması: `manuscript/srep/reference_verification_V3_2026-11-13.md` ve Crossref CSV'si (`evaluation/crossref_reference_check.py`).
  - Düzeltilen bulgular:
    - Smital 2026'ya cilt 13 ve makale 1567 eklendi.
    - ST-MEM bağlantısı ICLR bildiri sayfasına çevrildi.
    - Pan–Tompkins: kod sadeleştirilmiş bir yeniden uygulama (tek global eşik). Yöntem metni artık bunu söylüyor.
    - SiamQuality ifadesi inceltildi.
  - Albert & Anderson 1984 eklendi; srep kaynak sayısı 52.
- Sapma maddesine ayrışma teşhisi eklendi.
- srep ana metin yaklaşık 57 kelime uzadı (önceki sayımla ≈4.088); özet 199 kelime, değişmedi.
- Doğrulama: v2 155 sayı + 31 ifade, srep 147 + 30, 0 hata; srep'te 291 sayının hepsi kaynaklı; 44 test ve çapraz kontrol geçti. S32 PDF'te gözle incelendi.
- İnceleme paketi: `manuscript/srep/export/v3/review_package_v3/` (kod, CSV'ler, test/doğrulama kaydı, kaynak raporu).
- Commit edilmedi.

### 2026-11-14 — V4: üçüncü hakemin V3 yanıtı ("V3.docx"; karar: MAJOR REVISION, sayısal kapanış doğrulaması bekleniyor)

Yeni model eğitimi veya deney yok; tüm kontroller kayıtlı tahminlerden. V3 dışa aktarımları ve raporları V3 kaydı olarak duruyor.

- **PAPER-003 (ayrışma kanıtı):** yeni salt-okunur betik `evaluation/calibration_separation_diagnostic.py`.
  - Çıktılar, `results/v2_2026-11/` altında:
    - `calibration_separation_levels.csv`: düzey başına sınıf sayıları;
    - `calibration_separation_diagnostic.csv`: kesim, yön, eşitsizlikler, kesimdeki iki sınıf, kontrol sırası; tahmin edilen 41 eğim için scikit-learn karşılaştırması ve skor normu;
    - `calibration_separation_bootstrap.csv`: 2.000 yeniden örneklemede durum sayıları ve bağımsız kaba kuvvet teyidi.
  - **V3 ifadesinin düzeltilmesi:** "bir sınırdaki 74 kaydın hepsi kullanılamaz" ifadesi eksikti.
    - LR'de kesim üst sınırda: altındaki 76 kaydın hepsi kullanılamaz, üstünde kayıt yok, sınırda 149 kullanılamaz ve 773 kullanılabilir kayıt var (b → −∞).
    - MLP'de kesim alt sınırda: üstündeki 75 kayıt kullanılamaz, sınırda 150 kullanılamaz ve 773 kullanılabilir kayıt var (b → +∞).
    - Ana metin, S32 açıklaması, verify_v2_prose kontrolü ve test buna göre güncellendi.
  - Sayısal sonuçlar:
    - 41 tahmin scikit-learn ile en çok 4,3×10⁻¹⁰ farkla uyumlu (tolerans 10⁻⁴·max(1,|b|)); skor normu ≤ 1,0×10⁻¹².
    - Eğimi olmayan her yeniden örnekleme ayrışma; kaba kuvvetle teyit edildi; optimizasyon hatası 0.
  - S32: tanım açıkça yazıldı; sabit-skor kontrolünün önce geldiği belirtildi; aralıkların tahminli yeniden örneklemelere koşullu olduğu ve dışlananları temsil etmediği eklendi.
- **Ön baskı cümlesi:** "have no peer-reviewed version" yerine "searches on 13 November 2026 found no peer-reviewed version".
- **Ek S-M1:** Pan–Tompkins sadeleştirmesi burada da yazıldı.
- **Model sürümleri:** yerel ağırlıkların SHA-256'sı, HF API'nin belirtilen revizyondaki LFS oid'si ile eşleşti (`results/v2_2026-11/model_provenance_V4_2026-11-14.csv`).
- **İddia desteği:** öncelikli kaynak grupları kontrol edildi (`manuscript/srep/reference_verification_V4_2026-11-14.md`). Düzeltmeler:
  - "Three experts annotated" artık Smital 2026'ya atfediliyor (Smital 2020 bir yöntem makalesi);
  - HeartBEiT için "frozen … linear probe" ifadesi genelleştirildi;
  - Moeyersons için "often recommended" yerine "improved results in an earlier cross-dataset study";
  - Guo/Subbaswamy için "ranking" yerine "discrimination".
- **Dışa aktarım:** komutlar `evaluation/export_manuscript.sh` betiğine kaydedildi. Betik var olan bir tur klasörünün üzerine yazmıyor ve `MANIFEST_vN.txt` dosyasını üretiyor (git HEAD ve kirli durum, araç sürümleri, kaynak ve çıktıların SHA-256'sı).
- Commit edilmedi; commit kimliği ve arşiv DOI'si yazar onayı bekliyor.
- 2026-11-14 ek: commit `db7224c062237e46234970994b2133dfa4516661` oluşturuldu (yazar onayı; push yok).
  - Hakem notu: "kaynak hash'leri PDF'lerin bu commit'ten üretildiğini teyit eder" ifadem fazla güçlüydü.
  - `MANIFEST_v4.txt` değiştirilmedi. Yanına `evaluation/bind_export_to_commit.py` ile `BINDING_v4_db7224c.txt` yazıldı. İçeriği:
    - kaynaklar ve figürler commit ile eşleşiyor;
    - çıktı hash'leri değişmemiş;
    - dışa aktarım komutu ve araç sürümleri;
    - commit dışı girdiler: pandoc, Chrome, sistem fontları.
  - Figürler dışa aktarım anında hash'lenmemişti. `export_manuscript.sh` artık bundan sonraki turlar için onları da hash'liyor.

### 2026-11-14 — V5: analiz sürümü metne yazıldı (yazar kararı)

Hakem önerisi: commit kimliği arşiv DOI'sine bağlanmasın.
- Veri erişilebilirliği cümlesi artık şöyle (v2 ve srep): "correspond to commit db7224c062237e46234970994b2133dfa4516661 on the branch `v2-revision-2026-11` (analysis version); an archived release is [ARCHIVE DOI, TO BE ADDED AT SUBMISSION]".
- TRIPOD 18f'ye analiz sürümü eklendi.
- verify_v2_prose:
  - eski yer tutucu yasak, yeni cümle zorunlu;
  - commit kimliği sayı kaynağı kontrolünden çıkarıldı (rakam grupları yalnız v2 metnindeki aynı kimlikle "eşleşiyordu").
- `export_manuscript.sh`: figür içermeyen dosyalarda `grep` boş dönünce `pipefail` betiği durduruyordu; düzeltildi. Yarım kalan ilk deneme yalnız bir manifest üretmişti, silinip yeniden çalıştırıldı.
- Dışa aktarım `export/v5/` altında, `MANIFEST_v5.txt` figür hash'lerini de içeriyor. İnceleme paketi: `manuscript/srep/export/v5/review_package_v5.zip`.
- Doğrulama: 45 test; v2 155+41, srep 147+34 kontrol, 0 hata; 291 sayı kaynaklı; çapraz kontrol geçti. Ana metin s. 20 ve TRIPOD s. 5 gözle incelendi.
- Commit edilmedi. V5 metin değişikliği commit edildikten sonra V5 için de eşleştirme kaydı yazılacak.

### 2026-11-14 — V6: hakemin metin paketi incelemesi ("Metin paketi yeterli.docx")

Hakem 25 dosyayı hash'leriyle doğruladı ve PAPER-003'ü VERIFIED FIXED yaptı. Üç yeni bulgu: PAPER-013, 014, 015. Yeni model eğitimi yok.

- **PAPER-013 (eşit AUROC'ta lider):** `rank_uncertainty.py` güncellendi.
  - Birincil kural artık yazılı: eşitlikte sabit model sırasındaki ilk model (np.argmax).
  - Yeni satırlar: `ties_selection`, `ties_target`, `same_leader_any`, `same_leader_all`, `regret_best_case`, `regret_worst_case`.
  - Sonuç: tam veride eşitlik yok. 2.000 yeniden örneklemede BUT QDB'de 7, CinC 2011'de 0, CinC 2017'de 1 eşitlik.
  - Aynı lider, en az–en çok elverişli eşitlik bozma: CinC 2011 için 11–13, CinC 2017 için 0–1 (2.000'de). AUROC kaybı aralık sınırları en çok 0,002 değişiyor.
  - Birincil 18 satır önceki çıktıyla birebir aynı; yedek `rank_uncertainty_pre_V6_2026-11-14.csv`.
  - Ana metin cümlesi ve S44 (iki yeni sütun, açıklama) güncellendi; yeni test `test_leader_ties` (toplam 46).
- **PAPER-014:** S32 ve teşhis betiği açıklaması: log-olabilirlik sonlu bir supremuma yaklaşır, sonlu MLE eğimi yoktur.
- **PAPER-015:** ECGFounder düzeltmesi bulundu (doi:10.1056/AIx2500913, PubMed 42932615). Crossref'te bağlantı olmadığı için V3'te kaçmıştı.
  - İçeriği OKUNAMADI (yayıncı sayfası bot kontrolü). Atıf yalnız modelin varlığına dayandığı için etki beklenmiyor; yazar gönderimden önce bildirimi açmalı.
  - Tüm DOI'ler ikinci kaynak olarak Europe PMC'de tarandı: 4 düzeltme, geri çekme yok (`reference_europepmc_notices_V6_2026-11-14.csv`).
  - V3 raporuna tarihli düzeltme notu eklendi; yeni rapor `reference_verification_V6_2026-11-14.md`.
- **Yeniden hesaplama girdileri:** `export/v6/` altında `review_package_v6_part1.txt`, `review_package_v6_part2_predictions.txt` (üç tahmin tablosu) ve `local_recalibration_draws.csv`. Paketleme betiği: `evaluation/make_text_bundle.py`.
- Doğrulama: 46 test; v2 155+44, srep 147+35 kontrol, 0 hata; 291 sayı kaynaklı; çapraz kontrol geçti. Ana metin s. 4 ve ek s. 43 gözle incelendi.
- Commit edilmedi (V5 ve V6 değişiklikleri birlikte bekliyor).

### V7 (2026-11-14): hakemin V6 yanıtı — sürüm beyanı
- Hakem V6 girdileriyle sıralama ve sıfır-ret sonuçlarını bağımsız yeniden hesapladı. PAPER-001, 002, 006, 013, 014:
  VERIFIED FIXED. Açık kalanlar: PAPER-008 (ağırlık kökeni için bağımsız doğrulama sınırı), PAPER-009 (sürüm beyanı),
  PAPER-012 (mektup yer tutucuları), PAPER-015 (ECGFounder düzeltmesinin içeriği okunmadı; yazar okuyacak).
- PAPER-009: V6 metni bütün analizleri `db7224c` ile eşliyordu; V6 eşitlik analizi o commit'te yok. Hakemin önerdiği sıra
  uygulandı (yazar onayı, push yok):
  1. Analiz-dondurma commit'i `a21e70b89d2448565624f15be6d6ec98caebdaca`: analiz kodu, testler, sonuçlar, dışa aktarım betikleri.
  2. Ana metinler ve TRIPOD 18f artık analiz sürümü olarak `a21e70b…` gösteriyor; `verify_v2_prose.py` bunu zorunlu kılıyor.
  3. Bu metin değişiklikleri ayrı bir gönderim-kaynak commit'inde; V7 bu commit'ten dışa aktarıldı ve
     `BINDING_v7_<commit>.txt` ile bağlandı.
- Sunum: ana metinde "0.5% (11 of 2,000)" yerine "11 of 2,000 resamples (0.55%)".
- Yeni analiz ya da model eğitimi yok. `db7224c` V3/V4 durumunun kaydı olarak kalıyor.
- V7 kayıt düzeltmesi (2026-11-14, hakemin V7 yanıtı; yeni sürüm ya da dışa aktarım yok):
  - Gönderim-kaynak commit'i `34d9b9ed38db7493f4ed9dd30de53a33552668cd`; V7 bu commit'ten dışa aktarıldı.
  - Hakem dört V7 PDF'inin manifestle eşleştiğini doğruladı. İki kayıt hatası: `MANIFEST_v7.txt` izlenen dosyalarda
    değişiklik yokken "DIRTY" diyor (betik izlenmeyen `export/` klasörünü sayıyordu); `BINDING_v7_34d9b9e.txt` şekiller
    için V4'ten kalma "hash'lenmedi" ifadesini taşıyor (V7 manifesti dört şeklin hash'ini içerir).
  - İki kayıt değiştirilmedi. Yanlarına `CORRECTION_v7_2026-11-14.txt` yazıldı (iki varyant). Dışa aktarım anında
    `git status` çıktısı saklanmadığı için DIRTY nedeni çıkarım olarak, sonradan alınan çıktı da öyle etiketlenerek verildi.
  - Betikler düzeltildi: `export_manuscript.sh` izlenen değişiklikleri ve izlenmeyen dosyaları ayrı raporlar ve git
    çıktısını manifeste yazar; `bind_export_to_commit.py` manifestteki şekil hash'lerini commit ile karşılaştırır ve var
    olan bir kaydın üzerine yazmaz. Düzeltilmiş betikler yeni bir dışa aktarımda henüz kullanılmadı.

## 4. Devir notu (yeni oturum için, 2026-11-07)
Durum: v2 ana kuyruğu tamamlandı (`results/v2_2026-11/`, günlükler `logs/`). Kod değişiklikleri commit EDİLMEDİ
(36 izlenen dosya değişti + yeni dosyalar; `git status` ile görülebilir). v1 = commit 55095ed, `results/`.
Çalıştırma: `ECG_RESULTS_DIR=results/v2_2026-11 .venv/bin/python <betik>`; TabPFN ortamı `/Volumes/KIOXIA/AI/tabpfn_env/bin/python` (tabpfn 9.0.0).
Kuyruk devamı: `START_AT=<aşama> bash experiments/v2_queue.sh`. Testler: `.venv/bin/python -m unittest discover -s tests` (24, hepsi geçiyor).
Önbellek: öznitelikler `experiments/feature_cache.py` ile segment_id'ye göre okunur; mevcut setlerin üzerine yazılmaz.
Dikkat: `external_window_predictions.csv` içinde CinC 2011 derivasyon II ve 12-derivasyon setleri aynı segment_id'leri
paylaşır → her zaman `dataset` ile filtrele.

Sıradaki işler (öncelik sırasıyla):
1. Kalan analizleri v2'ye taşı: nested_thresholds, local_recalibration, butqdb_official_split (F1b, TabPFN dahil),
   exp4_revised_nstdb, seed_repeats_transfer / seed_stability, butqdb_class2_sensitivity, prevalence_subjects_features,
   cinc2011_constant_lead_check. Her birinde CinC 2011 kaynağının 998 kayda geçtiğini ve sabit `results/` yolu
   kalmadığını kontrol et (bazıları hâlâ cinc_random_draws dosyalarını okuyor olabilir).
2. 12 derivasyonlu analizi 998 kayda genişlet (cinc2011_twelve_lead_extraction + d1 12-derivasyon skorlama ~3 saat).
3. Gecikme: tek betikte birleştir, tek başına çalıştır; PIL raster ile 2D CNN'i yeniden eğitip AUROC'u karşılaştır.
4. Tablo/şekil üreticilerini (build_manuscript_tables, generate_q1_figures, assemble_q1_manuscript) v2'ye bağla.
5. Metin: İP-1 yeniden konumlandırma, İP-2 kaynaklar (`manuscript/references_v2_candidates_2026-11-06.md`; her kaynak
   için iddia desteği ayrıca kontrol edilecek), kronolojik dili temizle ama sapmaları dürüstçe beyan et, kısa ve doğru
   AI beyanı, TRIPOD+AI güncellemesi. Hedef dergi henüz seçilmedi (yazar kararı).
