# Physiological Measurement — v1 Gönderim Kontrol Listesi

**Makale:** "In-domain rankings did not identify the best-transferring model in ECG usability screening: a patient-independent, multi-dataset benchmark of task-specific, tabular and foundation models" (Research paper).

**Nasıl üretildi:**
- Paket `evaluation/build_pmea_package.py` ile, `manuscript/main_q1.md` ve `supplement_q1.md` dosyalarından üretildi.
- Bu iki dosya da kayıtlı sonuçlardan `evaluation/assemble_q1_manuscript.py` ile oluşturuluyor.
- Tablolardaki hiçbir sayı elle girilmedi. PMEA sürümündeki 73 tablo satırı kaynakla birebir aynı (betikle kontrol edildi).

**Bu işlem dergiye fiilî başvuru değildir.** Dosyaları portala siz yüklemelisiniz.

## Dosyalar

| Portal türü | Dosya |
|---|---|
| Ana metin (Word) | `PMEA_Manuscript_v1.docx` (tablolar ve şekiller metin içinde, tablo başlıkları üstte) |
| Ek materyal | `PMEA_Supplementary_Material_1_v1.docx` (Ek Yöntemler S-M1–S-M4, Tablo S1–S31) |
| Raporlama kontrol listesi (ek dosya) | `PMEA_TRIPOD_AI_Checklist_v1.docx` |
| Kapak mektubu | `PMEA_Cover_Letter_v1.docx` |
| Başlık sayfası | `PMEA_Title_Page_v1.docx` |
| Yazar bilgileri ve beyanlar | `PMEA_Author_Information_and_Declarations_v1.docx` |
| Şekiller (vektör) | `PMEA_Main_Figures/Figure_1–3.pdf` |
| Önizleme (portal için değil) | `PDF_preview/*_preview.pdf`: HTML'den üretildi, sayfa düzeni Word'den farklıdır |
| Hash kaydı | `PMEA_v1_manifest.json` |

## PMEA biçimine dönüştürmede yapılanlar

- **Yazar bloğu:** başlığın altına, önceki PMEA-107866 başvurusuyla aynı biçimde eklendi.
- **Özet:**
  - PMEA başlıkları kullanıldı: Objective / Approach / Main results / Significance.
  - Düz paragraf halinde, 242 kelime.
  - Anahtar kelimeler ve kısaltmalar eklendi.
- **Atıflar:** numaralı atıflar IOP Harvard (yazar–yıl) biçimine çevrildi.
  - Kaynakça alfabetik sırada; 26 kaynağın hepsine atıf var.
  - Ek materyalde atıf yok; köşeli parantezli değer aralıkları değiştirilmedi.
- **Beyan bölümleri:** PMEA düzenine göre: Acknowledgements, AI disclosure, Funding, Ethics, Conflicts of interest, Author contributions, Data availability.
- **Sayfa düzeni:**
  - Dikey A4, 2.2 cm kenar boşluğu.
  - Tablolarda kenarlık, küçük yazı ve her sayfada tekrarlanan başlık satırı var. Tablo satırları bölünmüyor; başlıklar tablolarıyla aynı sayfada kalıyor.

## Göndermeden önce

**26 Ekim 2026'da tamamlananlar:**
- Beyanlar yazar tarafından onaylandı ve yer tutucular kaldırıldı: AI disclosure (üç rol), Funding, Ethics, Conflicts of interest, Author contributions.
- Arşiv DOI'si eklenmedi; Data availability bölümü herkese açık GitHub deposuna atıf yapıyor.
- GitHub deposu herkese açık (yazar onayı).

**Sizin yapmanız gerekenler:**
1. **İnceleme PDF'i:** ScholarOne için PDF'leri Word'den dışa aktarın; tablo ve şekil yerleşimini Word'de gözle kontrol edin.
2. **Kör değerlendirme:** PMEA'nın güncel yazar kılavuzunda tek-kör/çift-kör ayarına bakın (yazar bilgisi ana metinde).

## Açık erişim (TÜBİTAK–IOP "Oku & Yayımla", 2026–2028)

Önceki başvurudaki kayda göre (24 Ekim 2026'da kontrol edilmiş): Dokuz Eylül Üniversitesi IOP'nin Türkiye listesinde, PMEA ise anlaşmanın List A'sında.
- **Portalda:** kurum adı olarak açılan listeden "Dokuz Eylül Üniversitesi"ni seçin ve kurumsal e-postanızı kullanın.
- **Ücret sorulursa:** anlaşma kapsamında açık erişimi seçin; kart veya fatura bilgisi girmeyin.
- **Kabulden sonra:** SciPris e-postasını 5 gün içinde yanıtlayın (fonlamayı onaylayın, CC BY 4.0 lisansını imzalayın).
- **Kota:** 2026 kotasının dolmadığını kütüphaneden teyit edin; bunu ben doğrulayamıyorum.

## İlgili başvuru

- **PMEA-107866** (TabPFN damıtma) değerlendirmede.
- Kapak mektubunda ve beyan formunda açıkça belirtildi: ortak veri kümesi yok (o çalışma PTB-XL, Chapman-Shaoxing ve CPSC2018 kullanıyor). İki çalışma da TabPFN kullanıyor, farklı rollerde.

## IOP politikası

Hakem raporlarını üretken yapay zekâ araçlarına yüklemeyin; yanıtları siz yazın (önceki gönderim kaydındaki not).

## Bu pakette yapılmayanlar

- Word/PDF baskı provası (Word otomasyonu kullanılmadı).
- Karar düzeyindeki ölçütler (τ = 0.5) için güven aralıkları (TRIPOD 23a "Partly").
- TRIPOD+AI özet kontrol listesine göre satır satır özet denetimi (madde 2 "Partly").
