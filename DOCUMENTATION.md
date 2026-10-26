# RAG Playground — Teknik Dokümantasyon

Bu doküman, `src/rag_playground` altında yer alan tüm paketlerin, sınıfların, veri modellerinin ve algoritmaların ne işe yaradığını ve birbirleriyle nasıl çalıştığını açıklar.

---

## 1. Genel Mimari ve Dizin Yapısı

```text
src/rag_playground/
├── core/         # Ayarlar (Settings), güvenlik/PII kuralları, loglama ve özel hata sınıfları
├── llm/          # Azure OpenAI ve OpenAI istemcileri ile LLMFactory soyutlaması
├── ingestion/    # Doküman okuma/yazma, PDF->Markdown çıkarma (Docling + OCR) ve Semantic Chunking
├── retrieval/    # Embedding servisi, BM25, Qdrant VectorDB, RRF Hybrid Search ve LLM Reranker
├── memory/       # Kısa süreli (Short-Term) ve uzun süreli (Long-Term) konuşma belleği yönetimi
├── assistant/    # Guardrails (Input/Output + LLM-as-a-Judge), Intent sınıflandırma ve Asistan orkestrasyonu
├── evaluation/   # Retrieval ve Generation (LLM-as-a-Judge) değerlendirme servisleri
├── models/       # Veri taşıyıcı dataclass modelleri (Citation, VectorRecord, SearchHit, Memory, FileType)
├── schemas/      # Pydantic doğrulama şemaları ve Structured Output (JSON Schema) tanımları
└── prompts/      # Sistem promptları (.md dosyaları) ve yükleyici modül
```

---

## 2. Core Katmanı (`rag_playground.core`)

Sistemin yapılandırma, güvenlik, loglama ve hata yönetimi altyapısını sağlar.

- **`settings.py` (`Settings`):**
  - `pydantic-settings` tabanlı merkezi yapılandırma sınıfıdır. `.env` dosyasından LLM deployment adlarını, Qdrant bağlantı bilgilerini, chunk boyutlarını (`MIN_CHUNK_CHARS=260`, `MAX_CHUNK_CHARS=900`, `SENTENCE_OVERLAP_RATIO=0.2`), arama limitlerini (`SEARCH_CANDIDATES=25`, `FUSED_SEARCH_CANDIDATES=10`, `TOP_K_CANDIDATES=5`), bellek eşiklerini ve prompt dosya yollarını okur.
- **`security.py`:**
  - **PII Tespiti ve Algoritmik Doğrulama:** TC Kimlik No (`is_tckn_valid`), TR IBAN (`is_iban_valid` - Mod 97 kontrolü), Kredi Kartı (`is_luhn_valid` - Luhn algoritması), e-posta, telefon numarası, plaka, IP adresi, doğum tarihi, isim ve ev adresi için Regex desenleri ve doğrulama fonksiyonları içerir.
  - **Prompt Injection Tespiti (`_PROMPT_INJECTION_RE`):** Türkçe, İngilizce ve diğer dillerde sistem talimatlarını yok sayma, gizli promptu ifşa etme, *developer mode / jailbreak* ve rol değiştirme saldırılarını yakalayan kapsamlı Regex kuralları barındırır.
- **`logger.py` (`get_logger`):**
  - `rag` isim alanı altında tarih, seviye, modül adı ve mesaj formatında standart konsol loglaması sağlar.
- **`exceptions.py`:**
  - Tüm özel hataların taban sınıfı olan `RAGError` ve alt sınıflarını (`LLMClientError`, `EmbeddingError`, `EmbeddingClientError`, `IntentClassificationError`, `MemoryServiceError`, `RetrievalServiceError`, `DocumentExtractionError`, `DocumentChunkingError`, `DocumentSavingError`, `InvalidGroundTruthSchemaError`, `LLMGenerationError`) tanımlar.

---

## 3. LLM Katmanı (`rag_playground.llm`)

Farklı dil modeli sağlayıcılarını ortak bir arayüzde soyutlar.

- **`llm_factory.py` (`LLMClient`, `LLMFactory`):**
  - `LLMClient`: `get_client()`, `generate_response()` ve `generate_embeddings()` soyut metotlarını tanımlayan temel sınıftır.
  - `LLMFactory`: `settings.LLM_PROVIDER` değerine göre (`azure`, `azure_openai`, `openai`, `openai_compatible`) ilgili istemci sınıfını dinamik olarak oluşturur.
- **`azure_client.py` (`AzureOpenAIClient`):**
  - `AsyncAzureOpenAI` kullanarak asenkron sohbet yanıtı (`chat.completions.create`) ve vektör (`embeddings.create`) üretir.
- **`openai_client.py` (`OpenAIClient`):**
  - Standart OpenAI veya OpenAI uyumlu (`base_url` destekli) API'ler üzerinden asenkron yanıt ve embedding üretir.

---

## 4. Ingestion Pipeline (`rag_playground.ingestion`)

Ham PDF dokümanlarını aranabilir ve zengin metaverili parçalara (`Citation`) dönüştürür.

- **`document_io.py` (`DocumentIO`):**
  - `save_document()`: Ham veya işlenmiş dokümanları `aiofiles` ile asenkron olarak diske kaydeder.
  - `read_file()`: Dosyayı okur ve mükerrer işlemeyi önlemek için `SHA-256` özetini (hash) hesaplar.
- **`ocr_factory.py` (`OCRFactory`, `TesseractOCRFactory`):**
  - Taranmış PDF'ler için Türkçe (`tur`) ve İngilizce (`eng`) dil destekli Tesseract OCR yapılandırmasını üretir.
- **`extractor.py` (`DocumentExtractor`, `fix_accents`):**
  - `_is_digital()`: `pdfplumber` ile ilk sayfaların metin yoğunluğunu ölçerek PDF'in dijital mi yoksa taranmış mı olduğunu otomatik tespit eder.
  - `extract()`: **Docling** (`DocumentConverter`) kullanarak PDF'i sayfa sonu işaretleyicileriyle (`---[PAGE BREAK]---`) Markdown formatına çevirir. `fix_accents()` ile PDF'lerden kaynaklanan bozuk Türkçe aksan/karakter birleşimlerini (`NFKC`) düzeltir.
- **`chunker.py` (`DocumentChunker`, `clean_markdown`):**
  - **Markdown Ayrıştırma (`_parse_md_blocks`):** Metni başlık (`heading`), tablo (`table`) ve paragraf (`text`) bloklarına ayırır. Başlıkları ekmek kırıntısı (`breadcrumb`, örn. `Bölüm 1 > Giriş`) olarak takip eder.
  - **Tablo Koruma (`_split_table`):** Uzun tabloları bölerken tablo başlık satırlarını her parçanın başına tekrar ekler.
  - **Semantik Parçalama (`_semantic_split`):** Cümlelerin embedding vektörlerini alır, ardışık cümleler arası kosinüs mesafesini ($1 - \text{CosineSimilarity}$) hesaplar. Mesafenin yüzdelik eşiği (`SEMANTIC_CHUNKING_BREAK_PERCENTAGE=0.85`) ve minimum mesafeyi (`0.20`) aştığı noktalardan metni böler. Küçük parçaları birleştirir (`_merge_small_groups`) ve komşu parçalar arasına cümle örtüşmesi (`_add_overlap`) ekler.
  - Çıktı olarak sayfa, paragraf, başlık yolu ve doküman kimliği içeren `Citation` nesneleri üretir.
- **`document_service.py` (`DocumentService`):**
  - `DocumentIO`, `DocumentExtractor` ve `DocumentChunker` bileşenlerini birleştirerek verilen PDF dosya listesini uçtan uca işler ve `SHA-256` kontrolüyle aynı dosyanın tekrar işlenmesini engeller.

---

## 5. Retrieval Pipeline (`rag_playground.retrieval`)

Kullanıcı sorgusuyla en alakalı doküman parçalarını hibrit arama ve yeniden sıralama ile bulur.

- **`embedding_service.py` (`EmbeddingService`, `AzureEmbeddingService`):**
  - Metin listelerini `EMBED_BATCH_SIZE` (`96`) büyüklüğünde gruplara bölerek asenkron vektör temsillerine dönüştürür.
- **`bm25.py` (`BM25`):**
  - Türkçe karakter normalizasyonlu (`İ, I, ı -> i`) bellek içi (in-memory) **Okapi BM25** ($k_1=1.5, b=0.75$) indeksleme ve anahtar kelime arama motorudur.
- **`similarity.py` (`cosine_similarity`):**
  - İki vektör arasındaki kosinüs benzerliğini hesaplar.
- **`rrf.py` (`reciprocal_rank_fusion`):**
  - Sparse (BM25) ve Dense (Vektör) arama sonuç listelerini $\text{RRF}(d) = \sum \frac{1}{k + \text{rank}(d)}$ ($k=60$) formülüyle tek bir sıralamada birleştirir.
- **`vector_db.py` (`VectorDB`, `AsyncQdrantVectorDB`):**
  - `store()`: Kayıtları deterministik ID'lerle Qdrant'a yazar ve eşzamanlı olarak bellek içi BM25 indeksini günceller.
  - `dense_search()`: Sadece embedding benzerliğiyle Qdrant üzerinde arama yapar (filtre destekli).
  - `search()`: **Hybrid Search** gerçekleştirir; Qdrant'tan gelen `dense_search` sonuçları ile `BM25` sonuçlarını `reciprocal_rank_fusion` ile birleştirerek en iyi adayları döndürür. Koleksiyon ilk kez sorgulanıyorsa `rebuild_bm25()` ile Qdrant'taki mevcut kayıtları BM25'e yükler.
- **`retrieval_service.py` (`RetrievalService`):**
  - `index_citations()`: `Citation` nesnelerini `VectorRecord`'a dönüştürüp vektör veritabanına kaydeder.
  - `retrieve()` & `_llm_reranker()`: Hibrit aramadan gelen aday parçaları (`FUSED_SEARCH_CANDIDATES=10`) alır ve **LLM-as-a-Reranker** ile her parçaya 0–10 arası alaka puanı (`rerank_score`) verir. `MIN_RERANK_SCORE` (`3`) altında kalanları eler ve en yüksek puanlı `top_k` (`5`) parçayı döndürür.

---

## 6. Memory Katmanı (`rag_playground.memory`)

Asistanın hem mevcut sohbet bağlamını hem de kullanıcıya dair kalıcı bilgileri hatırlamasını sağlar.

- **`memory_service.py` (`MemoryIService`, `MemoryService`):**
  - **Short-Term Memory (`update_short_term_memory`, `get_short_term_memories`):** Mevcut konuşma geçmişini tutar; mesaj sayısı (`HISTORY_MAX_MESSAGES=6` çift) ve kelime/token sınırına (`HISTORY_MAX_TOKENS=2000`) göre eski mesajları otomatik kırpar (`_trim_history`).
  - **Memory Extraction (`extract_from_memories`):** Konuşma geçmişini LLM ile analiz ederek kullanıcıya ait olgu (`fact`), tercih (`preference`), hedef (`goal`), olay (`event`) veya çıkarımları (`inference`) güven skoruyla (`confidence`) birlikte yapılandırılmış olarak çıkarır.
  - **Consolidation & Long-Term Storage (`schedule_consolidation`, `store_memories`):** Biriken mesaj sayısı `LONG_TERM_MEMORY_BATCH_SIZE` (`10`) değerine ulaştığında (veya geçmiş sıfırlandığında) arka planda asenkron görev başlatarak güven eşiği (`0.7`) üzerindeki anıları Qdrant `Memories` koleksiyonuna kaydeder.
  - **Memory Retrieval (`get_long_term_memories`):** Yeni sorgu geldiğinde kullanıcının `user_id` filtresiyle Qdrant üzerinde `dense_search` yapar ve benzerlik skoru `LONG_TERM_MEMORY_MIN_SCORE` (`0.5`) üzerinde olan kalıcı anıları getirir.

---

## 7. Assistant & Guardrails Katmanı (`rag_playground.assistant`)

Güvenlik kontrollerini, niyet sınıflandırmayı ve kaynaklı yanıt üretimini koordine eder.

- **`guardrails.py` (`GuardrailsService`):**
  - **`input_guardrail()`:** Görünmez karakterleri temizler, Unicode normalizasyonu yapar, **Prompt Injection** tespit edilirse isteği anında engeller. Seçilen moda (`strict`, `lenient`, `loose`) göre PII verilerini (`[MASKED_TCKN]`, `[MASKED_EMAIL]`, `[MASKED_IBAN]` vb.) ve kuruma özel kelimeleri (`proprietary_dictionary`) maskeler.
  - **`output_guardrail()`:** LLM'in ürettiği yanıtta yer alabilecek PII ve özel terimleri maskeler.
  - **`llm_as_judge()`:** Üretilen yanıtı kullanıcıya dönmeden önce soru ve kaynaklarla birlikte hakem LLM'e denetletir (`faithfulness`, `relevancy`, `hallucination`, `nfsw`, `contain_sensitive_info`). Hassas bilgi veya NSFW varsa yanıtı bloke eder; halüsinasyon veya ilgisizlik varsa düzeltme için `feedback` döndürür.
- **`intent.py` (`IntentClassifier`):**
  - Kullanıcı mesajını (ve varsa kısa süreli konuşma geçmişini) inceleyerek niyeti 4 sınıftan birine (`knowledge_question`, `chitchat`, `needs_clarification`, `out_of_scope`) atar, `confidence` skoru üretir ve zamirleri/bağlamı çözümlenmiş bağımsız arama sorgusunu (`resolved_query`) oluşturur.
- **`assistant.py` (`AssistantService`):**
  - Uçtan uca sohbet akışını yöneten ana sınıftır (`answer()`):
    1. Girdi güvenlik kontrolünü (`_input_guardrails_check`) ve niyet sınıflandırmasını (`get_intent`) çalıştırır.
    2. Niyet `knowledge_question` ise `resolved_query` ile `retriever` üzerinden ilgili kaynakları (`citations`) getirir ve numaralandırılmış bağlam (`[1] (doküman, s.X)`) oluşturur.
    3. Uzun süreli ve kısa süreli bellek kayıtlarını sistem mesajlarına ekler.
    4. `_generate_checked_answer()` ile yanıtı üretir, `output_guardrail` ve `llm_as_judge` denetiminden geçirir. Hakem LLM `feedback` döndürürse yanıtı geri bildirimle birlikte en fazla `max_retries` (`3`) kez yeniden üretir (Self-Correction).
    5. `show_citations()` ile yanıtta kullanılan kaynakları sayfa, paragraf ve skor bilgileriyle formatlar.

---

## 8. Evaluation Katmanı (`rag_playground.evaluation`)

RAG sisteminin arama ve yanıt üretme kalitesini `ground_truth.json` veri seti üzerinden ölçer.

- **`evaluation_service.py` (`EvaluationService`):**
  - Ground truth JSON dosyasını okur, Pydantic şemasıyla (`GroundTruth`) doğrular ve soruları normalize ederek (`strip().lower()`) değerlendiricilere sunar.
- **`retrieval_eval.py` (`RetrievalEvaluator`):**
  - Asenkron ve eşzamanlı (`asyncio.Semaphore`) olarak tüm sorularda retrieval çalıştırır ve şu metrikleri hesaplar:
    - **Effectiveness Metrikleri:** `context_precision`, `context_recall`, `hit_rate`, `f1`. (Chunk ID beklentisi yoksa otomatik olarak doküman ID bazlı hesaplamaya geçer).
    - **Ranking Metrikleri:** `mrr` (Mean Reciprocal Rank) ve `binary_ndcg` (Normalized Discounted Cumulative Gain).
- **`generation_eval.py` (`GenerationEvaluator`):**
  - **LLM-as-a-Judge** ve **Embedding benzerliği** kullanarak üretilen yanıtları 6 metrikte değerlendirir:
    - **`faithfulness()`:** Cevaptaki atomik iddiaların (`claims`) getirilen kaynaklarca desteklenme (`TP`) oranı.
    - **`answer_relevancy()`:** Cevaptan tersine türetilen 3 sorunun asıl soruyla olan kosinüs benzerliği ortalaması.
    - **`answer_quality()`:** Üretilen cevap ile beklenen referans cevabı karşılaştırıp `TP`, `FP`, `FN` iddialarını çıkarır; buradan **Completeness** (`Recall`), **Answer Accuracy** (`Factual F1`) ve **Answer Correctness** ($0.75 \cdot F_{1,\text{factual}} + 0.25 \cdot \text{CosineSim}$) skorlarını hesaplar.
    - **`noise_sensitivity()`:** Bağlama 20 adet alakasız gürültü parçası (`NOISE_CHUNKS`) eklendiğinde ortalama cevap kalitesinde yaşanan düşüş oranını ölçer.

---

## 9. Veri Modelleri, Şemalar ve Promptlar

- **`rag_playground.models`:**
  - `Citation`: Chunk metni, doküman adı/ID, sayfa, paragraf, cümle sayısı, `breadcrumb`, `rrf_score` ve `rerank_score` alanlarını tutan veri sınıfı.
  - `VectorRecord`: Vektör veritabanına yazılacak `text`, `payload` ve deterministik `key` yapısı.
  - `SearchHit`: Qdrant/Hybrid arama sonucu dönen `id`, `score` ve `payload` nesnesi.
  - `Memory`: Uzun süreli bellek kaydı modeli.
  - `FileType`: Desteklenen dosya uzantıları (`PDF`, `MD`, `DOCX`, `TXT`, `HTML`, `PPTX`, `XLSX`) Enum sınıfı.
- **`rag_playground.schemas`:**
  - OpenAI/Azure *Structured Outputs* (`response_format={"type": "json_schema", ...}`) ile %100 şema uyumlu JSON çıktısı almak için kullanılan Pydantic modelleri:
    - `intent_schema.py` (`Intent`, `LLM_INTENT_FORMAT`)
    - `rerank_schema.py` (`RerankResponse`, `LLM_RERANK_RESPONSE_FORMAT`)
    - `memory_schema.py` (`MemoryExtraction`, `MemoryResponse`, `LLM_MEMORY_RESPONSE_FORMAT`)
    - `guardrails_schema.py` (`GuardrailsResponseSchema`, `LLM_GUARDRAILS_FORMAT`)
    - `generation_eval_schemas.py` (`FaithfulnessEvalResponse`, `AnswerRelevancyEvalResponse`, `AnswerQualityEvalResponse`, `AnswerQualityScores`)
    - `ground_truth_schema.py` (`GroundTruth`, `ModifiedGroundTruth`)
    - `assistant_schema.py` (`AssistantResponse`)
- **`rag_playground.prompts`:**
  - `prompts.py` modülü, aynı klasördeki Markdown tabanlı sistem talimatlarını (`ASSISTANT_PROMPT.md`, `INTENT_PROMPT.md`, `RERANKER_PROMPT.md`, `MEMORY_PROMPT.md`, `GUARDRAILS_PROMPT.md`, `FAITHFULNESS_PROMPT.md`, `ANSWER_RELEVANCY_PROMPT.md`, `ANSWER_QUALITY_PROMPT.md`) yükleyerek ilgili servislere sunar.

