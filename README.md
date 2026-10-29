# RAG Playground

**RAG Playground**, kurumsal düzeyde uçtan uca bir **Retrieval-Augmented Generation (RAG)** mimarisini modüler olarak uygulayan, güvenli sorgu işleme, hibrit arama, yeniden sıralama (reranking), konuşma hafızası ve kapsamlı değerlendirme (evaluation) bileşenlerini barındıran bir Python kütüphanesidir.

Ek olarak, Notebook üzerinde her adımın nasıl çalıştığını ayrıntılı bir şekilde gösteren animasyonlar içerir.

---

## Kurulum

### 1. Gereksinimler
- **Python:** `3.12+`
- **Paket Yöneticisi:** [`uv`](https://docs.astral.sh/uv/) (önerilen) veya `pip`
- **Vektör Veritabanı:** **Qdrant** (varsayılan port: `6333`)
- **OCR (Opsiyonel):** Taranmış PDF dokümanları için `Tesseract OCR` (`eng`, `tur` dil paketleri)

### 2. Bağımlılıkların Yüklenmesi

Proje kök dizininde `uv` kullanarak sanal ortamı ve bağımlılıkları kurun:

```bash
uv sync
```

Veya `pip` ile geliştirici modunda yükleyin:

```bash
pip install -e .
```

### 3. Ortam Değişkenleri (`.env`)

Proje kök dizininde bir `.env` dosyası oluşturun ve kullandığınız LLM sağlayıcısına (`azure`, `openai` veya `gemini`) ve Qdrant bağlantı bilgilerinize göre aşağıdaki temel değişkenleri tanımlayın:

```env
LLM_PROVIDER=

OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_CHAT_TIMEOUT=60
OPENAI_CHAT_MAX_RETRIES=3
OPENAI_EMBEDDING_DEPLOYMENT=
OPENAI_CHAT_DEPLOYMENT=
OPENAI_RERANKER_DEPLOYMENT=
OPENAI_INTENT_DEPLOYMENT=
OPENAI_MEMORY_DEPLOYMENT=
OPENAI_GENERATION_EVAL_DEPLOYMENT=
OPENAI_GUARDRAILS_DEPLOYMENT=

AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_CHAT_TIMEOUT=60
AZURE_OPENAI_CHAT_MAX_RETRIES=3
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=
AZURE_OPENAI_CHAT_DEPLOYMENT=
AZURE_OPENAI_RERANKER_DEPLOYMENT=
AZURE_OPENAI_INTENT_DEPLOYMENT=
AZURE_OPENAI_MEMORY_DEPLOYMENT=
AZURE_OPENAI_GENERATION_EVAL_DEPLOYMENT=
AZURE_OPENAI_GUARDRAILS_DEPLOYMENT=

GEMINI_API_KEY=
GEMINI_CHAT_TIMEOUT=60
GEMINI_CHAT_MAX_RETRIES=3
GEMINI_EMBEDDING_DEPLOYMENT=
GEMINI_CHAT_DEPLOYMENT=
GEMINI_RERANKER_DEPLOYMENT=
GEMINI_INTENT_DEPLOYMENT=
GEMINI_MEMORY_DEPLOYMENT=
GEMINI_GENERATION_EVAL_DEPLOYMENT=
GEMINI_GUARDRAILS_DEPLOYMENT=

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=

VECTORDB_KNOWLEDGE_COLLECTION_NAME=Documents
VECTORDB_MEMORY_COLLECTION_NAME=Memories
VECTORDB_EVAL_COLLECTION_NAME=Evaluation
```

---

## Proje Neler İçeriyor?

`src/rag_playground` paketi altında uçtan uca bir RAG sisteminin tüm yapı taşları modüler olarak yer alır:

1. **Ingestion (`rag_playground.ingestion`):**
   - PDF dokümanlarının dijital veya taranmış (OCR) olarak tespit edilip **Docling** ile Markdown formatına dönüştürülmesi (`DocumentExtractor`).
   - Başlık hiyerarşisini (*breadcrumb*) ve tablo bütünlüğünü koruyan **Semantik Parçalama (Semantic Chunking)** ve zengin metadata üretimi (`DocumentChunker`).
2. **Retrieval (`rag_playground.retrieval`):**
   - Batch destekli, provider'dan bağımsız asenkron vektör üretimi (`LLMEmbeddingService`).
   - Kelime bazlı **BM25 (Sparse)** ve vektör bazlı **Qdrant (Dense)** aramalarını **Reciprocal Rank Fusion (RRF)** ile birleştiren **Hybrid Search** (`AsyncQdrantVectorDB`).
   - Aday parçaları soruyla ilgililiklerine göre 0–10 arasında puanlayıp filtreleyen **LLM-as-a-Reranker** (`RetrievalService`).
3. **Assistant & Güvenlik (`rag_playground.assistant` & `rag_playground.core.security`):**
   - Çok dilli **Prompt Injection** tespiti, algoritmik doğrulamalı (TCKN, IBAN, Kredi Kartı Luhn kontrolü, Telefon, E-posta vb.) **PII Maskeleme** ve **LLM-as-a-Judge** tabanlı çıktı denetimi / otomatik yanıt düzeltme (`GuardrailsService`).
   - Kullanıcı niyetini (`knowledge_question`, `chitchat`, `needs_clarification`, `out_of_scope`) ve zamir/bağlam çözümlemesini (`resolved_query`) çıkaran sınıflandırıcı (`IntentClassifier`).
   - Kaynak atıflı (`[1]`, `[2]`) yanıt üretimi yapan ana orkestrasyon servisi (`AssistantService`).
4. **Memory (`rag_playground.memory`):**
   - Token ve mesaj sınırına göre kırpılan **Kısa Süreli Bellek (Short-Term Memory)** ve konuşmalardan çıkarılan kalıcı kullanıcı bilgilerini (`fact`, `preference`, `goal` vb.) Qdrant'ta saklayan **Uzun Süreli Bellek (Long-Term Memory)** (`MemoryService`).
5. **Evaluation (`rag_playground.evaluation`):**
   - **Retrieval Değerlendirmesi:** `Precision`, `Recall`, `F1`, `Hit Rate`, `MRR` ve `Binary NDCG` (`RetrievalEvaluator`).
   - **Generation Değerlendirmesi:** LLM-as-a-Judge ve Embedding benzerliği ile `Faithfulness`, `Answer Relevancy`, `Answer Correctness`, `Completeness`, `Answer Accuracy` ve `Noise Sensitivity` (`GenerationEvaluator`).
6. **LLM İstemci Katmanı (`rag_playground.llm`):**
   - Azure OpenAI (`AzureOpenAIClient`) ve OpenAI uyumlu (`OpenAIClient`) sağlayıcıları tek bir arayüzde soyutlayan fabrika yapısı (`LLMFactory`).

---

## Detaylı Dokümantasyon

Mimari akış, tüm modüllerin, sınıfların, veri modellerinin ve yapılandırma ayarlarının detaylı açıklamaları için lütfen **[DOCUMENTATION.md](./DOCUMENTATION.md)** dosyasına bakın.

