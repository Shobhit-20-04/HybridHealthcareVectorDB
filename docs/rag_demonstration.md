# Clinical Retrieval-Augmented Generation (CLI-RAG) Demonstration & Evaluation Guide

**Project:** Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval  
**Database System:** PostgreSQL 18.6 on x86_64-windows with `pgvector` v0.8.6  
**Embeddings:** 384-dimensional dense vectors (`sentence-transformers/all-MiniLM-L6-v2`, Cosine Distance `<=>`)  
**Interface:** Headless Command-Line Interface (CLI only; strictly no web application, dashboard, or chatbot UI)  

> [!CAUTION]
> **ACADEMIC RESEARCH PROTOTYPE ONLY — NOT A CLINICAL DIAGNOSTIC SYSTEM**  
> This system is strictly an academic database research prototype designed for evaluating relational-plus-vector query engines, indexing tradeoffs, and citation groundedness. It is **NOT** a medical device, is **NOT** validated for clinical diagnosis, and must **NEVER** be used to direct patient care, diagnose illnesses, or make healthcare treatment decisions.

---

## 1. Architectural Overview & Privacy Safeguards

The clinical RAG demonstration demonstrates how the PostgreSQL database engine acts as the authoritative knowledge source for evidence retrieval, passing **only** qualifying database passages to a configurable language model.

```mermaid
flowchart TD
    subgraph Client["Headless Command-Line Interface"]
        UserQuery["Clinical Query + Optional Relational Filters\n(scripts/run_rag_demo.py)"]
    end

    subgraph DatabaseEngine["PostgreSQL 18.6 + pgvector Kernel"]
        Direction TB
        SQL_Gen["Parameterized SQL Generator\n(JOIN patients, encounters, documents, chunks)"]
        ExecPlan["Search Execution\n(Exact Flat / HNSW / IVFFlat / Planner)"]
        Passages["Top-K Retrieved Clinical Passages\n[SYN-DOC-xxxx, UUID, Dept, Snippet, Distance]"]
        SQL_Gen --> ExecPlan --> Passages
    end

    subgraph PrivacyGuard["Data Privacy & Opt-In Guard"]
        CheckOptIn{"Is External LLM\nRequested?"}
        CheckOptIn -- "No (Default: Mock / Ollama)" --> LocalExecution["Execute Locally & Offline\n(Zero Network Transmission)"]
        CheckOptIn -- "Yes & --allow-external not set" --> BlockError["ExternalAccessBlockedError\n(Strict Data Exfiltration Guard)"]
        CheckOptIn -- "Yes & --allow-external set" --> CloudLLM["Opt-In External API\n(Gemini / OpenAI API)"]
    end

    subgraph DecoupledEvaluation["Decoupled Academic Evaluation Suite"]
        RetrievalEval["Part 1: Retrieval Quality\n(Top-1 Distance, Mean Distance, Lexical Overlap, Recall@K)"]
        GroundednessEval["Part 2: Generation Groundedness\n(Citation Precision, Unsupported Claims, Refusal Correctness)"]
    end

    UserQuery --> SQL_Gen
    Passages --> PrivacyGuard
    Passages --> DecoupledEvaluation
```

### Core Privacy Principles:
1. **Zero External Data Transmission by Default:** All clinical text remains local to the system. External API calls are blocked by default.
2. **Strict Opt-In Gate:** The `ExternalAPIProvider` raises `ExternalAccessBlockedError` unless explicitly opted into via `--allow-external` or `ALLOW_EXTERNAL_LLM=true`.
3. **Retrieval-Only Mode:** Supports pure database retrieval verification where language model generation is completely skipped (`--provider retrieval-only`), displaying only database passages and metadata.
4. **Deterministic Local Evaluator:** Ships with `DeterministicMockLLMProvider`, an offline rule-based generator that verifies evidence presence, enforces citations, and tests refusal protocols without network calls.

---

## 2. Evidence Grounding & Mandatory Refusal Protocol

To prevent ungrounded clinical hallucinations and speculation, the pipeline enforces strict prompting constraints:

### 2.1 Distinction of Evidence vs. Uncertainty
Every generated response must explicitly separate findings into two standardized sections:
1. `**Documented Evidence:**`  
   Every assertion must cite its source document ID in brackets (e.g. `[SYN-DOC-0134-01-01]`).
2. `**Clinical Uncertainty & Limitations:**`  
   Explicitly notes that findings are limited to the retrieved excerpt context and that longitudinal disease trajectories, unrecorded contraindications, or alternative diagnoses cannot be established from the excerpts.

### 2.2 Mandatory Abstention / Refusal Rule
If the retrieved database passages:
- Contain **zero** matching records, OR
- Have cosine distance exceeding the relevance threshold ($d > 0.82$), OR
- Have **zero lexical overlap** with query clinical keywords ($d > 0.50$),

the pipeline mandates the verbatim refusal statement:
> **"The available documents do not establish an answer to the question."**

The model is strictly prohibited from inventing speculative assertions when evidence is absent.

---

## 3. Decoupled Evaluation Framework

Rather than combining retrieval and generation into a single opaque metric, the framework evaluates retrieval quality **separately** from generation groundedness.

### 3.1 Part 1: Retrieval Quality Evaluation (`RetrievalQualityEvaluator`)
Evaluates whether the database retrieved the relevant passages:
- **`top_distance` ($d_1$):** Cosine distance of the nearest retrieved chunk.
- **`mean_distance` ($\bar{d}$):** Average cosine distance across retrieved top-k passages.
- **`top_similarity` ($1 - d_1$):** Cosine similarity score.
- **`lexical_overlap_ratio`:** Fraction of informative query clinical terms appearing in retrieved text.
- **`has_adequate_evidence`:** Boolean indicator confirming whether retrieved documents satisfy distance and keyword thresholds.
- **`recall_at_k` / `precision_at_k`:** Evaluated when ground-truth document IDs are supplied.

### 3.2 Part 2: Generation Groundedness Evaluation (`GenerationGroundednessEvaluator`)
Evaluates whether the language model remained faithful to the retrieved context:
- **`citation_precision`:**
  $$\text{Citation Precision} = \frac{|\mathcal{C}_{\text{valid}}|}{|\mathcal{C}_{\text{valid}}| + |\mathcal{C}_{\text{invalid}}|}$$
  where $\mathcal{C}_{\text{valid}}$ are cited IDs that exist in the retrieved passages, and $\mathcal{C}_{\text{invalid}}$ are fabricated citations.
- **`unsupported_claim_detection`:** Parses factual assertions in the evidence section and flags clinical terms or dosages not substantiated by the retrieved text.
- **`refusal_correctness`:**
  - `CORRECT_ABSTENTION`: Properly refused when evidence was missing or inadequate.
  - `PROPERLY_ANSWERED`: Grounded response produced when adequate evidence was present.
  - `UNGROUNDED_HALLUCINATION`: Invented an answer when evidence was missing.
  - `UNNECESSARY_REFUSAL`: Refused despite adequate evidence.
- **`distinguishes_uncertainty`:** Boolean check verifying that an uncertainty/limitations section is present.
- **`overall_groundedness_score`:** Composite groundedness score normalized between 0.0 and 1.0.

---

## 4. One-Command CLI Execution Guide

### 4.1 Running the Demonstration Suite
Execute all three curated demonstration test cases with decoupled evaluation:
```powershell
.\.venv\Scripts\python.exe scripts/run_rag_demo.py --sample-suite
```

### 4.2 Querying Specific Clinical Questions
```powershell
# 1. Supported Clinical Query with Department Filter and Evaluation
.\.venv\Scripts\python.exe scripts/run_rag_demo.py "chest pain and dyspnea on exertion" --department Cardiology --evaluate

# 2. Negative Query Demonstrating Proper Refusal (Missing Evidence)
.\.venv\Scripts\python.exe scripts/run_rag_demo.py "What surgical protocol and burr hole sizes were utilized for pediatric craniotomy?" --evaluate

# 3. Retrieval-Only Mode (Generation Intentionally Skipped)
.\.venv\Scripts\python.exe scripts/run_rag_demo.py "hypertension and elevated blood pressure" --provider retrieval-only --evaluate

# 4. JSON Formatted Output
.\.venv\Scripts\python.exe scripts/run_rag_demo.py "severe diabetic ketoacidosis" --json
```

### 4.3 Running Automated Unit & Groundedness Tests
```powershell
# Run the 8 RAG and groundedness unit tests
.\.venv\Scripts\pytest.exe tests/test_rag_pipeline.py -v

# Run the complete database project test suite (106 tests)
.\.venv\Scripts\pytest.exe tests/ -v
```

---

## 5. Empirical Case Studies (Live System Output)

### Case 1: Grounded Answer with Explicit Citations
- **Query:** `"chest pain and dyspnea on exertion"` (`--department Cardiology`, `--top-k 3`)
- **Database Search Result:**
  - `[1] SYN-DOC-0134-01-01` (Cardiology, Cosine Distance: `0.6952`, Sim: `0.3048`)
  - `[2] SYN-DOC-0090-02-01` (Cardiology, Cosine Distance: `0.6952`, Sim: `0.3048`)
  - `[3] SYN-DOC-0089-02-01` (Cardiology, Cosine Distance: `0.6952`, Sim: `0.3048`)
- **Generated Response:**
  ```markdown
  **Documented Evidence:**
  - According to [SYN-DOC-0134-01-01] (Cardiology): "CHIEF COMPLAINT: Progressive dyspnea on exertion and bilateral lower extremity edema. Denies chest pain, lightheadedness, or syncope."
  - According to [SYN-DOC-0090-02-01] (Cardiology): "CHIEF COMPLAINT: Progressive dyspnea on exertion and bilateral lower extremity edema. Denies chest pain, lightheadedness, or syncope."
  - According to [SYN-DOC-0089-02-01] (Cardiology): "CHIEF COMPLAINT: Progressive dyspnea on exertion and bilateral lower extremity edema. Denies chest pain, lightheadedness, or syncope."

  **Clinical Uncertainty & Limitations:**
  - Evidence is derived strictly from 3 retrieved synthetic clinical document chunk(s).
  - Longitudinal disease progression, unmeasured confounders, and contraindications not recorded in these specific excerpts cannot be determined from the available records.
  ```
- **Evaluation Report:**
  - Retrieval Quality: Top Distance = `0.6952`, Query Lexical Overlap = `100.0%`, Adequate Evidence = `True`.
  - Groundedness: Citation Precision = `100.0%`, Refusal Status = `PROPERLY_ANSWERED`, Unsupported Claims = `0`, Score = `1.00 / 1.00`.

---

### Case 2: Missing Evidence Abstention (Negative Test Case)
- **Query:** `"What surgical protocol and burr hole sizes were utilized for pediatric craniotomy?"`
- **Database Search Result:**
  - Top retrieved chunks are adult hypertension consults (`SYN-DOC-0155-01-01`, Cosine Distance: `0.7947`).
  - Query Lexical Overlap = `0.0%` (0 matching query keywords).
- **Generated Response:**
  ```markdown
  The available documents do not establish an answer to the question.

  **Clinical Uncertainty & Limitations:**
  - The retrieved database passages (best cosine distance: 0.7947) do not contain sufficient documented evidence regarding the query terms: ['burr', 'craniotomy', 'hole', 'pediatric', 'protocol', 'sizes', 'surgical', 'utilized'].
  - No definitive clinical conclusion or treatment recommendation can be substantiated from available records.
  ```
- **Evaluation Report:**
  - Retrieval Quality: Adequate Evidence = `False`, Lexical Overlap = `0.0%`.
  - Groundedness: Refusal Protocol Status = `CORRECT_ABSTENTION`, Unsupported Claims = `0`, Score = `1.00 / 1.00`.

---

### Case 3: External LLM Opt-In Guard Verification
- **Command:** `python scripts/run_rag_demo.py "chest pain" --provider external`
- **Output:**
  ```
  [!] SECURITY ERROR: External LLM access requires explicit opt-in.
      Clinical passages will NOT be transmitted to an external service.
      Pass --allow-external if you explicitly choose to enable external API access.
  ```
- **Outcome:** The system terminates immediately with exit code 1; no network requests are dispatched, preventing accidental PHI leakage.
