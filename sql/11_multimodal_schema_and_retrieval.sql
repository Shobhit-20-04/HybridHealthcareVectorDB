-- =============================================================================
-- Migration 11: Multimodal Storage, Image Embeddings & Cross-Modal Retrieval
-- Project: Hybrid Relational–Vector Database Architecture for Healthcare
-- Extension: PostgreSQL pgvector (v0.8.6+)
-- =============================================================================
-- This migration extends the schema for multimodal healthcare assets:
--   1. Adds asset integrity tracking columns to medical_images (SHA256 checksum, file size, verification flag)
--   2. Enforces model registry alignment and prevents cross-space vector pollution
--   3. Provides repeatable Image-to-Image and Cross-Modal (Text-to-Image) similarity queries
--   4. Validates physical HNSW index scans on image_embeddings (512 dimensions)
-- =============================================================================

-- =============================================================================
-- SECTION 1: Asset Integrity Enhancements on medical_images
-- =============================================================================

ALTER TABLE medical_images 
    ADD COLUMN IF NOT EXISTS file_checksum_sha256 VARCHAR(64);

ALTER TABLE medical_images 
    ADD COLUMN IF NOT EXISTS file_size_bytes BIGINT;

ALTER TABLE medical_images 
    ADD COLUMN IF NOT EXISTS is_verified_asset BOOLEAN DEFAULT FALSE;

COMMENT ON COLUMN medical_images.file_checksum_sha256 IS 'Cryptographic SHA-256 digest of the synthetic imaging file asset on disk.';
COMMENT ON COLUMN medical_images.file_size_bytes IS 'On-disk file size of the synthetic image asset in bytes.';
COMMENT ON COLUMN medical_images.is_verified_asset IS 'True if the file reference has been verified to physically exist with valid checksum.';

-- Index on imaging asset verification status
CREATE INDEX IF NOT EXISTS idx_medical_images_verified 
    ON medical_images (is_verified_asset);

-- =============================================================================
-- SECTION 2: Model Registry Invariants & Multimodal Separation
-- Enforces that 384-d text models cannot be stored in image_embeddings,
-- and 512-d vision models cannot be stored in chunk_embeddings.
-- =============================================================================

-- Ensure multimodal CLIP model is registered in embedding_models
INSERT INTO embedding_models (model_id, modality, dimension, distance_metric, description)
VALUES (
    'openai/clip-vit-base-patch32',
    'multimodal',
    512,
    'cosine',
    'OpenAI CLIP ViT-B/32 multimodal vision-language model (512-dim normalized unit vectors).'
)
ON CONFLICT (model_id) DO NOTHING;

-- Verification check: chunk_embeddings requires 384-d; image_embeddings requires 512-d
-- Incompatible vector comparison between chunk_embeddings and image_embeddings is
-- rejected by PostgreSQL type-system due to dimension mismatch: vector(384) != vector(512).

-- =============================================================================
-- SECTION 3: Image-to-Image Similarity Retrieval Query Template
-- Retrieves nearest radiological studies given a 512-dimensional query image embedding.
-- =============================================================================

-- Replace '[0.01, ...]' with target 512-dimensional query vision vector
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT 
    ie.image_id,
    mi.synthetic_image_num,
    mi.modality,
    mi.body_site,
    mi.synthetic_file_path,
    mi.file_checksum_sha256,
    p.synthetic_mrn,
    p.gender,
    e.department,
    cd.title AS associated_report_title,
    (ie.embedding <=> '[0.01, 0.02, ...]'::vector) AS cosine_distance,
    (1.0 - (ie.embedding <=> '[0.01, 0.02, ...]'::vector)) AS cosine_similarity
FROM image_embeddings ie
JOIN medical_images mi ON ie.image_id = mi.image_id
JOIN encounters e ON mi.encounter_id = e.encounter_id
JOIN patients p ON mi.patient_id = p.patient_id
LEFT JOIN clinical_documents cd ON mi.document_id = cd.document_id
WHERE ie.model_id = 'openai/clip-vit-base-patch32'
  AND mi.modality = 'DX'
ORDER BY ie.embedding <=> '[0.01, 0.02, ...]'::vector ASC
LIMIT 5;

-- =============================================================================
-- SECTION 4: Cross-Modal Text-to-Image Retrieval Query Template
-- Retrieves medical images matching a natural language clinical description
-- encoded into the shared 512-dimensional CLIP multimodal latent space.
-- =============================================================================

-- Replace '[0.01, ...]' with 512-dim CLIP text-tower embedding of query text
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT 
    ie.image_id,
    mi.synthetic_image_num,
    mi.modality,
    mi.body_site,
    mi.synthetic_file_path,
    p.synthetic_mrn,
    e.department,
    cd.title AS related_clinical_note,
    (ie.embedding <=> '[0.01, 0.02, ...]'::vector) AS cosine_distance,
    (1.0 - (ie.embedding <=> '[0.01, 0.02, ...]'::vector)) AS cosine_similarity
FROM image_embeddings ie
JOIN medical_images mi ON ie.image_id = mi.image_id
JOIN encounters e ON mi.encounter_id = e.encounter_id
JOIN patients p ON mi.patient_id = p.patient_id
LEFT JOIN clinical_documents cd ON mi.document_id = cd.document_id
WHERE ie.model_id = 'openai/clip-vit-base-patch32'
ORDER BY ie.embedding <=> '[0.01, 0.02, ...]'::vector ASC
LIMIT 5;
