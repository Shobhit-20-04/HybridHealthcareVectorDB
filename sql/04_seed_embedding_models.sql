-- =============================================================================
-- Migration 04: Seed Canonical Embedding Models Registry
-- Project: Hybrid Relational–Vector Database Architecture for Healthcare
-- =============================================================================

INSERT INTO embedding_models (model_id, model_name, modality, dimension, distance_metric, description)
VALUES 
    (
        'sentence-transformers/all-MiniLM-L6-v2',
        'All MiniLM L6 v2 (General English Sentence Transformer)',
        'text',
        384,
        'cosine',
        'Standard 384-dimensional dense text embedding model with fast inference and strong general semantic retrieval performance.'
    ),
    (
        'pritamdeka/S-PubMedBert-MS-MARCO',
        'PubMedBERT Sentence Transformer MS-MARCO',
        'text',
        768,
        'cosine',
        'Biomedical domain-adapted sentence transformer pretrained on PubMed abstracts and fine-tuned for semantic passage retrieval.'
    ),
    (
        'openai/clip-vit-base-patch32',
        'OpenAI CLIP ViT-B/32',
        'multimodal',
        512,
        'cosine',
        'Vision-language foundation model projecting images and natural language queries into a unified 512-dimensional embedding space.'
    ),
    (
        'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        'Microsoft BiomedCLIP',
        'multimodal',
        512,
        'cosine',
        'Biomedical multimodal model trained on PMCOA pairs (biomedical text and clinical imaging) for cross-modal medical retrieval.'
    )
ON CONFLICT (model_id) DO UPDATE SET
    model_name = EXCLUDED.model_name,
    modality = EXCLUDED.modality,
    dimension = EXCLUDED.dimension,
    distance_metric = EXCLUDED.distance_metric,
    description = EXCLUDED.description;
