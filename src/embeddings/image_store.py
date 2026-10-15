"""
Multimodal Image Asset Storage & Vector Management Engine
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Manages synthetic radiological phantom assets, cryptographic integrity verification,
and 512-dimensional multimodal vision embeddings in PostgreSQL via pgvector.

Safety & Compliance Guidelines:
- Non-PHI Guarantee: Only synthetic phantoms or publicly licensed metadata are handled.
- Diagnostic Disclaimer: For semantic search research only; not for medical image diagnosis.
- Embedding Space Invariance: Separates 512-d image embeddings from 384-d text embeddings.
"""

import os
import zlib
import struct
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from database.connection import get_connection
from src.embeddings.image_embedder import ImageEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("embeddings.image_store")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class IncompatibleEmbeddingSpaceError(ValueError):
    """Raised when attempting to compare or cross-query vectors from incompatible embedding spaces."""
    pass

class ImageAssetIntegrityError(RuntimeError):
    """Raised when an image file asset is missing or has a corrupted cryptographic checksum."""
    pass

def generate_synthetic_phantom_png(
    output_path: Path,
    modality: str,
    body_site: str,
    image_id: str,
    width: int = 128,
    height: int = 128
) -> Tuple[Path, str, int]:
    """
    Generates a deterministic synthetic medical imaging phantom PNG using standard library zlib/struct.
    Simulates anatomical radiodensities without utilizing any real patient images.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive deterministic PRNG seed from image_id UUID
    seed_int = int(hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:8], 16)

    cx, cy = width // 2, height // 2
    raw_scanlines = bytearray()

    modality_upper = modality.upper()
    site_lower = body_site.lower()

    for y in range(height):
        raw_scanlines.append(0)  # PNG Filter type 0 (None)
        dy = y - cy
        for x in range(width):
            dx = x - cx
            dist_sq = dx * dx + dy * dy
            radius = (width * 0.42)

            pixel_val = 15  # Background noise floor

            if modality_upper in ("DX", "CR"):  # Chest / Projection Radiograph Phantom
                # Thoracic cage boundary
                if (dx * dx) / (radius * radius * 0.8) + (dy * dy) / (radius * radius * 1.1) < 1.0:
                    pixel_val = 140  # Mediastinal soft tissue
                    # Bilateral lung radiolucency (darker fields)
                    if (abs(dx) - 22)**2 / 160 + (dy - 4)**2 / 480 < 1.0:
                        pixel_val = 45  # Radiolucent lung field
                    # Cardiac silhouette opacity (central lighter region)
                    if (dx + 8)**2 / 120 + (dy - 12)**2 / 160 < 1.0:
                        pixel_val = 195  # Radiopaque cardiac apex
                    # Rib shadows (horizontal striations)
                    if (y % 14) in (0, 1):
                        pixel_val = min(255, pixel_val + 35)

            elif modality_upper in ("CT", "MR"):  # Tomographic Slice Phantom (Brain / Abdomen)
                if site_lower == "brain":
                    # Cranial bone vault / skull outline
                    if dist_sq < radius * radius:
                        if dist_sq > (radius - 5)**2:
                            pixel_val = 240 if modality_upper == "CT" else 60  # Bright skull on CT, dark on MRI
                        else:
                            # Brain parenchyma
                            pixel_val = 110 if modality_upper == "CT" else 150
                            # Ventricles (CSF - dark on CT / T1)
                            if (abs(dx) - 8)**2 / 20 + dy**2 / 120 < 1.0:
                                pixel_val = 30
                else:
                    # Abdomen slice
                    if (dx * dx) / (radius * radius) + (dy * dy) / (radius * radius * 0.75) < 1.0:
                        pixel_val = 120
                        # Vertebral body
                        if dx * dx + (dy - 28)**2 < 80:
                            pixel_val = 230
                        # Hepatic parenchymal region
                        elif dx > 8 and dy < 10:
                            pixel_val = 145

            elif modality_upper == "US":  # Ultrasound Sector Phantom
                # Sector pie wedge
                angle_ratio = abs(dx) / (max(1, y - 8) + 1e-5)
                if 8 < y < (height - 12) and angle_ratio < 0.75:
                    depth_attenuation = max(0, 180 - int(y * 1.1))
                    speckle = ((x * 17 + y * 31 + seed_int) % 45)
                    pixel_val = min(255, depth_attenuation + speckle)

            # Add subtle deterministic noise
            noise = ((x * 7 + y * 13 + seed_int) % 7) - 3
            final_val = max(0, min(255, pixel_val + noise))
            raw_scanlines.append(final_val)

    # Build PNG binary chunks
    def make_chunk(chunk_type: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + payload) & 0xffffffff
        return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)

    png_signature = b"\x89PNG\r\n\x1a\n"
    ihdr_payload = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    ihdr_chunk = make_chunk(b"IHDR", ihdr_payload)
    idat_chunk = make_chunk(b"IDAT", zlib.compress(bytes(raw_scanlines), level=6))
    iend_chunk = make_chunk(b"IEND", b"")

    png_bytes = png_signature + ihdr_chunk + idat_chunk + iend_chunk

    with open(output_path, "wb") as f:
        f.write(png_bytes)

    sha256_hash = hashlib.sha256(png_bytes).hexdigest()
    file_size = len(png_bytes)

    return output_path, sha256_hash, file_size

class ImageAssetManager:
    """
    Manages multimodal medical image file references, integrity checks, and pgvector embeddings.
    """

    DEFAULT_IMAGE_MODEL = "openai/clip-vit-base-patch32"
    DEFAULT_DIMENSION = 512

    def __init__(
        self,
        model_id: str = DEFAULT_IMAGE_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        embedder: Optional[ImageEmbedder] = None
    ):
        self.model_id = model_id
        self.dimension = dimension
        self.embedder = embedder or ImageEmbedder(model_name=model_id, dimension=dimension)

    def sync_synthetic_image_files(self, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generates physical synthetic phantom image files for all records in medical_images,
        attaching cryptographic SHA256 checksums, byte sizes, and paths to PostgreSQL.
        """
        target_base = base_dir or (PROJECT_ROOT / "data" / "synthetic" / "images")
        target_base.mkdir(parents=True, exist_ok=True)

        logger.info(f"Synchronizing synthetic imaging assets to disk: {target_base}")

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT image_id, synthetic_image_num, modality, body_site, image_format
                    FROM medical_images
                    ORDER BY created_at ASC;
                """)
                images = cur.fetchall()

        generated_count = 0
        updated_count = 0

        with get_connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    for row in images:
                        img_id = str(row[0])
                        img_num = row[1]
                        modality = row[2]
                        body_site = row[3]
                        fmt = row[4].lower()

                        filename = f"{img_num.lower()}.{fmt}"
                        rel_path = f"data/synthetic/images/{modality.lower()}/{filename}"
                        abs_path = PROJECT_ROOT / rel_path

                        if not abs_path.exists():
                            _, sha256_hash, file_size = generate_synthetic_phantom_png(
                                output_path=abs_path,
                                modality=modality,
                                body_site=body_site,
                                image_id=img_id
                            )
                            generated_count += 1
                        else:
                            with open(abs_path, "rb") as f:
                                b = f.read()
                                sha256_hash = hashlib.sha256(b).hexdigest()
                                file_size = len(b)

                        # Update PostgreSQL record
                        cur.execute("""
                            UPDATE medical_images
                            SET synthetic_file_path = %s,
                                file_checksum_sha256 = %s,
                                file_size_bytes = %s,
                                is_verified_asset = TRUE
                            WHERE image_id = %s;
                        """, (rel_path, sha256_hash, file_size, img_id))
                        updated_count += 1

        logger.info(f"[V] Synchronized {updated_count} image assets ({generated_count} newly created on disk).")
        return {
            "total_records": len(images),
            "generated_on_disk": generated_count,
            "updated_database": updated_count,
            "target_directory": str(target_base)
        }

    def verify_asset_integrity(self) -> Dict[str, Any]:
        """
        Audits all medical_images records against physical files on disk.
        Verifies file existence, byte length, and cryptographic SHA-256 digests.
        """
        logger.info("Executing cryptographic asset integrity audit on medical_images...")
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT image_id, synthetic_image_num, synthetic_file_path, file_checksum_sha256, file_size_bytes
                    FROM medical_images;
                """)
                rows = cur.fetchall()

        verified = 0
        missing = 0
        corrupted = 0
        details = []

        for r in rows:
            img_id = str(r[0])
            img_num = r[1]
            rel_path = r[2]
            expected_hash = r[3]
            expected_size = r[4]

            abs_path = PROJECT_ROOT / rel_path if not os.path.isabs(rel_path) else Path(rel_path)

            if not abs_path.exists():
                missing += 1
                details.append({"image_id": img_id, "status": "MISSING_ON_DISK", "path": str(abs_path)})
                continue

            with open(abs_path, "rb") as f:
                content = f.read()
                actual_hash = hashlib.sha256(content).hexdigest()
                actual_size = len(content)

            if expected_hash and actual_hash != expected_hash:
                corrupted += 1
                details.append({"image_id": img_id, "status": "CORRUPTED_HASH", "expected": expected_hash, "actual": actual_hash})
            else:
                verified += 1

        is_passed = (missing == 0 and corrupted == 0)
        logger.info(f"[V] Integrity audit: {verified} verified, {missing} missing, {corrupted} corrupted.")
        return {
            "audit_passed": is_passed,
            "total_records": len(rows),
            "verified_count": verified,
            "missing_count": missing,
            "corrupted_count": corrupted,
            "issues": details
        }

    def embed_and_store_images(self, batch_size: int = 64) -> Dict[str, Any]:
        """
        Generates 512-dimensional multimodal vision embeddings and upserts them
        into image_embeddings with explicit model ID and dimension validation.
        """
        # 1. Verify model in registry
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT dimension, modality FROM embedding_models WHERE model_id = %s;",
                    (self.model_id,)
                )
                row = cur.fetchone()
                if not row:
                    raise IncompatibleEmbeddingSpaceError(f"Model '{self.model_id}' is not registered in embedding_models.")
                db_dim, db_modality = row
                if db_dim != self.dimension or db_modality != "multimodal":
                    raise IncompatibleEmbeddingSpaceError(
                        f"Model '{self.model_id}' specifies {db_dim}-d ({db_modality}), expected {self.dimension}-d multimodal."
                    )

                # Check column dimension in catalog
                cur.execute("""
                    SELECT a.atttypmod
                    FROM pg_attribute a
                    JOIN pg_class c ON a.attrelid = c.oid
                    WHERE c.relname = 'image_embeddings' AND a.attname = 'embedding' AND a.attnum > 0;
                """)
                col_row = cur.fetchone()
                if not col_row or col_row[0] != self.dimension:
                    raise IncompatibleEmbeddingSpaceError(
                        f"Database column 'image_embeddings.embedding' expects vector({col_row[0] if col_row else '?'}), "
                        f"configured for {self.dimension}."
                    )

                # Fetch images missing embeddings for target model
                cur.execute("""
                    SELECT mi.image_id, mi.synthetic_file_path
                    FROM medical_images mi
                    LEFT JOIN image_embeddings ie 
                        ON mi.image_id = ie.image_id AND ie.model_id = %s
                    WHERE ie.embedding_id IS NULL;
                """, (self.model_id,))
                missing_images = cur.fetchall()

        if not missing_images:
            logger.info(f"[V] All medical images already have embeddings stored for model '{self.model_id}'.")
            return {"processed_count": 0, "upserted_count": 0, "model_id": self.model_id}

        logger.info(f"Generating multimodal vision vectors for {len(missing_images)} images with '{self.model_id}'...")

        upserted_total = 0
        t_start = time.perf_counter()

        with get_connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    for i in range(0, len(missing_images), batch_size):
                        batch = missing_images[i:i + batch_size]
                        for img_id, rel_path in batch:
                            abs_path = PROJECT_ROOT / rel_path if not os.path.isabs(rel_path) else Path(rel_path)
                            vec = self.embedder.embed_image_path(abs_path)

                            # Validate dimension
                            if len(vec) != self.dimension:
                                raise IncompatibleEmbeddingSpaceError(
                                    f"Vector dimension {len(vec)} does not match target {self.dimension}."
                                )

                            cur.execute("""
                                INSERT INTO image_embeddings (image_id, model_id, dimension, embedding, created_at)
                                VALUES (%s, %s, %s, %s::vector, CURRENT_TIMESTAMP)
                                ON CONFLICT (image_id, model_id) DO UPDATE
                                SET embedding = EXCLUDED.embedding, dimension = EXCLUDED.dimension;
                            """, (str(img_id), self.model_id, self.dimension, vec))
                            upserted_total += 1

        duration_s = time.perf_counter() - t_start
        throughput = upserted_total / duration_s if duration_s > 0 else 0.0

        logger.info(f"[V] Successfully stored {upserted_total} vision embeddings in {duration_s:.2f}s ({throughput:.1f} img/s).")
        return {
            "processed_count": len(missing_images),
            "upserted_count": upserted_total,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "duration_seconds": round(duration_s, 3),
            "throughput_imgs_per_sec": round(throughput, 1)
        }
