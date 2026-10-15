# PostgreSQL & pgvector Setup & Database Initialization Guide (Windows)

This document provides exact verification checks, installation instructions, and initialization workflows for **PostgreSQL** and the **`pgvector`** extension on Windows.

---

## 1. Prerequisite Checks on Windows

Before running migrations, run the following diagnostic commands in PowerShell to inspect your current environment.

### 1.1 Check PostgreSQL CLI & Version
```powershell
where.exe psql
psql --version
```
- **Expected:** Path to `psql.exe` (e.g., `C:\Program Files\PostgreSQL\18\bin\psql.exe`) and version `PostgreSQL 16.x`, `17.x`, or `18.x`.
- **If missing:** PostgreSQL is either not installed or not added to your system `PATH`. Add `C:\Program Files\PostgreSQL\<version>\bin` to your `PATH`.

### 1.2 Check PostgreSQL Windows Service
```powershell
Get-Service -Name "postgresql*"
```
- **Expected:** Status should be `Running` (e.g. `postgresql-x64-18` or `postgresql-x64-16`).
- **If stopped:** Start the service with administrative PowerShell:
  ```powershell
  Start-Service postgresql-x64-18
  ```

### 1.3 Check for `pgvector` Extension Files
Verify if the compiled C library and SQL control files are installed in your PostgreSQL directory:
```powershell
$PG_DIR = "C:\Program Files\PostgreSQL\18" # Adjust version if 16 or 17
Test-Path "$PG_DIR\lib\vector.dll"
Test-Path "$PG_DIR\share\extension\vector.control"
```
- **If `True`:** `pgvector` is installed. Proceed directly to Section 3.
- **If `False`:** Follow Section 2 to install `pgvector`.

---

## 2. Installing PostgreSQL & `pgvector` on Windows

### 2.1 Installing PostgreSQL (If Not Already Present)
If PostgreSQL is not installed on the system, install via `winget` or the official EnterpriseDB installer:
```powershell
# Using Windows Package Manager:
winget install PostgreSQL.PostgreSQL.16
```
Or download the interactive installer from [EnterpriseDB PostgreSQL Downloads](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads).

### 2.2 Installing the `pgvector` Extension
Because `pgvector` is a native C extension, it requires a compiled dynamic library (`vector.dll`) and SQL control scripts.

#### Method A: Precompiled Extension Binaries (Recommended)
1. Download the precompiled binary package for Windows matching your PostgreSQL major version (e.g. from the [pgvector GitHub releases](https://github.com/pgvector/pgvector/releases) or community Windows builds).
2. Extract the archive.
3. Copy `vector.dll` into your PostgreSQL library folder:
   ```powershell
   Copy-Item ".\vector.dll" -Destination "C:\Program Files\PostgreSQL\18\lib\" -Force
   ```
4. Copy `vector.control` and all `vector--*.sql` files into the extensions folder:
   ```powershell
   Copy-Item ".\vector.control" -Destination "C:\Program Files\PostgreSQL\18\share\extension\" -Force
   Copy-Item ".\vector--*.sql" -Destination "C:\Program Files\PostgreSQL\18\share\extension\" -Force
   ```
5. Restart PostgreSQL service:
   ```powershell
   Restart-Service postgresql-x64-18
   ```

#### Method B: Compiling from Source with Microsoft Visual C++ (MSVC)
If building from source:
1. Open the **x64 Native Tools Command Prompt for VS** (from Visual Studio 2022 / Build Tools).
2. Clone the official repository:
   ```cmd
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   ```
3. Set your PostgreSQL installation path:
   ```cmd
   set "PGROOT=C:\Program Files\PostgreSQL\18"
   ```
4. Compile and install:
   ```cmd
   nmake /F Makefile.msvc
   nmake /F Makefile.msvc install
   ```
5. Restart the PostgreSQL service.

---

## 3. Database Creation & Configuration

1. Create a dedicated database for the project (e.g. `healthcare_vectordb`):
   ```powershell
   # If prompted, provide your postgres superuser password:
   createdb -U postgres healthcare_vectordb
   ```
2. Verify connectivity:
   ```powershell
   psql -U postgres -d healthcare_vectordb -c "SELECT version();"
   ```

3. Configure your local [`.env`](file:///D:/HybridHealthcareVectorDB/.env) file from [`.env.example`](file:///D:/HybridHealthcareVectorDB/.env.example):
   ```powershell
   Copy-Item .env.example .env
   ```
   Open `.env` and set:
   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=healthcare_vectordb
   DB_USER=postgres
   DB_PASSWORD=your_actual_password
   ```

---

## 4. Running Migrations & Initializing Schema

### Method A: Automated Python Migration Runner
Run the verified database initialization script:
```powershell
python scripts/init_database.py
```
This script will:
- Connect securely using credentials from `.env`
- Sequentially apply `sql/01_extensions.sql`, `sql/02_schema.sql`, `sql/03_indexes.sql`, and `sql/04_seed_embedding_models.sql`
- Report table creation and extension verification status

### Method B: Manual CLI Execution via `psql`
Execute each script in numerical sequence:
```powershell
$env:PGPASSWORD = "your_actual_password"

psql -U postgres -d healthcare_vectordb -f sql/01_extensions.sql
psql -U postgres -d healthcare_vectordb -f sql/02_schema.sql
psql -U postgres -d healthcare_vectordb -f sql/03_indexes.sql
psql -U postgres -d healthcare_vectordb -f sql/04_seed_embedding_models.sql
```

---

## 5. Validating the Database Installation

After initialization, run the following verification query inside `psql`:

```sql
-- 1. Verify Extension
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- 2. Verify Vector Arithmetic (Cosine Distance <=>)
SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector AS orthogonal_cosine_distance;
-- Expected output: 1.0

-- 3. Verify Seeded Embedding Models
SELECT model_id, modality, dimension, distance_metric FROM embedding_models;
-- Expected output: 4 rows (MiniLM 384-d, PubMedBERT 768-d, CLIP 512-d, BiomedCLIP 512-d)

-- 4. Verify Active Relational Tables
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;
-- Expected: 9 tables (chunk_embeddings, clinical_documents, diagnoses, document_chunks, 
--           embedding_models, encounters, image_embeddings, medical_images, patients)
```
