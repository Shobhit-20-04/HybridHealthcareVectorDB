# PostgreSQL & pgvector Windows Setup and Verification Manual

This document provides exact, reproducible steps for setting up, configuring, and verifying **PostgreSQL** with the **`pgvector`** extension on Windows.

---

## 1. Quick Verification Checklist (Run in PowerShell)

Execute these diagnostic commands in PowerShell to inspect the local environment:

```powershell
# 1. Check if psql CLI is in PATH
where.exe psql
psql --version

# 2. Check if the PostgreSQL Windows service is running
Get-Service -Name "postgresql*"

# 3. Check TCP port reachability (Port 5432)
Test-NetConnection -ComputerName localhost -Port 5432

# 4. Check if pgvector compiled binaries are present in PostgreSQL 18
$PG_DIR = "C:\Program Files\PostgreSQL\18"  # Adjust version if 16 or 17
Test-Path "$PG_DIR\lib\vector.dll"
Test-Path "$PG_DIR\share\extension\vector.control"
```

---

## 2. PostgreSQL and `pgvector` Installation Instructions

### 2.1 PostgreSQL Status
- If PostgreSQL is already installed (e.g. `postgresql-x64-18` running), do not re-install.
- If PostgreSQL is not installed, download the official installer from [EnterpriseDB PostgreSQL](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads) or via `winget install PostgreSQL.PostgreSQL.16`.

### 2.2 Installing the `pgvector` Extension on Windows
PostgreSQL requires compiled C binaries (`vector.dll`) and extension control files to enable vector data types and HNSW/IVFFlat indexing.

#### Option A: Precompiled Binary Files (Fastest & Safest)
1. Download the precompiled Windows package matching your PostgreSQL major version (PostgreSQL 16, 17, or 18) from [pgvector Releases](https://github.com/pgvector/pgvector/releases).
2. Extract the archive and copy the files with Administrator privileges:
   ```powershell
   # Copy dynamic library into PostgreSQL lib
   Copy-Item ".\vector.dll" -Destination "C:\Program Files\PostgreSQL\18\lib\" -Force

   # Copy control and SQL files into extension folder
   Copy-Item ".\vector.control" -Destination "C:\Program Files\PostgreSQL\18\share\extension\" -Force
   Copy-Item ".\vector--*.sql" -Destination "C:\Program Files\PostgreSQL\18\share\extension\" -Force
   ```
3. Restart the PostgreSQL Windows service to load the extension:
   ```powershell
   Restart-Service postgresql-x64-18
   ```

#### Option B: Compiling from Source via Visual Studio C++ (MSVC)
If you have Visual Studio C++ Build Tools installed:
1. Open the **x64 Native Tools Command Prompt for VS**.
2. Clone and build:
   ```cmd
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   set "PGROOT=C:\Program Files\PostgreSQL\18"
   nmake /F Makefile.msvc
   nmake /F Makefile.msvc install
   ```
3. Restart the PostgreSQL service.

---

## 3. Database Creation & Credentials Setup

### Step 3.1: Create Project Database
Open PowerShell and create the dedicated database:
```powershell
# Create database (you will be prompted for your postgres password):
createdb -U postgres healthcare_vectordb
```
Alternatively, execute the repeatable SQL script:
```powershell
psql -U postgres -d postgres -f sql/00_create_database.sql
```

### Step 3.2: Configure `.env`
Copy the safe configuration template:
```powershell
Copy-Item .env.example .env
```
Edit `.env` with your preferred editor and fill in your local PostgreSQL password:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=healthcare_vectordb
DB_USER=postgres
DB_PASSWORD=your_actual_postgres_password
DB_SSLMODE=prefer
```

---

## 4. Applying Database Migrations

You can apply the schema migrations using either the Python runner or the master SQL script:

### Method A: Automated Python Migration Runner
```powershell
.\.venv\Scripts\python.exe scripts/init_database.py
```
This executes:
1. `sql/01_extensions.sql` (Enables `vector`, `uuid-ossp`, and `pgcrypto`)
2. `sql/02_schema.sql` (Creates all 9 normalized relational & vector tables)
3. `sql/03_indexes.sql` (Creates B-Tree foreign keys, composite filters, GIN, and HNSW indexes)
4. `sql/04_seed_embedding_models.sql` (Seeds canonical model definitions)
5. `sql/06_security_rls.sql` (Enables Row-Level Security and creates role policies)

### Method B: Native `psql` Master Script
```powershell
$env:PGPASSWORD = "your_actual_postgres_password"
psql -U postgres -d healthcare_vectordb -f sql/init_all.sql
```

---

## 5. Running the Database Health-Check

Run the diagnostic health-check script to verify the entire stack:
```powershell
.\.venv\Scripts\python.exe scripts/db_healthcheck.py
```

### Expected Output upon Successful Setup:
```
==============================================================================
  POSTGRESQL & PGVECTOR HEALTH-CHECK DIAGNOSTIC REPORT
==============================================================================
[1] Environment Configuration Check:
    [V] .env file detected at project root.
[2] Network Connectivity Check:
    [V] PostgreSQL port 5432 on 'localhost' is REACHABLE.
[3] Database Authentication & Connection Check:
    [V] Connected successfully!
    - Connected Database : healthcare_vectordb
    - Connected User     : postgres
[4] pgvector Extension Verification:
    [V] pgvector Extension is ACTIVE (Version 0.8.x).
    [V] Cosine operator (<=>) test PASSED.
[5] Schema Tables & Row Counts Audit:
    Table Name               | Status     | Rows     | RLS Policy
    ----------------------------------------------------------
    patients                 | Present    | 100      | Off
    encounters               | Present    | 248      | Off
    diagnoses                | Present    | 360      | Off
    clinical_documents       | Present    | 386      | ENABLED
    document_chunks          | Present    | 453      | ENABLED
    embedding_models         | Present    | 4        | Off
    chunk_embeddings         | Present    | 453      | ENABLED
    medical_images           | Present    | 179      | ENABLED
    image_embeddings         | Present    | 179      | ENABLED
    [V] All 9 approved schema tables are present in the database.
==============================================================================
```

---

## 6. Verification with SQL Script

To execute deep vector arithmetic and HNSW index tests directly within PostgreSQL:
```powershell
psql -U postgres -d healthcare_vectordb -f sql/07_verify_vector_support.sql
```
This verifies:
1. `vector` extension version in `pg_extension`
2. Cosine distance (`<=>`), L2 distance (`<->`), and Inner Product (`<#>`)
3. All 9 approved tables in `information_schema.tables`
4. Column data types and explicit dimension bounds
5. Temporary HNSW index construction and approximate nearest neighbor query execution.
