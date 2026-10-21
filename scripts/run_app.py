"""
Interactive Server & Frontend Launcher for Hybrid Healthcare VectorDB
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Starts the FastAPI server which serves both the REST API endpoints and the built React frontend.
Accessible at http://127.0.0.1:8000
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    frontend_dist = PROJECT_ROOT / "frontend" / "dist"
    
    print("=" * 80)
    print("  HYBRID HEALTHCARE VECTORDB — REACT FRONTEND & FASTAPI BACKEND")
    print("=" * 80)

    if not frontend_dist.exists() or not (frontend_dist / "index.html").exists():
        print("[*] Building production React bundle (frontend/dist)...")
        try:
            subprocess.run(["npm", "run", "build"], cwd=str(PROJECT_ROOT / "frontend"), check=True, shell=True)
            print("[V] React build completed successfully.")
        except Exception as e:
            print(f"[!] Warning: Failed to build React bundle automatically: {e}")
            print("    Run 'cd frontend; npm run build' manually.")

    print("\n[+] Launching FastAPI Application Server at http://127.0.0.1:8000")
    print("    - REST API & Endpoints   : http://127.0.0.1:8000/api")
    print("    - Interactive API Docs   : http://127.0.0.1:8000/docs")
    print("    - React Web Application  : http://127.0.0.1:8000")
    print("    (For live Vite hot-reload, run 'cd frontend; npm run dev' in a separate terminal)")
    print("=" * 80 + "\n")

    try:
        import uvicorn
        uvicorn.run("server.app:app", host="127.0.0.1", port=8000, reload=False)
    except KeyboardInterrupt:
        print("\n[V] Server stopped gracefully.")

if __name__ == "__main__":
    main()
