import os
import sys

from dotenv import load_dotenv

sys.path.append(os.getcwd())
from src.database import run_ingestion

load_dotenv()

if __name__ == "__main__":
    print("Checking for PDFs in /data...")
    success = run_ingestion()
    if success:
        print("✅ Ingestion complete.")
    else:
        print("⚠️ No PDFs found. Agent will use Mock data.")