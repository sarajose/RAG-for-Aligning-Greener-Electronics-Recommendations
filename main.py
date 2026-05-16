#!/usr/bin/env python3
"""
Entry point for the RAG policy-alignment pipeline.

Exposes the full CLI: build, prompt, evaluate and download-models

Usage:
    python main.py build -i outputs/evidence.csv -m bge-m3
    python main.py prompt -i data/recommendations_whitepaper/recommendations_empty.csv --judge
    python main.py evaluate --models bge-m3 e5-large-v2
"""
from pipeline import main

if __name__ == "__main__":
    main()