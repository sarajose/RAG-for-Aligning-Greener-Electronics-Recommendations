#!/usr/bin/env python3
"""
Entry point for the RAG policy-alignment pipeline.

Delegates to :pymod:`pipeline` which exposes the full CLI
(``build``, ``evaluate``, ``classify``, ``run``).

Usage::

    python main.py build -i outputs/evidence.csv -m bge-m3
    python main.py evaluate --gold gold_standard_doc_level/gold_standard.csv
    python main.py classify -i outputs/recommendations.csv -o outputs/classified.csv
    python main.py run -i outputs/recommendations.csv --gold gold_standard_doc_level/gold_standard.csv
"""

from pipeline import main

if __name__ == "__main__":
    main()