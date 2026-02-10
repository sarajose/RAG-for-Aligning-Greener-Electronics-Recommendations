#!/usr/bin/env python3
"""
main.py - Orchestrates:
  - EU evidence scraping+chunking (chunking_evidence.py)
  - recommendation extraction (chunking_recommendations.py)

Does NOT modify your modules. It tries to call their functions robustly.

usage: python main.py --mode recs --recs-dir "data/recommendations"
"""


#import chunking_evidence
import chunking_recommendations


if __name__ == "__main__":
    pass