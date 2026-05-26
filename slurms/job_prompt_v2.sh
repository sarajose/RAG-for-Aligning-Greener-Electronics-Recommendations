#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-124
#SBATCH -J rag_prompt
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1            # 48 GB VRAM
#SBATCH -t 0-03:00:00                    # 3 hours
#SBATCH -o logs/prompt_%j.out
#SBATCH -e logs/prompt_%j.err

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
unset PYTHONPATH
source /mimer/NOBACKUP/groups/naiss2026-4-124/roig/venv/bin/activate

cd /cephyr/users/roig/Alvis/rag_thesis

python main.py prompt \
  --input data/recommendations_whitepaper/recommendations_v2.csv \
  --output outputs/bge_m3_split_dense_rerank_v8.csv \
  --model bge-m3 \
  --top-k 10 \
  --rerank-top 5 \
  --retrieval-mode split_evidence_retrieval \
  --inner-retrieval-method dense \
  --judge
