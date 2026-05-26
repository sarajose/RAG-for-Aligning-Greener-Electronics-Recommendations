#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-124
#SBATCH -J rag_eval_split
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1            # 48 GB VRAM
#SBATCH -t 0-06:00:00                    # 6 hours
#SBATCH -o logs/eval_split_%j.out
#SBATCH -e logs/eval_split_%j.err

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
unset PYTHONPATH
source /mimer/NOBACKUP/groups/naiss2026-4-124/roig/venv/bin/activate

cd /cephyr/users/roig/Alvis/rag_thesis

python main.py evaluate \
  --models bge-m3 e5-large-v2 \
  --k-values 1 3 5 10 20 \
  --output-dir outputs/eval_split_v9 \
  --evidence-csv outputs/evidence.csv \
  --top-k 10 \
  --rerank-top 5 \
  --retrieval-mode split_evidence_retrieval
