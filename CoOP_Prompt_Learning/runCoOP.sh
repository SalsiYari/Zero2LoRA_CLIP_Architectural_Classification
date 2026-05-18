#!/bin/bash
#SBATCH --job-name=clip_experiments
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2025
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G   
#SBATCH --time=10:00:00
#SBATCH --constraint="gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_A40_48G"
#SBATCH --output=/homes/ysalsi/ArClip/Outputs/%x_%j.out
#SBATCH --error=/homes/ysalsi/ArClip/Outputs/%x_%j.err

#-e (exit on error)
#-o (the return value of a pipeline is the status of the last command to exit with a non-zero status)
set -eo pipefail

source /homes/admin/spack/opt/spack/linux-ivybridge/anaconda3-2023.09-0-dyblbfizocp3rfv77i4yhkbvbb5xamnh/bin/activate clipVENV

PROJECT_DIR="/work/cvcs2025/carpaneto_salsi_serafini/ArClip/CoOP"
cd $PROJECT_DIR

echo "=========================================================="
echo "Starting Runs - Job ID: $SLURM_JOB_ID"
echo "=========================================================="

# ==========================================
# ESPERIMENTI CoOp (Context Optimization)
# ==========================================

echo ""
echo "[--INFO--] 1. Esecuzione CoOp ===> Modalità: SHARED (Context unificato)"
# --no-tqdm è molto utile su SLURM per non riempire il file di log (.out) con le righe di aggiornamento della barra
python main_coop.py --mode shared --epochs 10 --no-tqdm

echo "- - - - - - - - - - - - - - - - - - - - "
echo "[--INFO--] 2. Esecuzione CoOp ===> Modalità: PER_CLASS (Context specifico)"
python main_coop.py --mode per_class --epochs 10 --no-tqdm


# ==========================================
# ESPERIMENTI PROMPT ENSEMBLE
# ==========================================

echo "- - - - - - - - - - - - - - - - - - - - "
echo "[--INFO--] 3. Esecuzione Ensemble ===> Modalità: GLOBALE"
# --eval-uniform fa stampare anche la baseline senza addestramento
python main_ensemble.py --epochs 10 --eval-uniform --save ensemble_weights_global.pt

echo "- - - - - - - - - - - - - - - - - - - - "
echo "[--INFO--] 4. Esecuzione Ensemble ===> Modalità: PER_CLASS"
python main_ensemble.py --epochs 10 --per-class --eval-uniform --save ensemble_weights_per_class.pt


echo "=========================================================="
echo "[--INFO--] Tutti i Job completati con successo!"
echo "=========================================================="
