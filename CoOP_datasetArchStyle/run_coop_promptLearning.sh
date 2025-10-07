#!/bin/bash
#SBATCH --job-name=coop_train_light 
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2025
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G   
#SBATCH --time=15:00:00
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_A40_48G"
#SBATCH --output=/work/cvcs2025/carpaneto_salsi_serafini/ArClip/CoOp/coop_outputs/%x_%j.out
#SBATCH --error=/work/cvcs2025/carpaneto_salsi_serafini/ArClip/CoOp/coop_outputs/%x_%j.err


#-e (exit on error)
#-o (the return value of a pipeline is the status of the last command to exit with a non-zero status)
set -eo pipefail

source /homes/admin/spack/opt/spack/linux-ivybridge/anaconda3-2023.09-0-dyblbfizocp3rfv77i4yhkbvbb5xamnh/bin/activate clipVENV

mkdir -p logs

# Execute the 1 training script
echo "- - - - - - - - - - - - - - - - - - - - "
echo "[--INFO--] Esecuzione per_Class con Prompt = TEMPLATES[] shared"
python /work/cvcs2025/carpaneto_salsi_serafini/ArClip/CoOp/coop_hard/coop_cluster_promptLearning.py --save /work/cvcs2025/carpaneto_salsi_serafini/ArClip/CoOp/coop_outputs/prompt_ensemble_weights.pt

# Execute the 2 training script
echo "- - - - - - - - - - - - - - - - - - - - "
echo "[--INFO--] Esecuzione per_Class con Prompt = TEMPLATES[] shared"
python /work/cvcs2025/carpaneto_salsi_serafini/ArClip/CoOp/coop_hard/coop_cluster_promptLearning.py --per-class --save /work/cvcs2025/carpaneto_salsi_serafini/ArClip/CoOp/coop_outputs/prompt_ensemble_weights.pt


# Execute the 3 training script
echo "[--INFO--] Esecuzione Shared con prompt = {} "
python /work/cvcs2025/carpaneto_salsi_serafini/ArClip/CoOp/coop_hard/coop_cluster_shared.py --template-zero '{}' --template-ctx '{}'

# Execute the 4 training script
echo "- - - - - - - - - - - - - - - - - - - - "
echo "[--INFO--] Esecuzione per_Class con prompt = {}"
python /work/cvcs2025/carpaneto_salsi_serafini/ArClip/CoOp/coop_hard/coop_cluster_perClass.py --template-zero '{}' --template-ctx '{}'

# Execute the 5 training script
echo "[--INFO--] Esecuzione Shared con prompt = a photo of a {} "
python /work/cvcs2025/carpaneto_salsi_serafini/ArClip/CoOp/coop_hard/coop_cluster_shared.py 

# Execute the 6 training script
echo "- - - - - - - - - - - - - - - - - - - - "
echo "[--INFO--] Esecuzione per_Class con prompt = a photo of a {}"
python /work/cvcs2025/carpaneto_salsi_serafini/ArClip/CoOp/coop_hard/coop_cluster_perClass.py






echo "[--INFO--] Job completato"