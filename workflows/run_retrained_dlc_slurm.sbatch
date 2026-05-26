#!/usr/bin/env bash
#SBATCH --job-name=run-retrained-dlc
#SBATCH --account=sheinmark-users
#SBATCH --partition=gpu-general
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch200/%u/logs/dlc/%x-%j.out
#SBATCH --error=/scratch200/%u/logs/dlc/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "/scratch200/$USER/logs/dlc"

echo "[INFO] Started at: $(date '+%F %T %Z')"
echo "[INFO] Hostname:   $(hostname)"
echo "[INFO] Job ID:     ${SLURM_JOB_ID:-<unset>}"

cd "$SCRIPT_DIR"

bash "$SCRIPT_DIR/run_retrained_dlc.sh"

echo "[INFO] Finished at: $(date '+%F %T %Z')"
