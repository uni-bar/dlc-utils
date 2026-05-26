# Workflow Scripts

These are the user-facing entrypoint scripts.
Edit the parameters near the top of the file you need, then run that file.

## Which script to use

- `retrain_front_top_from_retrained.sh`
  - Most common retrain flow.
  - Uses all labels under `~/sandbox/dlc-utils/manual_labels`.
  - Starts from `front_head_only_resnet_152_retrained`.
  - Saves the exported model to `front_top_head_resnet_152_retrained`.

- `retrain_front_head_only_resnet_152_retrained.sh`
  - Lower-level retrain preset.
  - Edit this if you want to change the source project, base model, output model, or GPU/retrain defaults.

- `run_retrained_dlc.sh`
  - Runs PreyTouch prediction with the currently selected retrained model on multiple date folders.

- `run_retrained_dlc_nohup.sh`
  - Separate best-effort detached variant of the rerun script for cases where `nohup` works well in your current session.

- `run_retrained_dlc_slurm.sbatch`
  - Submits the same rerun through Slurm so it keeps going after your laptop sleeps or your SSH session disconnects.

- `rerun_front_with_retrained_dlc.sh`
  - Front-camera rerun preset across several dates.

- `rerun_pv82_20260123_front_with_retrained_dlc.sh`
  - Single-date front-camera rerun preset for PV82 on 20260123.

## Generic helpers kept outside this folder

- `../retrain.sh`
  - Generic CLI wrapper for retraining.

- `../retrain_dlc_from_manual_labels.py`
  - Python helper that builds the labeled dataset and runs DLC retraining/export.
