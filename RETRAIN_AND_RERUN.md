## Retrain + rerun (DLC)

First retrain (updates the exported model under `~/data/rep4/output/models/deeplabcut/front_head_only_resnet_152_retrained/`):

```bash
cd ~/sandbox/dlc-utils/workflows && ./retrain_front_head_only_resnet_152_retrained.sh
```

Then rerun predictions on the date folders you configured in the script (always uses the latest retrained model via `--model_override`):

```bash
cd ~/sandbox/dlc-utils/workflows && ./rerun_front_with_retrained_dlc.sh
```

For the simpler "new labels -> retrain from last retrained model -> save as front_top" flow, run:

```bash
cd ~/sandbox/dlc-utils/workflows && ./retrain_front_top_from_retrained.sh
```

All reusable entrypoint scripts now live under `~/sandbox/dlc-utils/workflows/`.

If you add/modify labeled frames in `~/sandbox/dlc-utils/manual_labels`, run the retrain command again before rerunning predictions.
