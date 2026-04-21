# Task A Interactive Trajectory Game (Pygame)

This tool lets you freely draw missing trajectory segments. The system will resample your drawn curve automatically.

During gameplay, reference routes are off by default:
- GT / baseline23 / baseline28 are hidden at startup.
- You can toggle them on for inspection when needed.
- They are always used in exported metrics and case comparison figures.

Outputs per case:
- player_pred.pkl
- metrics.json
- case_overlay.png

Session outputs:
- case_pool.json
- session_summary.json
- session_state.json
- player_predictions_saved_cases.pkl
- player_predictions_all_loaded_cases.pkl

## 1. Install dependency

Run in your existing venv:

pip install pygame

## 2. Quick start

From workspace root:

python task_A_recovery/interactive_game.py --dataset 16

Two-round player study launcher (1/8 then 1/16, 40+40 cases by default):

python task_A_recovery/launch_player_study.py --cases-per-dataset 40

Defaults:
- input: task_A_recovery/val_input_16.pkl
- gt: task_A_recovery/val_gt.pkl
- pred23: task_A_recovery/pred_hmm_val_16_b23_e5_gapaware.pkl
- pred28: task_A_recovery/pred_hmm_val_16_b28_turncurve.pkl
- map: map
- case pool size: 30

## 3. Controls

Mouse:
- Left click / drag: draw node chain
- Right click: delete nearest node

Drawing rule:
- Nodes are connected in sequence by straight lines.
- Right-click deletion removes that node and reconnects its previous/next nodes.
- You only need to sketch the curve shape for each gap.
- You do NOT need to click exact endpoint known points.
- For each gap, the system anchors to the two known points and uniformly resamples to the missing-point count.

Keyboard:
- Enter: auto-finish drawn gaps and submit
- Backspace: remove last created node
- Left/Right: previous/next gap (Right auto-finishes current gap)
- U or Ctrl+Z: undo current gap
- R: reset current gap
- G: toggle GT visibility
- 1: toggle baseline23 visibility
- 2: toggle baseline28 visibility
- M: toggle map visibility
- S: save current case outputs (submitted case only)
- N: next case. If not submitted, this case is skipped and not recorded.
- T: replay current case (reset to initial state; next submitted save overwrites old one)
- P: go to previous case
- H: toggle help text
- Q or Esc: quit and save session summary

## 4. Path options

### Build case pool only (no UI)

python task_A_recovery/interactive_game.py --dataset 8 --no-ui --case-pool-size 30

### Show round/progress label in UI panel

python task_A_recovery/interactive_game.py --dataset 8 --round-label round_1_of_2 --progress-offset 0 --progress-total 80

### Play one specific trajectory

python task_A_recovery/interactive_game.py --dataset 16 --traj-id 12345

### Use a prebuilt case pool

python task_A_recovery/interactive_game.py --dataset 16 --case-pool-json task_A_recovery/game_outputs/session_xxx/case_pool.json

### Disable map overlay

python task_A_recovery/interactive_game.py --dataset 16 --disable-map

## 6. Player study workflow (60-100 cases)

Recommended:
- Use 30~50 cases per dataset (total 60~100 cases).
- Run two rounds in fixed order: dataset 1/8 first, then dataset 1/16.

Launch example:

python task_A_recovery/launch_player_study.py --cases-per-dataset 40 --seed 42

This creates session names:
- <prefix>_r8
- <prefix>_r16

After finishing both rounds, run analysis:

python task_A_recovery/analyze_player_study.py --session-dir-8 task_A_recovery/game_outputs/<prefix>_r8 --session-dir-16 task_A_recovery/game_outputs/<prefix>_r16 --out-dir task_A_recovery/game_outputs/<prefix>_analysis

Main analysis outputs:
- case_metrics.csv
- global_metrics.json
- summary.md
- case_montage_all.png
- case_montage_dataset8.png
- case_montage_dataset16.png

## 7. Notes

- The editor only fills missing intervals between known points.
- Player does not need to place exactly missing_count nodes.
- Player only draws curve shape; system handles sampling.
- Known points remain locked and are never modified.
- Gap filling always uses start/end known points as anchors and resamples uniformly along the drawn path.
- Case pool is stratified by combined MAE of baseline23 and baseline28.
