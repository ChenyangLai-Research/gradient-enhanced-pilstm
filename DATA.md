# Local data contract

No research data are bundled. Obtain access from the dataset owner; a publication DOI does not grant redistribution rights. `--data-root` is the root of the corresponding original project, not this repository.

Under `battery_final/varying_LR_PINN_GE/1st_test_fold/`: `trajectory_cell_B1.mat`, `trajectory_cell_B2.mat`, `trajectory_cell_B3.mat`, each containing the correspondingly named MATLAB cell array. Trajectories have 32 columns. Use columns 0:24 excluding index 13 (23 features), the last column as RUL, and row index as time.

## Preprocessing and split

Split entire cells deterministically 80/20 (seed 22). Fit feature, time and RUL normalization on training cells. Use nonoverlapping length-32 windows, hidden size 32, one LSTM layer and output 33→16→1. Loss weights: data 0.9, first derivative 0.1, second derivative 0.01. Normalized dRUL/dt = -range(time)/range(RUL).

Missing files produce an actionable error. Dataset schemas were inspected locally; the synthetic demo only checks the code path and is not a substitute for research data.
