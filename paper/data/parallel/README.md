# Parallel-scaling measurements (Section "Parallel execution", Fig. parallel, Tables parallel and gpu)

The paper scripts only re-plot these stored measurements:
`paper/scripts/studies/s15_parallel_scaling.py` (Fig. `fig_parallel_scaling`, `tab_parallel.tex`,
`generated/parallel_numbers.tex`) and `paper/scripts/studies/s14_gpu_tables.jl` (`tab_gpu.tex`,
`generated/gpu_numbers.tex`, from `gpu/results/gpu_final_*`).

| file | what | measured with | hardware |
|---|---|---|---|
| `thread_scaling_local.csv` | CPU threads 1..12, 400x400 charts, three systems, Float64 | `gpu/scripts/thread_scaling.jl` (KernelAbstractions CPU backend) | Intel i5-10400 (6 cores, 12 threads), Windows |
| `thread_scaling_colab.csv` | CPU threads 1..48 and full GPU charts | `gpu/scripts/thread_scaling.jl` | Colab G4 instance (48 vCPU host, RTX PRO 6000 Blackwell Server Edition) |
| `lane_scaling_<GPU>_<F32/F64>.csv` | exactly n GPU threads (work-queue schedule, workgroup min(256, n)), chart side clamp(ceil(sqrt(64 n)), 64, 2048), best of 3 runs (1 run above 2 s) | `gpu/scripts/gpu_lane_scaling.jl` | Colab T4, L4, A100-SXM4-40GB, RTX PRO 6000 (G4) |
| `gpu/results/gpu_final_raw_2026-10-06.txt` | full HD and 8K, one thread per point, median of 10 / 5 | `gpu/scripts/gpu_final.jl` | same four Colab GPUs |

All GPU runs (2026-10-05/06) used the NyquistGPU engine of branch `hill-argument-principle`
(commit 0edf90f; generic engine parts identical to `gpu-cuda` db622a8) through the Colab notebooks
`gpu/colab/NyquistGPU_Lane_Scaling.ipynb` and `gpu/colab/NyquistGPU_Final.ipynb` of that branch.
GPU clocks were not locked; the T4 (70 W, passive) and the L4 (72 W) are clock- and power-limited,
which shows as scatter between runs.
