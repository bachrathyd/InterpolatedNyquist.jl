"""gpu_final_raw_<date>.txt (lines printed by gpu/scripts/gpu_final.jl on each Colab GPU) ->
gpu_final_summary.csv (full HD) and gpu_final_8k_summary.csv (8K), the inputs of
paper/scripts/studies/s14_gpu_tables.jl."""
import glob, os
here = os.path.dirname(os.path.abspath(__file__))
raw = sorted(glob.glob(os.path.join(here, "gpu_final_raw_*.txt")))[-1]
name = {"Float64  w_max=1e5 (reference)": "Float64 wmax=1e5 (reference)", "Float32  w_max=1e5": "Float32 wmax=1e5",
        "Float32  w_max=1e5  :queue": "Float32 wmax=1e5 queue", "Float64  w_max=15": "Float64 wmax=15",
        "Float32  w_max=15": "Float32 wmax=15", "Float16  w_max=15  scaled": "Float16 wmax=15 scaled",
        "slider loop Float32": "slider loop Float32 wmax=1e5"}
hd, k8 = [], []
for line in open(raw, encoding="utf-8"):
    if line.startswith("#") or not line.strip():
        continue
    f = line.rstrip("\n").split(",")
    dev, case = f[0], f[1]
    if case not in name:
        continue                       # the host-side re-check total is not a device measurement
    is8k = dev.endswith(" 8K")
    dev = dev[:-3] if is8k else dev
    kmed, frame, mpts, evals, wrong, flagged = f[2], f[4], f[5], f[6], f[7], f[8]
    row = ",".join([dev, name[case], kmed, frame, mpts, evals, wrong, flagged])
    (k8 if is8k else hd).append(row)
hdr = "gpu,case,kernel_ms_med,frame_ms,mpts_per_s,evals_med,wrong_pct,flagged_pct"
for fn, rows in (("gpu_final_summary.csv", hd), ("gpu_final_8k_summary.csv", k8)):
    open(os.path.join(here, fn), "w", encoding="utf-8").write(hdr + "\n" + "\n".join(rows) + "\n")
    print(fn, len(rows))
