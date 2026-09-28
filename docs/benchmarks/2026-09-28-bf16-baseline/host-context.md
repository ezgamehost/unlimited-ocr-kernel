# Host observations during the BF16 resident run

This run is on a shared M4 Pro, without host isolation. The raw observations are
in `llama-resident/host-snapshot-midrun.json` and
`llama-resident/host-snapshot-interval.json`.

At 2026-09-28 12:17:30 UTC, host swap usage was 8,899.88 MiB. Over the following
81.480 seconds, host counters increased by 260,085 page-ins, 2,393 page-outs,
6,230 swap-ins and zero swap-outs. The reported page size was 16,384 bytes.
These are host-wide counters, not benchmark-process counters; allocated swap
alone would not establish active swapping, but this interval does show swap-ins.
No thermal/performance warning had been recorded by `pmset -g therm` at the
snapshot. That output is not proof of continuously stable clock frequency.

A process check found only the active benchmark driver (PID 55918) and its
llama.cpp child (PID 55932) among matching inference/benchmark commands. At
roughly 22 minutes elapsed, `ps` reported the server at 797.1% CPU and RSS
6,516,080 KiB. This is instantaneous RSS, not the final exact-child peak collected
by wait4. No other applications were terminated or modified.

Retain all measurements. Do not attribute the BF16-versus-earlier-Q8 timing gap
solely to kernel or precision differences: host paging and noncontemporaneous
runs are material confounders. Report the BF16 distribution as observed on this
host and keep final process resource evidence separate from these host snapshots.
The running protocol has not been restarted or shortened in response to timings.
