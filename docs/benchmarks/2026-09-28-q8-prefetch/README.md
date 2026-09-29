# Bounded Q8 prefetch experiment — interrupted

A test-only candidate adds `PRFM PLDL1KEEP` for four row-major weight streams,
256 bytes ahead every four blocks. A strict block bound keeps each hinted address
inside its row. Layout, output arithmetic and storage remain unchanged. The
missing helper first fails (`red.log`), then exact tests pass for aligned and
unaligned inputs, multiple widths, signed-zero scales and the prefetch threshold.

The AC comparison is **invalid/incomplete**: baseline finishes on AC, but the
first candidate process begins on AC and its post-phase check finds battery
power. The runner exits 1 at that check; both inference children have already
exited normally with code 0 and no forced kills. All eight completed repetitions
have identical full-logit hashes, which supplies correctness evidence only.
Neither the mixed-power candidate timings nor the incomplete order supplies an
accepted performance comparison. No speedup/regression claim is made.

The prototype and evidence are preserved; production source and the accepted
CLI are unchanged. `experiment.patch` restores the prototype against accepted
runtime `5ce97a5`. `run.py --output-name FRESH` requires a fresh AC-powered series
before any comparison can be considered. The duplicate original kernel is kept
in this prototype for its direct equivalence test, not proposed as final code.
