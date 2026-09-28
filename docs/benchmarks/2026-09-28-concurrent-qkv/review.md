# Independent read-only review

Reviewer `/root/qkv_review` reviewed the production and test diff in `src/decoder/mod.rs` against `39e32ac`. No Critical or Important findings.

The review confirmed checked aggregate scratch and safe partitioning among three workers, sequential fallback for low budgets/fewer workers/multiple rows, reuse of the existing pool, preserved row arithmetic and q/k/v error priority. Concurrent failures can finish other independent projections before returning, without mutating decoder state.

Minor coverage limitation: the direct projection-equivalence test uses original BF16 weights with Backend::Auto. Q8 and Q4 receive full-runtime logit/OCR coverage, but direct Scalar and generic F16/F32 projection fixtures remain follow-up work. Generic F16/F32 may call Accelerate concurrently; their performance is unmeasured and the documented Rust-pool limit is not a global vendor-thread cap. No concrete correctness defect was found.

The reviewer ran no builds, tests or benchmarks. Validation evidence is recorded separately.
