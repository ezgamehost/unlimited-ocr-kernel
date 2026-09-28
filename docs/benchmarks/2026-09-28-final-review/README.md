# Source-review fixes and verification

The independent read-only review of `5d6923b..2e93b02` identified three
important defects and no critical defects. Root source inspection confirmed
all three. The fixes were made only after the current Q8 ABBA series and raw
audit completed; they cannot have altered its recorded timing.

- Fresh-process benchmark timeouts killed `/usr/bin/time` without killing its
  inference descendants. `run_llama` now creates a private process group,
  terminates it on timeout or interruption, escalates to SIGKILL even if the
  wrapper exits first, and reaps the wrapper. Partial logs and failed-request
  classification are preserved. A real child/grandchild regression failed
  before the fix with a still-running descendant, then passed; it includes
  descendants ignoring SIGTERM and cleans up even on test failure.
- The full model accepted decoder widths incompatible with its fixed-width
  vision projector. Loading now rejects non-1280 hidden widths before opening
  the checkpoint, while generic decoder configurations remain supported.
  Image-feature insertion validates storage, shape, row count and positions
  before modifying embeddings. Tests cover widths 640/2560, malformed feature
  shapes/storage, invalid positions and preservation of surrounding tokens.
- Convolution cloned weights without accounting for the copy in its scratch
  budget. It now borrows flattened storage through the same private linear
  implementation. A thread-local allocation observer caught the original
  32768-byte copy under a 768-byte budget; the corrected operation passes.
  Scalar and Auto checks cover channels, padding, stride, bias and exact
  scratch boundaries. Arithmetic and BLAS dispatch are preserved.

Verification logs record 66 passing standard Rust tests (10 opt-in tests
ignored), a passing Python suite (36 tests, two existing skips), Clippy with
warnings denied, formatting, two explicit original decoder reference checks
and eight vision checks including original fixtures. The CLI was built after
tests and preserved as
`f2433f3b6d50536de9990b3acdaa02e59ae749a9865ecf58d22a8d949ff4bfb8`.
`candidate-cli.json` records production source hashes, and `source.diff`
records tracked changes relative to the preceding benchmark commit.

The final exact-output gate passed: all twelve Q8 and twelve hybrid cases
match preserved responses from binary `c191d896...` in full text, token IDs,
EOS, image-token counts and precision labels. Both processes exited normally
without forced kills. Q8 retains its two one-character table errors (original
and matched prompts); the other ten Q8 cases and all twelve hybrid cases have
zero CER. These cases do not include the previously disclosed max32 hybrid
failure. `quality.py` retains raw outputs, artifact identities and child-specific
resources. No new performance gain is claimed for the convolution allocation
change.

The broader teacher-forced logits/router fixture audit, precision and
fresh-process comparison matrix, and original-upstream max32 attribution
remain open. The hybrid max32 accuracy failure remains disclosed. This
review/fix pass is not project-completion certification.
