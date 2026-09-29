# Independent read-only review

Reviewer `/root/route_fixture_review` found capture/comparison/alignment sound.
An initial Minor finding noted that deriving coverage from mutable configuration
could let the tests become vacuous with only one layer. Both tests now assert
12 layers, one initial dense layer, six selected experts, 64 routed experts,
window 128, ten heads and hidden width 1280. They also verify head-major caches,
absolute position progression and bounded active counts through wraparound.
The reviewer confirmed the finding is addressed, with no remaining findings.

The capture occurs on the calling pool worker; synchronous routing precedes
expert parallelism, and nested decoder steps stay in the same pool. RAII resets
the thread-local capture after success or unwinding. Layer-major prefill and
per-layer decode captures align with the fixture. Canonical sorting preserves
ID/weight associations and does not compare arbitrary upstream top-k order.

The numerical routing threshold was selected before these runs; it is not a
calibrated whole-model parity threshold. No builds, tests or inference were run
by the reviewer. All reported execution evidence comes from the parent's logs.
