# Changelog

## v0.1.1 — 2026-10-04

- Bundle the upstream Turbo workflow and test every built-in profile offline.
- Restore reference-outfit extraction compatibility and plugin-wide Danbooru
  concurrency limits without changing request-local configuration snapshots.
- Keep submitted-but-unconfirmed jobs distinct from definitive failures; add
  caller-scoped history checks and explicit retrieval of never-sent old results.
  Never resample or resend an uncertain delivery.
- Persist private actual workflow graphs, sampling parameters and stage timings.
  Failed auxiliary notifications no longer abort generation.
- Reserve independent status capacity, restore redacted diagnostics, and label
  custom-workflow readiness as deferred rather than incorrectly unavailable.
- Validate canvas/queue-related values, hide ineffective reasoning controls, move
  historical reads/cleanup off the event loop, and scan retention in bounded batches.
- Add Windows/Linux Python 3.12/3.13 offline CI and regression coverage. No new
  image editing, artist/character permission restrictions or automatic retries.

Verification: 190 offline tests, Ruff lint/format, Python compilation and diff
checks. Runtime configuration schema remains 0.1.0 for existing installations.

## v0.1.0 — 2026-10-03

First independent MaiBot release, based on Anima Master `main @ 4f6a0f1`.

- Explicit `/anm`, `/anima`, and `/comfyui` commands; ordinary and structured
  multi-person text-to-image generation using ComfyUI.
- Upstream prompt expansion, artists, named characters, Danbooru correction,
  tag cleaning, white-background defaults, and Turbo behavior retained.
- Explicit current-message/quoted-image input, generation metadata extraction,
  multimodal reverse prompting, and reference-description-assisted generation.
- Persistent artist/character chat presets and request-local atomic task records,
  message deduplication, and bounded retention.
- Image-send RPC waits up to 180 seconds without automatic retries. A documented
  maintenance tool inspects/applies the shared NapCat/SnowLuma timeout settings.

Verification: 147 offline tests, Ruff lint/format, Python compilation and diff
checks. Authorized live tests covered ordinary/two-person generation, metadata,
vision, presets, missing-input isolation and QQ receipts. The latest fresh image
send took approximately 12.7 seconds; uploads exceeding one minute remain untested.

Not included: pixel-level image editing, post-generation visual validation,
candidate regeneration, or guarantees of 3–4 person/custom-workflow image quality.
