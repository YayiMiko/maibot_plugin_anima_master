# MaiBot Anima port

This is an independent repository. Do not edit or commit AstrBot core, the
original Anima plugin, or Qwen when working here. Upstream baseline is Anima
Master main at 4f6a0f1, including the post-0.9.3 white-background rollback.

The implementation includes text/multi-person planning and explicit image
references, metadata extraction and vision reverse prompting. Keep the
artist/character chat library in plugin-owned atomic storage, not Host internals.
Chat overrides are shared, persist across reloads and never alter queued snapshots.
Preserve upstream prompt, artist, fixed-character, tag cleaning, Danbooru and
workflow contracts. Do not add implicit generation tools, change the host persona, or
enable unsupported editing routes silently. Preserve original image metadata.
Never resolve global latest images or cross-conversation quotes. Do not expose secrets,
addresses, file paths or other users' records in chat replies.

Use public MaiBot SDK APIs only; do not import host src internals. Keep tasks,
configuration snapshots, output paths and delivery records request-local.
Do not auto-resubmit jobs, resend uncertain deliveries, or globally interrupt
ComfyUI. Keep runtime configs and generated images out of Git.
Retention may delete only expired terminal task records and their validated
task-ID directories. Never follow links/junctions, delete active tasks or use
stored output paths as deletion targets. Preserve presets and orphan data.

All verification must remain offline until the user authorizes live testing.
Do not contact ComfyUI, send QQ tests, deploy or restart services. Tests block
live HTTP access and mock model calls, processes and delivery.

Run Ruff, pytest, compile checks and git diff --check for changed code. Parent
workspace discovery may fail due to unrelated conflicts; use an isolated venv
without modifying that parent. Keep comments and logs in English.
