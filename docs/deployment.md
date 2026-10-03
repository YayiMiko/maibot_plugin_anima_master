# Deployment and send timeout maintenance

This plugin runs on the MaiBot host. ComfyUI may run on another machine; configure
a reachable ComfyUI address without changing models or prompt presets.
Use MaiBot 1.3.1, Plugin SDK 2.8.2, and Python 3.12+ for the verified baseline.

## Install or update

1. Back up the installed plugin and its runtime configuration outside `plugins/`.
2. Install the plugin source and Python dependencies from `pyproject.toml`.
3. Preserve `config.toml`, plugin-owned `chat_presets.json`, and task data.
4. Check the shared send timeout layers below, especially after adapter updates.
5. Reload MaiBot during an idle period. Never restart an active generation job
   automatically or resubmit a task after a reload.

No deployment requires changing the original AstrBot Anima repository, the Qwen
plugin, account permissions, or the bot persona. Do not copy credentials into Git.

## Coordinated QQ send timeouts

The tested deployment has four separate waiting layers:

| Layer | Setting | Value |
| --- | --- | --- |
| NapCat OneBot | `timeout.baseTimeout` | 90000 ms plus size-based estimate |
| SnowLuma transport | `client.action_timeout_sec` | 120 seconds |
| SnowLuma gateway | `@MessageGateway(timeout_ms=...)` | 150000 ms |
| Anima SDK image send | `ctx.send.image(timeout_ms=...)` | 180000 ms |

NapCat estimates additional time from image bytes and `uploadSpeedKBps`; for very
large files its estimate may exceed the outer transport budget. These values are
not an unlimited upload guarantee. Keep the other NapCat timeout fields unchanged.

Only the Anima SDK timeout ships in this plugin. The other three settings belong
to shared infrastructure: do not mistake them for per-plugin settings. They also
affect other QQ sends. An adapter upgrade can overwrite the gateway decorator.

Use the maintenance tool to inspect the mounted configuration files. It performs
no API requests, sends no messages, and does not restart services:

```text
python tools/configure_send_timeouts.py --adapter-dir /path/to/MaiBot-SnowLuma-Adapter --napcat-config /path/to/onebot11_ACCOUNT.json
```

To apply the verified values, explicitly supply `--apply`. Changed files are backed
up before replacement; the tool prints only filenames and timeout settings, never
tokens or the full configuration. It refuses unfamiliar gateway source or higher
existing timeout values rather than silently lowering a deployment's limits.

```text
python tools/configure_send_timeouts.py --adapter-dir /path/to/MaiBot-SnowLuma-Adapter --napcat-config /path/to/onebot11_ACCOUNT.json --apply --backup-dir /path/to/backups
```

Review the changes before reloading MaiBot and NapCat as needed. A NapCat restart
may invalidate the saved QQ session: be prepared to scan the login QR again. An
adapter change must be reloaded for its gateway metadata to take effect. Schedule
the reload while idle; do not assume editing the files changed the running values.

## Acceptance and recovery

Check `/anm 状态`, then send one new image request in an authorized test group.
Record the QQ receipt and image-send duration separately from generation time.
Repeated images can reuse uploaded resources and are not a fresh-upload test.
Do not automatically resend an uncertain delivery or regenerate its image.

`/anm 核对任务 [TASK_ID]` reads the saved prompt history/queue using the original
private connection snapshot. It does not submit or send an image.
`/anm 恢复任务 [TASK_ID]` explicitly retrieves completed results only when no image
delivery was previously attempted. Both commands are restricted to the caller's
original conversation and user scope. Omit the ID to use the latest generation.
Unknown deliveries are never resent; missing prompt IDs or snapshots require
manual investigation. A cleared ComfyUI history is unknown, not proof of failure.

Unsubmitted work interrupted by a reload is `interrupted`. Submitted work with an
unconfirmed remote outcome is `remote_unknown`; cancellation during sending is
`delivery_unknown`. Status checks do not overwrite the latest-generation index.
Back up the private task/runtime directories if reproducibility or recovery is
important. Generated workflow graphs, config snapshots and prompts are private
runtime data, not release assets. Unknown remote jobs are not retention targets.

On 2026-10-03 a fresh image was generated and sent once with a QQ message ID;
image sending took approximately 12.7 seconds. Uploads exceeding one minute are
still unverified. A longer timeout allows waiting; it does not accelerate QQ.

Temporary acceptance plugins must have no public commands, use separate plugin
data, and send only to the authorized test group. Archive them after testing and
verify that only production extensions remain loaded. Restore saved files to
roll back a deployment; never delete runtime records, presets, or shared QQ cache.
