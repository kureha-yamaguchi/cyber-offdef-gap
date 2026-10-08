# M0 runbook — infra & smoke (aligned model)

Goal: stand up vLLM serving `openai/gpt-oss-120b`, verify harmony + reasoning-effort
parity, and smoke the four benchmark harnesses (AuditBench end-to-end; the three
Docker harnesses: clone/install + resolve the docker-host topology).

## Provisioned infra (this session)
- Pod: `cyber-offdef-vllm-m0` id **7vefomehzknwgj** — 1x H100 80GB SXM (CUDA 13 host), EUR-NO-2,
  SECURE, $3.49/hr, container disk 200GB, ports 8000/http + 22/tcp.
- SSH (proxy): `ssh 7vefomehzknwgj-64411fa1@ssh.runpod.io`
- vLLM OpenAI endpoint (public proxy once serving): https://7vefomehzknwgj-8000.proxy.runpod.net/v1
- Budget: separate $800 cap (approved 2026-10-05). REMEMBER TO STOP/TERMINATE when idle.

## Not mine — do not touch
- Pod `vuln-patch-runner` (bv0v0bw6k48gno) + volume vuln-patch-workspace: belongs to the
  separate vuln-patch project sessions.
- Volume auditbench-uplift-durable (zues5s7zwd, EU-FR-1, 300GB): the auditbench-uplift project.

## M1 run (AuditBench pair) — this session (2026-10-06)
- Old aligned pod 7vefomehzknwgj: EXITED, host out of GPUs, could not resume. Left stopped.
- New single node serves BOTH models sequentially (parity: identical HW + vLLM, weights-only delta):
  Pod `cyber-offdef-vllm-pair` id **a2wfij577308nv** — 2x H200 SXM 141GB (TP=2), SECURE, $9.18/hr,
  container disk 600GB, machine kxchllmtoe2w.
- SSH: `ssh -i ~/.ssh/id_ed25519 -p 18546 root@103.196.86.37`  (my local ed25519 key injected via PUBLIC_KEY)
- Serving is PRIVATE (PLAN §8): vLLM on 0.0.0.0:8000 inside pod, reached via local SSH tunnel
  `ssh -L 8000:localhost:8000 ...`; harness base_url = http://localhost:8000/v1 for BOTH sides.
- Serve flags (identical both models): TP=2, --max-model-len 32768, --gpu-memory-utilization 0.92, --async-scheduling.
- REMEMBER: stop/terminate a2wfij577308nv when done.
