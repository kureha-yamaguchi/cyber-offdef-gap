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
