# cyber-offdef-gap

Does refusal-ablation degrade a model's **defensive** cyber (DFIR) capability while
sparing its **offensive** capability — and if defence drops, is it **sandbagging** or
**capability loss**?

One model pair (`openai/gpt-oss-120b` vs `wangzhang/gpt-oss-120b-abliterated`) is run
on two offensive benchmarks (CyberGym, ExploitBench) and two defensive benchmarks
(AuditBench, ExCyTIn-Bench), with the serving stack, scaffold, prompts, and sampling
held identical across the pair. See **[PLAN.md](PLAN.md)**.

Status: plan only; nothing has been run.
