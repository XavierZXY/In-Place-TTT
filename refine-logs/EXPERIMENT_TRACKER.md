# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | 复现 outer 路径 | ttt_write_rule=outer (现状ckpt) | RULER 16k 全任务 | overall score | MUST | DONE | ✅ 0.2854,逐任务diff=+0.000,复现完美 |
| R002 | M0 | η=0 退化检查 | NLMS, η=0 | RULER 16k | overall | MUST | DONE | ✅ 0.4583,与tttlr0基线完全一致 |
| R003 | M1 | drop-in 预判 | NLMS-1024 (no train) | NIAH multikey2/3+multiquery+vt 16k | trim score, read entropy | MUST | TODO | faithful forward |
| R004 | M1 | drop-in 预判 | NLMS-256 (no train) | 同上 + 4k/8k 护栏 | trim score | MUST | TODO | 先验最优 granularity |
| R005 | M1 | 负对照 | keynorm (no train) | 同上 | trim score | MUST | TODO | 不应优于 NLMS |
| R006 | M2 | 容量定律 | offline probe 3指标 | RULER 16k 11检索任务 per-task | r=-0.726/+0.751 | MUST | DONE | ✅ C2成立:collision↔失败强相关 |
| R007 | M3 | 主基线 CPT | outer + CPT | RULER 子集 + 护栏 | RULER score | MUST | TODO | 冻 backbone,stage3 yaml,seed×3 |
| R008 | M3 | 主方法 CPT | NLMS-256 + CPT | 同 R007 | RULER score | MUST | TODO | 同 tokens/opt/步数 |
| R009 | M3 | 负对照 CPT | keynorm + CPT | 同 R007 | RULER score | MUST | TODO | matched 预算 |
| R010 | M4 | granularity sweep | NLMS subchunk∈{1024,512,256,128} | RULER 子集 | score vs granularity | NICE | TODO | Block1 正信号后 |
| R011 | M4 | modernization | diagonal-RLS variant | RULER 子集 | score | NICE | TODO | 仅 scalar NLMS 正后 |

## 状态图例
TODO / RUNNING / DONE / FAIL / BLOCKED

## 关键 gate
- R001 不过 → 全部 BLOCKED(先修递推实现)
- R003-R005 大幅掉 → 不否定,查 train/infer mismatch,仍进 M2/M3
- R006 全 AUC≈0.5 → C2 不成立,削弱 C1 动机,考虑转 value/objective(idea ⑦)
- R007 vs R008 = 论文主判定
