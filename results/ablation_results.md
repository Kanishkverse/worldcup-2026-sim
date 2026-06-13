# Ablation study — random vs LLM agents

|                   |   runs |   distinct |   entropy_bits | top1          |   top5_share_pct |
|:------------------|-------:|-----------:|---------------:|:--------------|-----------------:|
| LLM agents        |    100 |         25 |           4.3  | Spain 12%     |               45 |
| Random agents     |    100 |         24 |           4.16 | Spain 13%     |               50 |
| Default decisions |    100 |         27 |           4.48 | Argentina 11% |               37 |

- goals/match: random 2.57, defaults 2.58, LLM 2.69 (real World Cups 2.64)
- TV distance LLM↔random 0.280, LLM↔defaults 0.330, random↔defaults 0.290
- Spearman rho (title odds): LLM↔random 0.64 (p=0.000); LLM↔defaults 0.53 (p=0.002)
- chi-square LLM vs random champion tables: p=0.277
- Split-half baseline: TV between two 50-run halves of the SAME arm is 0.36 (random) and 0.42 (defaults) — larger than the 0.28 between LLM and random arms. Cross-arm differences are entirely within sampling noise at n=100.

## Verdict

The flat champion field is **sampling noise, not emergent tactics**. With agent
brains replaced by coin-flips (same action space), the championship distribution
is statistically indistinguishable from the LLM runs (chi-square p=0.28; 24 vs 25
distinct champions; entropy 4.16 vs 4.30 bits; Spain on top in both at 13%/12%).
Even zero-variance default decisions yield 27 distinct champions — the spread
comes from the calibrated match engine itself (Poisson goal process + knockout
single-elimination), not from who is making the decisions. LLM steering changes
match *texture* (who scores, which chances die, substitution stories), not
championship *outcomes*.
