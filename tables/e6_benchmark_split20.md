# E6 -- real-stream benchmark (imbalanced Split-CIFAR100, all 10 tasks)

| method | norm | n seeds | corrected avg acc | backward transfer | worst-task acc | forgetting variance | mean wall (s) | mean peak mem (MB) | mean skipped/total steps |
|---|---|---|---|---|---|---|---|---|---|
| v2 | group | 1 | 0.5789 +/- 0.0000 | +0.0436 | 0.4450 | 0.01852 | 11446.3 | 525 | 0.0% |

Not implemented in this configuration: the oracle arm and the equal 10x10-split control stream.
