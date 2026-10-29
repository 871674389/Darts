# Tab.4: point-adjust F1 可作弊(自证)—— C3 评测口径证据

> 数据来自主对比表(A100 docs/master_comparison.csv)。展示 threshold-free AUROC 与 point-adjust F1 的**排名反转**:AUROC 最差的方法反而 PA-F1 最高。

## MSL 上的排名反转(最典型)
| 方法 | threshold-free AUROC | PA-F1 | 反转 |
|---|---|---|---|
| **Darts** | **0.666(第1)** | **0.192(倒1)** | AUROC最好→PA-F1最差 |
| DCdetector | 0.496(倒2) | 0.850(第4) | AUROC近最差→PA-F1近最好 |
| USAD | 0.535 | **0.893(并列第1)** | AUROC中下→PA-F1最高 |
| DAGMM | 0.571 | **0.893(并列第1)** | 同上 |
| OmniAnomaly | 0.518 | 0.882 | 同上 |

## 机制
point-adjust:一个异常段内只要命中1点,整段判对。→ 只要方法在每段随机撒够高分点就能刷高 PA-F1,与真实排序能力无关。故 threshold-free 最差(近随机)的方法 PA-F1 反而饱和到最高。
## 结论
**支撑以 threshold-free AUROC/AUPR 为主口径、point-adjust 仅作参考**。WADI 上更极端:多方法 PA-F1 并列 0.744(段调整饱和到 flag-all)。
