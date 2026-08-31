| Problem | Path length | Decision space | Cost | Plateau-heavy | All perms valid paths? | Reorder alone? |
|---|---|---|---|---|---|---|
| MDKP | fixed (n) | subset (index scan) | sum, max profit | no | n/a | no — no ordering |
| TSPTW | fixed (n) | permutation | sum | no | no (time windows) | yes |
| CVRP | fixed (n) | permutation + vehicle breaks | sum | no | no (capacity) | NO — breaks are a real choice |
| m-PDTSP | fixed (n) | linear ext. of P[j] | sum | no | poset + load/edges | yes |
| OPTW | fixed (n) | subset + sequence | sum, max profit | order-neutral | no (time windows) | no — surrogate only |
| SALBP-1 | variable (n+V) | linear ext. of P[j] | count V | yes | poset only | yes |
| MOSP | fixed (n) | permutation | max | yes | YES | yes |
| Bin Packing | fixed (n) | permutation | count k | yes | no — symmetry breaking | yes |
| Graph-Clear | fixed (n) | permutation | max | yes | YES | yes |
| Talent Sched. | fixed (n) | permutation | sum | no | no — dominance | yes |
| 1‖Σw·T + prec. | fixed (n) | linear ext. of P[j] | sum | no | poset only | yes |