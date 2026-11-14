# Leakage check: main run vs all-versions run

Main run: pre-T text = papers whose first and latest version are before T. All-versions run: first version before T only, so abstracts revised in or after T are included.

| T | pre-T docs (all versions) | removed by the filter | share |
|---|--:|--:|--:|
| 2016 | 6,042 | 298 | 4.9% |
| 2017 | 8,125 | 443 | 5.5% |
| 2018 | 11,089 | 576 | 5.2% |
| 2019 | 15,655 | 862 | 5.5% |
| 2020 | 22,179 | 1,310 | 5.9% |
| 2021 | 30,682 | 1,848 | 6.0% |
| 2022 | 40,285 | 2,089 | 5.2% |
| 2023 | 50,910 | 2,424 | 4.8% |

### Candidates `all`: mean P@k over the 28 pairs scored in both runs

| method | P@10 main | P@10 all versions | P@20 main | P@20 all versions | P@50 main | P@50 all versions |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| word2vec output·input (paper) | 0.252 ± 0.011 | 0.240 ± 0.011 | 0.277 ± 0.010 | 0.262 ± 0.009 | 0.279 ± 0.006 | 0.269 ± 0.001 |
| word2vec input·input | 0.094 ± 0.005 | 0.092 ± 0.009 | 0.094 ± 0.011 | 0.102 ± 0.012 | 0.104 ± 0.003 | 0.102 ± 0.003 |
| SPECTER2 abstract means | 0.529 | 0.496 | 0.491 | 0.479 | 0.467 | 0.442 |
| 2nd-order co-occurrence | 0.800 | 0.800 | 0.791 | 0.777 | 0.760 | 0.747 |
| frequency | 0.457 | 0.454 | 0.475 | 0.452 | 0.513 | 0.504 |
| random (1000 draws) | 0.138 | 0.129 | 0.137 | 0.130 | 0.137 | 0.130 |

### Candidates `phrases`: mean P@k over the 28 pairs scored in both runs

| method | P@10 main | P@10 all versions | P@20 main | P@20 all versions | P@50 main | P@50 all versions |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| word2vec output·input (paper) | 0.298 ± 0.027 | 0.283 ± 0.022 | 0.280 ± 0.019 | 0.253 ± 0.009 | 0.254 ± 0.002 | 0.239 ± 0.005 |
| word2vec input·input | 0.180 ± 0.005 | 0.173 ± 0.008 | 0.158 ± 0.006 | 0.152 ± 0.002 | 0.155 ± 0.003 | 0.142 ± 0.005 |
| SPECTER2 abstract means | 0.479 | 0.479 | 0.429 | 0.427 | 0.362 | 0.349 |
| 2nd-order co-occurrence | 0.596 | 0.575 | 0.561 | 0.530 | 0.475 | 0.454 |
| frequency | 0.368 | 0.375 | 0.337 | 0.330 | 0.281 | 0.271 |
| random (1000 draws) | 0.103 | 0.098 | 0.102 | 0.099 | 0.102 | 0.098 |