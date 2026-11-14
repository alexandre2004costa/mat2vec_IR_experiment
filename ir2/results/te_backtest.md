# Thermoelectric backtest on arXiv cond-mat.mtrl-sci (small-scale reproduction of Fig. 3a)

Candidates: chemical formulas seen ≥ 5 times before T and never in an abstract with thermoelectric/thermoelectrics/thermoelectricity before T. Found: in ≥ 1 abstract with such a keyword from T on (the paper's criterion). P@50 (5 y) = share of the top 50 found within 5 years (T … T+4). word2vec: mean ± std over 3 seeds.

| T | materials | already TE | candidates | base rate (5 y) |
|---|:-:|:-:|:-:|:-:|
| 2010 | 757 | 63 | 694 | 0.075 |
| 2012 | 970 | 92 | 878 | 0.074 |
| 2014 | 1,266 | 150 | 1,116 | 0.070 |
| 2016 | 1,591 | 220 | 1,371 | 0.070 |
| 2018 | 1,991 | 288 | 1,703 | 0.063 |
| 2020 | 2,456 | 405 | 2,051 | 0.051 |

| method | P@20 (5 y) | P@50 (5 y) | × random (P@50, 5 y) | P@50 frequency-matched | P@50 − freq-matched | P@50 (to 2026) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| word2vec output·input (paper) | 0.214 ± 0.005 | 0.220 ± 0.010 | 3.3× | 0.098 ± 0.005 | 0.122 ± 0.006 | 0.343 ± 0.015 |
| word2vec input·input | 0.097 ± 0.005 | 0.082 ± 0.008 | 1.2× | 0.035 ± 0.001 | 0.047 ± 0.009 | 0.168 ± 0.010 |
| co-occurrence with known thermoelectrics | 0.133 | 0.130 | 2.0× | 0.076 | 0.054 | 0.213 |
| frequency | 0.408 | 0.340 | 5.1× | 0.268 | 0.072 | 0.570 |
| random (1000 draws) | 0.067 | 0.067 | 1.0× | — | — | 0.147 |

Per cutoff, P@50 within 5 years (word2vec: mean of 3 seeds):

|    T |   cooc2 |   frequency |   random |   w2v_in |   w2v_out |
|-----:|--------:|------------:|---------:|---------:|----------:|
| 2010 |    0.12 |        0.36 |    0.074 |    0.04  |     0.087 |
| 2012 |    0.14 |        0.44 |    0.072 |    0.127 |     0.2   |
| 2014 |    0.12 |        0.4  |    0.07  |    0.113 |     0.253 |
| 2016 |    0.12 |        0.34 |    0.07  |    0.113 |     0.36  |
| 2018 |    0.14 |        0.28 |    0.063 |    0.053 |     0.193 |
| 2020 |    0.14 |        0.22 |    0.05  |    0.047 |     0.227 |

Top 20 of the paper's scoring at T = 2016 (seed 0): `BCLi` ✗, `NbP` ✓2020, `LuPtSb` ✓2020, `O2Sn` ✓2017, `B2CNi2Y` ✗, `Fe1Ta` ✗, `BiS2` ✓2018, `O3Sn` ✗, `BiFS2Sr` ✗, `AsNb` ✗, `CoFe2O4` ✗, `FeLaOP` ✗, `S3Ti` ✓2024, `MnTe` ✓2020, `As2Cd3` ✓2016, `CuLaOS` ✗, `LaMnO3` ✓2018, `Ge3Mn5` ✓2020, `HfTe5` ✓2017, `AsTa` ✓2022
