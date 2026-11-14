# Thermoelectric backtest on arXiv cond-mat.mtrl-sci (small-scale reproduction of Fig. 3a), setup `paper`: 200 dimensions, cutoffs 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021

Candidates: chemical formulas seen ≥ 5 times before T and never in an abstract with thermoelectric/thermoelectrics/thermoelectricity before T. Found: in ≥ 1 abstract with such a keyword from T on (the paper's criterion). P@50 (5 y) = share of the top 50 found within 5 years (T … T+4). word2vec: mean ± std over 3 seeds.

| T | materials | already TE | candidates | base rate (5 y) |
|---|:-:|:-:|:-:|:-:|
| 2010 | 757 | 63 | 694 | 0.075 |
| 2011 | 852 | 80 | 772 | 0.071 |
| 2012 | 970 | 92 | 878 | 0.074 |
| 2013 | 1,112 | 111 | 1,001 | 0.072 |
| 2014 | 1,266 | 150 | 1,116 | 0.070 |
| 2015 | 1,399 | 187 | 1,212 | 0.065 |
| 2016 | 1,591 | 220 | 1,371 | 0.070 |
| 2017 | 1,795 | 254 | 1,541 | 0.070 |
| 2018 | 1,991 | 288 | 1,703 | 0.063 |
| 2019 | 2,200 | 347 | 1,853 | 0.061 |
| 2020 | 2,456 | 405 | 2,051 | 0.051 |
| 2021 | 2,755 | 463 | 2,292 | 0.050 |

| method | P@20 (5 y) | P@50 (5 y) | × random (P@50, 5 y) | P@50 frequency-matched | P@50 − freq-matched | P@50 (to 2026) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| word2vec output·input (paper) | 0.249 ± 0.009 | 0.228 ± 0.002 | 3.5× | 0.108 ± 0.002 | 0.120 ± 0.004 | 0.358 ± 0.003 |
| word2vec input·input | 0.099 ± 0.013 | 0.093 ± 0.000 | 1.4× | 0.033 ± 0.000 | 0.060 ± 0.000 | 0.166 ± 0.003 |
| co-occurrence with known thermoelectrics | 0.146 | 0.125 | 1.9× | 0.073 | 0.052 | 0.203 |
| frequency | 0.404 | 0.335 | 5.1× | 0.262 | 0.073 | 0.547 |
| random (1000 draws) | 0.066 | 0.066 | 1.0× | — | — | 0.140 |

Per cutoff, P@50 within 5 years (word2vec: mean of 3 seeds):

|    T |   cooc2 |   frequency |   random |   w2v_in |   w2v_out |
|-----:|--------:|------------:|---------:|---------:|----------:|
| 2010 |    0.12 |        0.36 |    0.074 |    0.047 |     0.147 |
| 2011 |    0.14 |        0.4  |    0.071 |    0.067 |     0.14  |
| 2012 |    0.14 |        0.44 |    0.074 |    0.14  |     0.2   |
| 2013 |    0.1  |        0.42 |    0.071 |    0.133 |     0.207 |
| 2014 |    0.12 |        0.4  |    0.071 |    0.113 |     0.247 |
| 2015 |    0.06 |        0.34 |    0.064 |    0.107 |     0.3   |
| 2016 |    0.12 |        0.34 |    0.07  |    0.133 |     0.367 |
| 2017 |    0.16 |        0.3  |    0.069 |    0.08  |     0.227 |
| 2018 |    0.14 |        0.28 |    0.065 |    0.08  |     0.227 |
| 2019 |    0.14 |        0.26 |    0.06  |    0.08  |     0.2   |
| 2020 |    0.14 |        0.22 |    0.05  |    0.067 |     0.22  |
| 2021 |    0.12 |        0.26 |    0.05  |    0.073 |     0.253 |

Top 20 of the paper's scoring at T = 2019 (seed 0): `MoP` ✗, `OSn` ✓2020, `TeZr` ✗, `BiFS2Sr` ✗, `BaO3Sn` ✓2023, `NbP` ✓2020, `CuI` ✓2022, `NiS2` ✗, `Ga2O3` ✓2025, `S2Sn` ✓2021, `BCLi` ✗, `Fe2P` ✗, `CMgNi3` ✗, `FeSb3` ✓2025, `FeLaOP` ✗, `FeTe` ✗, `BaBi2Mn2` ✗, `AsB` ✓2019, `Bi2Pt` ✗, `O3Sn` ✗
