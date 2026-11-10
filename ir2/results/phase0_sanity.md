# Phase 0: sanity check with the pretrained mat2vec embeddings

Model: `pretrained_embeddings` from storage.googleapis.com/mat2vec, loaded with gensim (vocabulary 529,686 tokens, dimension 200). Formulae are normalized as in mat2vec (elements sorted alphabetically) before lookup.

## 1. Nearest neighbours of LiCoO2 (normalized `CoLiO2`)

| rank | token | cosine |
|---|---|---|
| 1 | `LiMn2O4` | 0.837 |
| 2 | `LiNi` | 0.806 |
| 3 | `Li2Mn3NiO8` | 0.796 |
| 4 | `CoLi5Ni4O10` | 0.787 |
| 5 | `AlCo3Li20Ni16O40` | 0.781 |
| 6 | `3Co1_/_3Mn1_/_3O2` | 0.780 |
| 7 | `3Mn1_/_3Co1_/_3O2` | 0.777 |
| 8 | `LiNiO2` | 0.758 |
| 9 | `Co2Li10Mn3Ni5O20` | 0.750 |
| 10 | `Li2MnNiO4` | 0.748 |
| 11 | `CoLi` | 0.747 |
| 12 | `FeLiO4P` | 0.731 |
| 13 | `LNMO` | 0.729 |
| 14 | `lithium_manganese_oxide` | 0.728 |
| 15 | `Li4O12Ti5` | 0.720 |

## 2. Analogy: ferromagnetic − NiFe + IrMn (normalized `FeNi`, `IrMn`)

| rank | token | cosine |
|---|---|---|
| 1 | `antiferromagnetic` | 0.737 |
| 2 | `anti-ferromagnetic` | 0.736 |
| 3 | `FM` | 0.715 |
| 4 | `ferromagnet` | 0.702 |
| 5 | `ferromagnetic_ordering` | 0.695 |
| 6 | `canted` | 0.685 |
| 7 | `ferrimagnetic` | 0.685 |
| 8 | `spontaneous_magnetization` | 0.673 |
| 9 | `antiferromagnetically_coupled` | 0.668 |
| 10 | `antiferromagentic` | 0.665 |

`antiferromagnetic` rank in top 10: 1.

## 3. Materials ranked for `thermoelectric`

Candidates: every vocabulary token that parses as a simple formula of ≥2 distinct elements (a regex stand-in for mat2vec's pymatgen check). Two scores are shown: input·input cosine (`most_similar`), and the paper's score, the normalized **output** embedding (`syn1neg`) of the material dotted with the normalized **input** embedding of `thermoelectric`.

Caveat: the paper also removes materials that already co-occur with thermoelectric keywords in the corpus. The corpus is not distributed, so this filter cannot be applied here: this list mixes known and predicted thermoelectrics.

110,057 formula candidates in the vocabulary.

| rank | input·input | cos | output·input (paper) | score |
|---|---|---|---|---|
| 1 | `Bi2Te3` | 0.748 | `Bi2Te3` | 0.193 |
| 2 | `AgMgSb` | 0.700 | `AgMgSb` | 0.138 |
| 3 | `Ag97Mg100Sb99` | 0.694 | `Ca9Mn10NdO30` | 0.132 |
| 4 | `Bi20Se3Te27` | 0.688 | `PbTe` | 0.128 |
| 5 | `BiSb3Te6` | 0.686 | `Pb2SeTe` | 0.126 |
| 6 | `AgPbSbTe` | 0.677 | `Co12Fe4InSb48` | 0.125 |
| 7 | `SSe4Sn5` | 0.673 | `Bi3Sb17Te30` | 0.123 |
| 8 | `Bi10Se3Te12` | 0.672 | `CoSb3` | 0.121 |
| 9 | `BiSb19Te30` | 0.672 | `Bi2Sb8Te15` | 0.119 |
| 10 | `Bi20Se27Te3` | 0.672 | `CeCoFe3Sb12` | 0.118 |
| 11 | `Bi19LuTe30` | 0.670 | `BiSb3Te6` | 0.114 |
| 12 | `Ge2PbSnTe4` | 0.665 | `Ca50Mn49NbO150` | 0.112 |
| 13 | `Bi2Sb8Te15` | 0.664 | `Bi12Sb38Te75` | 0.110 |
| 14 | `Mo150Ni3Sb270Te80` | 0.659 | `Bi20Se3Te27` | 0.110 |
| 15 | `AgKTe` | 0.659 | `Bi10Se3Te12` | 0.108 |
| 16 | `Cu3SbSe4` | 0.658 | `FeNbSb` | 0.107 |
| 17 | `Co8La7O24Sr` | 0.655 | `BiSbTe` | 0.104 |
| 18 | `Bi19Te30Tm` | 0.655 | `Sb3Zn4` | 0.104 |
| 19 | `Bi2S2Se` | 0.653 | `CoSSb` | 0.102 |
| 20 | `Ca50Mn49NbO150` | 0.652 | `Ge2PbSnTe4` | 0.101 |

Rank (output·input) of a few textbook thermoelectrics: Bi2Te3 → 1, PbTe → 4, SnSe → 74, Sb2Te3 → 21, CoSb3 → 8, Mg2Si → 138.
