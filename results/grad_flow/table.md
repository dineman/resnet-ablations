| network | activation std, first → last block | ‖∂L/∂W‖, first → last block | ratio first/last grad | logit std | initial loss |
|---|---|---|---|---|---|
| resnet20 | 0.85 → 1.5 | 0.35 → 0.24 | 1.5 | 0.83 | 2.63 |
| plain20 | 0.6 → 0.6 | 0.69 → 0.13 | 5.2 | 0.25 | 2.35 |
| resnet56 | 0.84 → 2.7 | 1.2 → 0.32 | 3.6 | 1.8 | 3.75 |
| plain56 | 0.58 → 0.66 | 2.5e+02 → 0.19 | 1.3e+03 | 0.35 | 2.36 |
| resnet20_nobn | 0.58 → 5.3 | 6.6 → 17 | 0.39 | 2.2 | 4.65 |
| plain20_nobn | 0.39 → 0.12 | 0.021 → 0.066 | 0.31 | 0.077 | 2.31 |
