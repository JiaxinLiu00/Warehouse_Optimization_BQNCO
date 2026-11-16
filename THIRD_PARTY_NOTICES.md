# Third-party notices

## BQ-NCO

The following files contain BQ-NCO code attributed to NAVER:

- `bqnco_framework/__init__.py`
- `bqnco_framework/encoder.py`
- `bqnco_framework/attention.py`

Copyright (c) 2023-present NAVER Corp.

Licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The original notices are retained in the source files.

Local changes in `attention.py` include uneven attention head dimensions,
optional entmax loading and bias-free initialization.

The project-specific policy in `bqnco_framework/role_policy.py` extends these modules.
`MaskedEncoderLayer` adds a padding mask to `EncoderLayer`.
