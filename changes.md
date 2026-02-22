# Code clean up
- create package structure
- split unified functions into modules
  - plotting functions moved to `utils.plots`
  - mskcc nomogram prediction and helper functions moved to `ml.mskcc_nomogram`
  - discrimination_threshold moved to `utils.discrimination_threshold`
  - training functions moved to `ml.training`
  - smote functions moved to `ml.smote`
- organize training results returning
  -  add TrainingResult dataclass
- added type hints
- refactored notebooks to new functions and modules
  - ISUP_upgrade_analysis.ipynb
  - epe_unified.ipynb

TODO:
- refactor remaining notebooks
- clean up notebooks 
- delete unnecessary notebooks -> archive
- resolve TODOs in code