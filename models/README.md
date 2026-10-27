# models/

`wetland_cover_models.npz` holds the seven random forests (1000 trees each)
from "Final binomial model no coordinates unbalanced models 05 Dec 2025.rda",
exported with R's getTree() (scripts/export_trees.R) and packed by
scripts/build_model_npz.py. Predictors, in order: wetness, brightness,
greenness, mean_TCW, mean_TCB, mean_TCG, Q10W, Q90W, Q10B, Q90B, Q10G, Q90G.
