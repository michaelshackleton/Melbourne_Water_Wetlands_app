# Exports the seven binomial cover models' trees so the Python app can use
# them without R. Only needed if the models are retrained - the app already
# ships with models/wetland_cover_models.npz built from
# "Final binomial model no coordinates unbalanced models 05 Dec 2025.rda".
#
# 1. Edit MODEL_PATH / OUT_DIR below and run this script in R
#    (needs the randomForest package).
# 2. Then, from the app folder with its Python environment active:
#      python scripts/build_model_npz.py "<OUT_DIR>"
#    which writes models/wetland_cover_models.npz.
# 3. If the thresholds changed too, update config/thresholds.csv.
#
# getTree() is randomForest's own documented way of reading a tree back out,
# so the Python side never has to guess at randomForest's internal storage.
# The script also writes r_validation.csv (R's own predictions for the
# training data) so the port can be checked - see README "Model check".

library(randomForest)

MODEL_PATH <- "D:/Wetland inundation modelling/Final binomial model no coordinates unbalanced models 05 Dec 2025.rda"
OUT_DIR    <- "exported_trees"
MODELS     <- c("bare_40", "dryVeg_20", "dryVegBare_50", "greenVeg_40", "openwater_50", "veg_30", "water_60")
PREDICTORS <- c("wetness", "brightness", "greenness", "mean_TCW", "mean_TCB", "mean_TCG",
                "Q10W", "Q90W", "Q10B", "Q90B", "Q10G", "Q90G")

m <- readRDS(MODEL_PATH)      # saved with saveRDS() despite the .rda extension
dir.create(OUT_DIR, showWarnings = FALSE)

for (k in MODELS) {
  fm <- m[[k]]$finalModel     # caret train object -> randomForest
  stopifnot(inherits(fm, "randomForest"))
  if (!identical(sort(rownames(fm$importance)), sort(PREDICTORS)))
    stop(k, ": predictors differ from the expected 12 - update PREDICTORS here and in build_model_npz.py")
  out <- vector("list", fm$ntree)
  for (t in seq_len(fm$ntree)) {
    d <- getTree(fm, k = t, labelVar = TRUE)
    d$tree <- t
    d$node <- seq_len(nrow(d))
    out[[t]] <- d
  }
  write.csv(do.call(rbind, out), file.path(OUT_DIR, paste0(k, ".csv")), row.names = FALSE)
  cat("Exported", k, "(", fm$ntree, "trees )\n")
}

X <- m[[MODELS[1]]]$trainingData[, PREDICTORS]
P <- sapply(MODELS, function(k) predict(m[[k]], X, type = "prob")[, "Presence"])
write.csv(cbind(X, P), file.path(OUT_DIR, "r_validation.csv"), row.names = FALSE)
cat("Wrote", file.path(OUT_DIR, "r_validation.csv"), "\n")
