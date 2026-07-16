required_packages <- c(
  "data.table",
  "lubridate",
  "Matrix",
  "xgboost",
  "lightgbm",
  "catboost"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  stop(
    paste(
      "Missing required packages:",
      paste(missing_packages, collapse = ", ")
    )
  )
}

suppressPackageStartupMessages({
  library(data.table)
  library(lubridate)
  library(Matrix)
  library(xgboost)
  library(lightgbm)
  library(catboost)
})

set.seed(123)

DATA_DIR <- "."
TRAIN_FILE <- file.path(DATA_DIR, "train.csv")
RESULTS_DIR <- file.path("results", "learner_comparison")

OUTER_VALID_DAYS <- 48L
INNER_FOLDS <- 3L
INNER_VALID_DAYS <- 14L
MIN_INNER_TRAIN_DAYS <- 45L

MAX_ROUNDS <- 2000L
EARLY_STOPPING_ROUNDS <- 100L
LEARNING_RATE <- 0.05
TREE_DEPTH <- 6L
SUBSAMPLE <- 0.80
FEATURE_FRACTION <- 0.80
L1_REG <- 0
L2_REG <- 1
LGB_NUM_LEAVES <- 63L
LGB_MIN_DATA_IN_LEAF <- 20L
XGB_MIN_CHILD_WEIGHT <- 1
CAT_RANDOM_STRENGTH <- 1
CAT_TASK_TYPE <- "CPU"
PRINT_EVERY <- 100L
SEED <- 123L

available_cores <- parallel::detectCores()
if (is.na(available_cores)) available_cores <- 2L
N_THREADS <- max(1L, available_cores - 1L)

dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

safe_parse_datetime <- function(x) {
  if (inherits(x, "POSIXct") || inherits(x, "POSIXt")) {
    return(as.POSIXct(x, tz = "UTC"))
  }

  x_chr <- as.character(x)
  parsed <- suppressWarnings(ymd_hms(x_chr, tz = "UTC", quiet = TRUE))

  bad <- is.na(parsed) & !is.na(x_chr) & x_chr != ""
  if (any(bad)) {
    parsed[bad] <- suppressWarnings(
      ymd_hm(x_chr[bad], tz = "UTC", quiet = TRUE)
    )
  }

  bad <- is.na(parsed) & !is.na(x_chr) & x_chr != ""
  if (any(bad)) {
    parsed[bad] <- suppressWarnings(
      as.POSIXct(x_chr[bad], tz = "UTC")
    )
  }

  parsed
}

rmsle <- function(actual, predicted) {
  actual <- pmax(0, actual)
  predicted <- pmax(0, predicted)
  sqrt(mean((log1p(predicted) - log1p(actual))^2, na.rm = TRUE))
}

add_time_features <- function(data) {
  out <- copy(data)
  out[, date := as.Date(datetime_utc, tz = "UTC")]
  out[, hour := hour(datetime_utc)]
  out[, minute := minute(datetime_utc)]
  out[, weekday := wday(datetime_utc, week_start = 1)]
  out[, is_weekend := as.integer(weekday %in% c(6L, 7L))]
  out[, month := month(datetime_utc)]
  out[, day := day(datetime_utc)]
  out[, yday := yday(datetime_utc)]
  out[, week := isoweek(datetime_utc)]
  out
}

build_c2_profiles <- function(source_data) {
  station_profile <- source_data[
    ,
    .(
      station_hour_minute_mean = mean(bikes, na.rm = TRUE),
      station_hour_minute_median = median(bikes, na.rm = TRUE),
      station_hour_minute_sd = sd(bikes, na.rm = TRUE),
      station_hour_minute_min = min(bikes, na.rm = TRUE),
      station_hour_minute_max = max(bikes, na.rm = TRUE),
      station_hour_minute_n = .N
    ),
    by = .(station_number, hour, minute)
  ]

  station_profile[
    is.na(station_hour_minute_sd),
    station_hour_minute_sd := 0
  ]

  global_profile <- source_data[
    ,
    .(
      global_hour_minute_mean = mean(bikes, na.rm = TRUE),
      global_hour_minute_median = median(bikes, na.rm = TRUE),
      global_hour_minute_sd = sd(bikes, na.rm = TRUE),
      global_hour_minute_min = min(bikes, na.rm = TRUE),
      global_hour_minute_max = max(bikes, na.rm = TRUE),
      global_hour_minute_n = .N
    ),
    by = .(hour, minute)
  ]

  global_profile[
    is.na(global_hour_minute_sd),
    global_hour_minute_sd := 0
  ]

  list(
    station = station_profile,
    global = global_profile
  )
}

add_c2_profiles <- function(target_data, profiles) {
  out <- copy(target_data)
  out[, row_order_internal := .I]

  out <- merge(
    out,
    profiles$station,
    by = c("station_number", "hour", "minute"),
    all.x = TRUE,
    sort = FALSE
  )

  out <- merge(
    out,
    profiles$global,
    by = c("hour", "minute"),
    all.x = TRUE,
    sort = FALSE
  )

  setorder(out, row_order_internal)
  out[, row_order_internal := NULL]
  out
}

apply_shared_imputation <- function(
  train_data,
  valid_data,
  selected_features
) {
  train_out <- copy(train_data)
  valid_out <- copy(valid_data)
  numeric_features <- setdiff(selected_features, "station_number")

  for (feature_name in numeric_features) {
    train_values <- as.numeric(train_out[[feature_name]])
    valid_values <- as.numeric(valid_out[[feature_name]])

    train_values[!is.finite(train_values)] <- NA_real_
    valid_values[!is.finite(valid_values)] <- NA_real_

    fill_value <- median(train_values, na.rm = TRUE)
    if (!is.finite(fill_value)) fill_value <- 0

    train_values[is.na(train_values)] <- fill_value
    valid_values[is.na(valid_values)] <- fill_value

    train_out[, (feature_name) := train_values]
    valid_out[, (feature_name) := valid_values]
  }

  list(
    train = train_out,
    valid = valid_out
  )
}

make_validation_splits <- function(data) {
  max_date <- max(data$date, na.rm = TRUE)
  min_date <- min(data$date, na.rm = TRUE)

  outer_valid_start <- max_date - OUTER_VALID_DAYS + 1L
  outer_train_end <- outer_valid_start - 1L
  development_max_date <- outer_train_end

  inner_rows <- vector("list", INNER_FOLDS)

  for (i in seq_len(INNER_FOLDS)) {
    blocks_after <- INNER_FOLDS - i
    valid_end <- development_max_date - blocks_after * INNER_VALID_DAYS
    valid_start <- valid_end - INNER_VALID_DAYS + 1L
    train_end <- valid_start - 1L

    train_rows <- data[date <= train_end, .N]
    valid_rows <- data[
      date >= valid_start & date <= valid_end,
      .N
    ]
    train_days <- data[date <= train_end, uniqueN(date)]

    if (
      train_rows == 0L ||
      valid_rows == 0L ||
      train_days < MIN_INNER_TRAIN_DAYS
    ) {
      stop("The requested chronological validation folds cannot be created.")
    }

    inner_rows[[i]] <- data.table(
      fold_id = paste0("inner_", i),
      split_type = "inner_development",
      train_start = min_date,
      train_end = train_end,
      valid_start = valid_start,
      valid_end = valid_end,
      train_rows = train_rows,
      valid_rows = valid_rows,
      train_unique_dates = train_days,
      valid_unique_dates = data[
        date >= valid_start & date <= valid_end,
        uniqueN(date)
      ]
    )
  }

  outer_row <- data.table(
    fold_id = "outer_48d",
    split_type = "outer_confirmation",
    train_start = min_date,
    train_end = outer_train_end,
    valid_start = outer_valid_start,
    valid_end = max_date,
    train_rows = data[date <= outer_train_end, .N],
    valid_rows = data[date >= outer_valid_start, .N],
    train_unique_dates = data[date <= outer_train_end, uniqueN(date)],
    valid_unique_dates = data[date >= outer_valid_start, uniqueN(date)]
  )

  rbindlist(c(inner_rows, list(outer_row)), use.names = TRUE)
}

prepare_xgb_matrices <- function(
  train_data,
  valid_data,
  selected_features
) {
  train_frame <- as.data.frame(train_data[, ..selected_features])
  valid_frame <- as.data.frame(valid_data[, ..selected_features])

  station_levels <- sort(unique(as.character(train_frame$station_number)))

  train_frame$station_number <- factor(
    as.character(train_frame$station_number),
    levels = station_levels
  )

  valid_frame$station_number <- factor(
    as.character(valid_frame$station_number),
    levels = station_levels
  )

  if (anyNA(valid_frame$station_number)) {
    stop("Validation contains stations absent from the training period.")
  }

  n_train <- nrow(train_frame)
  combined_frame <- rbind(train_frame, valid_frame)

  sparse_all <- sparse.model.matrix(
    ~ . - 1,
    data = combined_frame,
    na.action = na.pass
  )

  list(
    train = sparse_all[seq_len(n_train), , drop = FALSE],
    valid = sparse_all[
      (n_train + 1L):nrow(sparse_all),
      ,
      drop = FALSE
    ],
    n_columns = ncol(sparse_all)
  )
}

prepare_lgb_matrices <- function(
  train_data,
  valid_data,
  selected_features
) {
  train_frame <- as.data.frame(train_data[, ..selected_features])
  valid_frame <- as.data.frame(valid_data[, ..selected_features])

  station_levels <- sort(unique(as.character(train_frame$station_number)))

  train_station <- factor(
    as.character(train_frame$station_number),
    levels = station_levels
  )

  valid_station <- factor(
    as.character(valid_frame$station_number),
    levels = station_levels
  )

  if (anyNA(valid_station)) {
    stop("Validation contains stations absent from the training period.")
  }

  train_frame$station_number <- as.integer(train_station) - 1L
  valid_frame$station_number <- as.integer(valid_station) - 1L

  train_matrix <- data.matrix(train_frame)
  valid_matrix <- data.matrix(valid_frame)

  colnames(train_matrix) <- selected_features
  colnames(valid_matrix) <- selected_features

  list(
    train = train_matrix,
    valid = valid_matrix,
    categorical_feature = "station_number",
    n_columns = ncol(train_matrix)
  )
}

prepare_catboost_frames <- function(
  train_data,
  valid_data,
  selected_features
) {
  train_frame <- as.data.frame(train_data[, ..selected_features])
  valid_frame <- as.data.frame(valid_data[, ..selected_features])

  station_levels <- sort(unique(as.character(train_frame$station_number)))

  train_frame$station_number <- factor(
    as.character(train_frame$station_number),
    levels = station_levels
  )

  valid_frame$station_number <- factor(
    as.character(valid_frame$station_number),
    levels = station_levels
  )

  if (anyNA(valid_frame$station_number)) {
    stop("Validation contains stations absent from the training period.")
  }

  list(
    train = train_frame,
    valid = valid_frame,
    n_columns = ncol(train_frame)
  )
}

get_xgb_best_round <- function(model) {
  evaluation_log <- tryCatch(
    as.data.table(model$evaluation_log),
    error = function(e) NULL
  )

  if (!is.null(evaluation_log)) {
    valid_columns <- grep(
      "valid.*rmse|eval.*rmse",
      names(evaluation_log),
      ignore.case = TRUE,
      value = TRUE
    )

    if (length(valid_columns) > 0L) {
      return(which.min(evaluation_log[[valid_columns[1L]]]))
    }
  }

  best_iteration <- tryCatch(
    as.integer(model$best_iteration),
    error = function(e) NA_integer_
  )

  if (length(best_iteration) == 1L && is.finite(best_iteration)) {
    return(best_iteration + 1L)
  }

  best_ntreelimit <- tryCatch(
    as.integer(model$best_ntreelimit),
    error = function(e) NA_integer_
  )

  if (length(best_ntreelimit) == 1L && is.finite(best_ntreelimit)) {
    return(best_ntreelimit)
  }

  MAX_ROUNDS
}

predict_xgb_at_best <- function(model, data, best_round) {
  tryCatch(
    predict(
      model,
      data,
      iterationrange = c(1L, best_round + 1L)
    ),
    error = function(e) {
      predict(model, data)
    }
  )
}

get_lgb_best_round <- function(model) {
  candidates <- c(
    suppressWarnings(as.integer(model$best_iter)),
    suppressWarnings(as.integer(model$best_iteration))
  )

  candidates <- candidates[
    is.finite(candidates) & candidates > 0L
  ]

  if (length(candidates) > 0L) {
    return(candidates[1L])
  }

  MAX_ROUNDS
}

get_catboost_best_round <- function(model) {
  tree_count <- tryCatch(
    as.integer(model$tree_count),
    error = function(e) NA_integer_
  )

  if (length(tree_count) != 1L || !is.finite(tree_count)) {
    tree_count <- tryCatch(
      as.integer(attr(model, "tree_count")),
      error = function(e) NA_integer_
    )
  }

  if (
    length(tree_count) == 1L &&
    is.finite(tree_count) &&
    tree_count > 0L
  ) {
    return(tree_count)
  }

  MAX_ROUNDS
}

fit_xgboost <- function(
  train_model,
  valid_model,
  selected_features,
  fold_id
) {
  matrices <- prepare_xgb_matrices(
    train_model,
    valid_model,
    selected_features
  )

  y_train_log <- log1p(train_model$bikes)
  y_valid_log <- log1p(valid_model$bikes)

  dtrain <- xgb.DMatrix(
    data = matrices$train,
    label = y_train_log,
    missing = NA,
    nthread = N_THREADS
  )

  dvalid <- xgb.DMatrix(
    data = matrices$valid,
    label = y_valid_log,
    missing = NA,
    nthread = N_THREADS
  )

  params <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    eta = LEARNING_RATE,
    max_depth = TREE_DEPTH,
    min_child_weight = XGB_MIN_CHILD_WEIGHT,
    subsample = SUBSAMPLE,
    colsample_bytree = FEATURE_FRACTION,
    alpha = L1_REG,
    lambda = L2_REG,
    tree_method = "hist",
    nthread = N_THREADS,
    seed = SEED
  )

  train_start <- Sys.time()

  model <- tryCatch(
    xgb.train(
      params = params,
      data = dtrain,
      nrounds = MAX_ROUNDS,
      evals = list(train = dtrain, valid = dvalid),
      early_stopping_rounds = EARLY_STOPPING_ROUNDS,
      maximize = FALSE,
      print_every_n = PRINT_EVERY,
      verbose = 0
    ),
    error = function(e) {
      xgb.train(
        params = params,
        data = dtrain,
        nrounds = MAX_ROUNDS,
        watchlist = list(train = dtrain, valid = dvalid),
        early_stopping_rounds = EARLY_STOPPING_ROUNDS,
        maximize = FALSE,
        print_every_n = PRINT_EVERY,
        verbose = 0
      )
    }
  )

  train_seconds <- as.numeric(
    difftime(Sys.time(), train_start, units = "secs")
  )

  best_round <- get_xgb_best_round(model)

  predict_start <- Sys.time()
  pred_log <- predict_xgb_at_best(model, dvalid, best_round)
  predict_seconds <- as.numeric(
    difftime(Sys.time(), predict_start, units = "secs")
  )

  pred_bikes <- pmax(0, expm1(pred_log))

  result <- data.table(
    fold_id = fold_id,
    learner = "xgboost",
    valid_rmsle = rmsle(valid_model$bikes, pred_bikes),
    valid_rmse_log = sqrt(
      mean((pred_log - y_valid_log)^2, na.rm = TRUE)
    ),
    best_iteration = best_round,
    train_seconds = train_seconds,
    predict_seconds = predict_seconds,
    total_seconds = train_seconds + predict_seconds,
    n_input_columns = matrices$n_columns,
    categorical_encoding = "one-hot station_number"
  )

  rm(model, matrices, dtrain, dvalid, pred_log, pred_bikes)
  gc(verbose = FALSE)

  result
}

fit_lightgbm <- function(
  train_model,
  valid_model,
  selected_features,
  fold_id
) {
  matrices <- prepare_lgb_matrices(
    train_model,
    valid_model,
    selected_features
  )

  y_train_log <- log1p(train_model$bikes)
  y_valid_log <- log1p(valid_model$bikes)

  train_dataset <- lightgbm::lgb.Dataset(
    data = matrices$train,
    label = y_train_log,
    colnames = selected_features,
    categorical_feature = matrices$categorical_feature,
    free_raw_data = FALSE
  )

  valid_dataset <- lightgbm::lgb.Dataset(
    data = matrices$valid,
    label = y_valid_log,
    reference = train_dataset,
    colnames = selected_features,
    categorical_feature = matrices$categorical_feature,
    free_raw_data = FALSE
  )

  params <- list(
    objective = "regression",
    metric = "rmse",
    learning_rate = LEARNING_RATE,
    max_depth = TREE_DEPTH,
    num_leaves = LGB_NUM_LEAVES,
    min_data_in_leaf = LGB_MIN_DATA_IN_LEAF,
    feature_fraction = FEATURE_FRACTION,
    bagging_fraction = SUBSAMPLE,
    bagging_freq = 1L,
    lambda_l1 = L1_REG,
    lambda_l2 = L2_REG,
    max_bin = 255L,
    first_metric_only = TRUE,
    force_col_wise = TRUE,
    num_threads = N_THREADS,
    seed = SEED,
    feature_fraction_seed = SEED,
    bagging_seed = SEED,
    data_random_seed = SEED,
    verbosity = -1L
  )

  train_start <- Sys.time()

  model <- lightgbm::lgb.train(
    params = params,
    data = train_dataset,
    nrounds = MAX_ROUNDS,
    valids = list(train = train_dataset, valid = valid_dataset),
    early_stopping_rounds = EARLY_STOPPING_ROUNDS,
    eval_freq = PRINT_EVERY,
    verbose = -1
  )

  train_seconds <- as.numeric(
    difftime(Sys.time(), train_start, units = "secs")
  )

  best_round <- get_lgb_best_round(model)

  predict_start <- Sys.time()
  pred_log <- predict(
    model,
    matrices$valid,
    num_iteration = best_round
  )
  predict_seconds <- as.numeric(
    difftime(Sys.time(), predict_start, units = "secs")
  )

  pred_bikes <- pmax(0, expm1(pred_log))

  result <- data.table(
    fold_id = fold_id,
    learner = "lightgbm",
    valid_rmsle = rmsle(valid_model$bikes, pred_bikes),
    valid_rmse_log = sqrt(
      mean((pred_log - y_valid_log)^2, na.rm = TRUE)
    ),
    best_iteration = best_round,
    train_seconds = train_seconds,
    predict_seconds = predict_seconds,
    total_seconds = train_seconds + predict_seconds,
    n_input_columns = matrices$n_columns,
    categorical_encoding = "native categorical station_number"
  )

  rm(
    model,
    matrices,
    train_dataset,
    valid_dataset,
    pred_log,
    pred_bikes
  )
  gc(verbose = FALSE)

  result
}

fit_catboost <- function(
  train_model,
  valid_model,
  selected_features,
  fold_id
) {
  frames <- prepare_catboost_frames(
    train_model,
    valid_model,
    selected_features
  )

  y_train_log <- log1p(train_model$bikes)
  y_valid_log <- log1p(valid_model$bikes)

  train_pool <- catboost::catboost.load_pool(
    data = frames$train,
    label = y_train_log,
    feature_names = as.list(selected_features),
    thread_count = N_THREADS
  )

  valid_pool <- catboost::catboost.load_pool(
    data = frames$valid,
    label = y_valid_log,
    feature_names = as.list(selected_features),
    thread_count = N_THREADS
  )

  params <- list(
    loss_function = "RMSE",
    eval_metric = "RMSE",
    iterations = MAX_ROUNDS,
    learning_rate = LEARNING_RATE,
    depth = TREE_DEPTH,
    l2_leaf_reg = L2_REG,
    random_strength = CAT_RANDOM_STRENGTH,
    bootstrap_type = "Bernoulli",
    subsample = SUBSAMPLE,
    rsm = FEATURE_FRACTION,
    random_seed = SEED,
    early_stopping_rounds = EARLY_STOPPING_ROUNDS,
    use_best_model = TRUE,
    task_type = CAT_TASK_TYPE,
    thread_count = N_THREADS,
    allow_writing_files = FALSE,
    verbose = PRINT_EVERY
  )

  train_start <- Sys.time()

  model <- catboost::catboost.train(
    learn_pool = train_pool,
    test_pool = valid_pool,
    params = params
  )

  train_seconds <- as.numeric(
    difftime(Sys.time(), train_start, units = "secs")
  )

  best_round <- get_catboost_best_round(model)

  predict_start <- Sys.time()
  pred_log <- catboost::catboost.predict(
    model = model,
    pool = valid_pool,
    prediction_type = "RawFormulaVal",
    thread_count = N_THREADS
  )
  predict_seconds <- as.numeric(
    difftime(Sys.time(), predict_start, units = "secs")
  )

  pred_bikes <- pmax(0, expm1(pred_log))

  result <- data.table(
    fold_id = fold_id,
    learner = "catboost",
    valid_rmsle = rmsle(valid_model$bikes, pred_bikes),
    valid_rmse_log = sqrt(
      mean((pred_log - y_valid_log)^2, na.rm = TRUE)
    ),
    best_iteration = best_round,
    train_seconds = train_seconds,
    predict_seconds = predict_seconds,
    total_seconds = train_seconds + predict_seconds,
    n_input_columns = frames$n_columns,
    categorical_encoding = "native factor station_number"
  )

  rm(model, frames, train_pool, valid_pool, pred_log, pred_bikes)
  gc(verbose = FALSE)

  result
}

fit_learner <- function(
  learner,
  train_model,
  valid_model,
  selected_features,
  fold_id
) {
  if (learner == "xgboost") {
    return(
      fit_xgboost(
        train_model,
        valid_model,
        selected_features,
        fold_id
      )
    )
  }

  if (learner == "lightgbm") {
    return(
      fit_lightgbm(
        train_model,
        valid_model,
        selected_features,
        fold_id
      )
    )
  }

  if (learner == "catboost") {
    return(
      fit_catboost(
        train_model,
        valid_model,
        selected_features,
        fold_id
      )
    )
  }

  stop(paste("Unknown learner:", learner))
}

FEATURES_C2 <- c(
  "station_number",
  "lat",
  "lng",
  "hour",
  "minute",
  "weekday",
  "is_weekend",
  "month",
  "day",
  "yday",
  "week",
  "station_hour_minute_mean",
  "station_hour_minute_median",
  "station_hour_minute_sd",
  "station_hour_minute_min",
  "station_hour_minute_max",
  "station_hour_minute_n",
  "global_hour_minute_mean",
  "global_hour_minute_median",
  "global_hour_minute_sd",
  "global_hour_minute_min",
  "global_hour_minute_max",
  "global_hour_minute_n"
)

LEARNERS <- c(
  "xgboost",
  "lightgbm",
  "catboost"
)

if (!file.exists(TRAIN_FILE)) {
  stop("Cannot find train.csv in the project root.")
}

train <- fread(TRAIN_FILE)
setDT(train)

required_columns <- c(
  "datetime",
  "station_number",
  "bikes",
  "lat",
  "lng"
)

missing_columns <- setdiff(required_columns, names(train))
if (length(missing_columns) > 0L) {
  stop(
    paste(
      "Missing required columns:",
      paste(missing_columns, collapse = ", ")
    )
  )
}

train[, row_id_original := .I]
train[, station_number := as.character(station_number)]
train[, datetime_utc := safe_parse_datetime(datetime)]

if (anyNA(train$datetime_utc)) {
  stop("Some datetime values could not be parsed.")
}

if (anyNA(train$bikes) || any(train$bikes < 0)) {
  stop("The bikes target contains invalid values.")
}

train <- add_time_features(train)
setorder(train, datetime_utc, station_number, row_id_original)

validation_splits <- make_validation_splits(train)

fwrite(
  validation_splits,
  file.path(RESULTS_DIR, "validation_splits.csv")
)

fwrite(
  data.table(
    feature_order = seq_along(FEATURES_C2),
    feature = FEATURES_C2,
    n_features = length(FEATURES_C2)
  ),
  file.path(RESULTS_DIR, "selected_23_features.csv")
)

inner_splits <- validation_splits[
  split_type == "inner_development"
]

fold_results <- vector("list", nrow(inner_splits))

for (fold_index in seq_len(nrow(inner_splits))) {
  split_row <- inner_splits[fold_index]

  source_data <- copy(train[date <= split_row$train_end])
  valid_data <- copy(train[
    date >= split_row$valid_start &
      date <= split_row$valid_end
  ])

  profiles <- build_c2_profiles(source_data)

  train_model <- add_c2_profiles(source_data, profiles)
  valid_model <- add_c2_profiles(valid_data, profiles)

  imputed <- apply_shared_imputation(
    train_model,
    valid_model,
    FEATURES_C2
  )

  train_model <- imputed$train
  valid_model <- imputed$valid

  one_fold_results <- vector("list", length(LEARNERS))

  for (learner_index in seq_along(LEARNERS)) {
    learner_name <- LEARNERS[learner_index]

    one_fold_results[[learner_index]] <- fit_learner(
      learner = learner_name,
      train_model = train_model,
      valid_model = valid_model,
      selected_features = FEATURES_C2,
      fold_id = split_row$fold_id
    )
  }

  fold_results[[fold_index]] <- rbindlist(one_fold_results)

  rm(
    source_data,
    valid_data,
    profiles,
    train_model,
    valid_model,
    imputed,
    one_fold_results
  )
  gc(verbose = FALSE)
}

learner_fold_scores <- rbindlist(fold_results)

learner_summary <- learner_fold_scores[
  ,
  .(
    n_features = length(FEATURES_C2),
    mean_rmsle = mean(valid_rmsle),
    sd_rmsle = sd(valid_rmsle),
    worst_fold_rmsle = max(valid_rmsle),
    best_fold_rmsle = min(valid_rmsle),
    mean_best_iteration = mean(best_iteration),
    median_best_iteration = median(best_iteration),
    mean_train_seconds = mean(train_seconds),
    total_train_seconds = sum(train_seconds),
    mean_predict_seconds = mean(predict_seconds),
    mean_total_seconds = mean(total_seconds),
    categorical_encoding = first(categorical_encoding)
  ),
  by = learner
]

setorder(
  learner_summary,
  mean_rmsle,
  sd_rmsle,
  worst_fold_rmsle,
  mean_total_seconds
)

learner_summary[, rank := seq_len(.N)]
learner_summary[
  ,
  delta_vs_best := mean_rmsle - min(mean_rmsle)
]

learner_recommendation <- copy(learner_summary[1])
learner_recommendation[
  ,
  selection_rule :=
    "Lowest mean RMSLE across the three chronological folds"
]

fwrite(
  learner_fold_scores,
  file.path(RESULTS_DIR, "learner_fold_scores.csv")
)

fwrite(
  learner_summary,
  file.path(RESULTS_DIR, "learner_summary.csv")
)

fwrite(
  learner_recommendation,
  file.path(RESULTS_DIR, "learner_recommendation.csv")
)

outer_split <- validation_splits[fold_id == "outer_48d"]

outer_source_data <- copy(train[date <= outer_split$train_end])
outer_valid_data <- copy(train[
  date >= outer_split$valid_start &
    date <= outer_split$valid_end
])

outer_profiles <- build_c2_profiles(outer_source_data)

outer_train_model <- add_c2_profiles(
  outer_source_data,
  outer_profiles
)

outer_valid_model <- add_c2_profiles(
  outer_valid_data,
  outer_profiles
)

outer_imputed <- apply_shared_imputation(
  outer_train_model,
  outer_valid_model,
  FEATURES_C2
)

outer_train_model <- outer_imputed$train
outer_valid_model <- outer_imputed$valid

selected_learner <- learner_recommendation$learner[1]

outer_confirmation <- fit_learner(
  learner = selected_learner,
  train_model = outer_train_model,
  valid_model = outer_valid_model,
  selected_features = FEATURES_C2,
  fold_id = "outer_48d"
)

outer_confirmation[
  ,
  inner_mean_rmsle := learner_recommendation$mean_rmsle[1]
]

outer_confirmation[
  ,
  inner_sd_rmsle := learner_recommendation$sd_rmsle[1]
]

fwrite(
  outer_confirmation,
  file.path(RESULTS_DIR, "outer_confirmation.csv")
)

parameter_summary <- data.table(
  learner = LEARNERS,
  parameters = c(
    paste0(
      "eta=", LEARNING_RATE,
      "; max_depth=", TREE_DEPTH,
      "; min_child_weight=", XGB_MIN_CHILD_WEIGHT,
      "; subsample=", SUBSAMPLE,
      "; colsample_bytree=", FEATURE_FRACTION,
      "; alpha=", L1_REG,
      "; lambda=", L2_REG
    ),
    paste0(
      "learning_rate=", LEARNING_RATE,
      "; max_depth=", TREE_DEPTH,
      "; num_leaves=", LGB_NUM_LEAVES,
      "; min_data_in_leaf=", LGB_MIN_DATA_IN_LEAF,
      "; bagging_fraction=", SUBSAMPLE,
      "; feature_fraction=", FEATURE_FRACTION,
      "; lambda_l1=", L1_REG,
      "; lambda_l2=", L2_REG
    ),
    paste0(
      "learning_rate=", LEARNING_RATE,
      "; depth=", TREE_DEPTH,
      "; bootstrap_type=Bernoulli",
      "; subsample=", SUBSAMPLE,
      "; rsm=", FEATURE_FRACTION,
      "; random_strength=", CAT_RANDOM_STRENGTH,
      "; l2_leaf_reg=", L2_REG
    )
  )
)

fwrite(
  parameter_summary,
  file.path(RESULTS_DIR, "learner_parameters.csv")
)

run_configuration <- data.table(
  setting = c(
    "timezone",
    "target",
    "inverse_transformation",
    "inner_folds",
    "inner_validation_days",
    "outer_validation_days",
    "max_rounds",
    "early_stopping_rounds",
    "seed",
    "threads",
    "post_processing",
    "profile_source"
  ),
  value = c(
    "UTC",
    "log1p(bikes)",
    "pmax(0, expm1(pred_log))",
    as.character(INNER_FOLDS),
    as.character(INNER_VALID_DAYS),
    as.character(OUTER_VALID_DAYS),
    as.character(MAX_ROUNDS),
    as.character(EARLY_STOPPING_ROUNDS),
    as.character(SEED),
    as.character(N_THREADS),
    "none",
    "current fold training period only"
  )
)

fwrite(
  run_configuration,
  file.path(RESULTS_DIR, "run_configuration.csv")
)

print(validation_splits)
print(learner_summary)
print(learner_recommendation)
print(outer_confirmation)
