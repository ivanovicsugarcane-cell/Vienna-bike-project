required_packages <- c(
  "data.table",
  "lubridate",
  "xgboost",
  "Matrix"
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
  library(xgboost)
  library(Matrix)
})

set.seed(123)

DATA_DIR <- "."
TRAIN_FILE <- file.path(DATA_DIR, "train.csv")
RESULTS_DIR <- file.path("results", "hpo_config")

TEST_MODE <- FALSE
RESUME <- TRUE
SEED <- 123L

OUTER_VALID_DAYS <- 48L
INNER_FOLDS <- 3L
INNER_VALID_DAYS <- 14L
MIN_INNER_TRAIN_DAYS <- 45L

N_RANDOM_CONFIGS <- 60L
N_LOCAL_CONFIGS <- 20L
N_LOCAL_CENTERS <- 3L
SUCCESSIVE_HALVING_ETA <- 3L

MAX_NROUNDS <- 5000L
EARLY_STOPPING_ROUNDS <- 100L
PRINT_EVERY_N <- 200L

ETA_LOWER <- 0.015
ETA_UPPER <- 0.150
MAX_DEPTH_LOWER <- 3L
MAX_DEPTH_UPPER <- 9L
MIN_CHILD_WEIGHT_LOWER <- 1
MIN_CHILD_WEIGHT_UPPER <- 40
SUBSAMPLE_LOWER <- 0.60
SUBSAMPLE_UPPER <- 1.00
COLSAMPLE_LOWER <- 0.60
COLSAMPLE_UPPER <- 1.00
GAMMA_LOWER_NONZERO <- 0.001
GAMMA_UPPER <- 5
LAMBDA_LOWER <- 0.05
LAMBDA_UPPER <- 30
ALPHA_LOWER_NONZERO <- 0.0001
ALPHA_UPPER <- 10

available_cores <- parallel::detectCores()
if (is.na(available_cores)) available_cores <- 2L
N_THREADS <- max(1L, available_cores - 1L)

if (TEST_MODE) {
  N_RANDOM_CONFIGS <- 6L
  N_LOCAL_CONFIGS <- 3L
  MAX_NROUNDS <- 400L
  EARLY_STOPPING_ROUNDS <- 30L
  PRINT_EVERY_N <- 100L
}

dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

FINAL_SET <- c(
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

if (length(FINAL_SET) != 23L) {
  stop("FINAL_SET must contain exactly 23 features.")
}

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
  sqrt(
    mean(
      (log1p(predicted) - log1p(actual))^2,
      na.rm = TRUE
    )
  )
}

log_uniform <- function(n, lower, upper) {
  exp(runif(n, min = log(lower), max = log(upper)))
}

clamp <- function(x, lower, upper) {
  pmin(upper, pmax(lower, x))
}

config_signature <- function(
  eta,
  max_depth,
  min_child_weight,
  subsample,
  colsample_bytree,
  gamma,
  lambda,
  alpha
) {
  paste(
    sprintf("%.8f", eta),
    as.integer(max_depth),
    sprintf("%.8f", min_child_weight),
    sprintf("%.8f", subsample),
    sprintf("%.8f", colsample_bytree),
    sprintf("%.8f", gamma),
    sprintf("%.8f", lambda),
    sprintf("%.8f", alpha),
    sep = "|"
  )
}

config_row_to_params <- function(config_row) {
  list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    booster = "gbtree",
    tree_method = "hist",
    eta = as.numeric(config_row$eta[1]),
    max_depth = as.integer(config_row$max_depth[1]),
    min_child_weight = as.numeric(config_row$min_child_weight[1]),
    subsample = as.numeric(config_row$subsample[1]),
    colsample_bytree = as.numeric(config_row$colsample_bytree[1]),
    gamma = as.numeric(config_row$gamma[1]),
    lambda = as.numeric(config_row$lambda[1]),
    alpha = as.numeric(config_row$alpha[1]),
    nthread = N_THREADS,
    seed = SEED
  )
}

get_best_nrounds <- function(model) {
  evaluation_log <- tryCatch(
    as.data.table(model$evaluation_log),
    error = function(e) NULL
  )

  if (!is.null(evaluation_log)) {
    valid_columns <- grep(
      "valid.*rmse",
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

  MAX_NROUNDS
}

predict_xgb_at_best <- function(model, dmatrix, best_round) {
  tryCatch(
    predict(
      model,
      dmatrix,
      iterationrange = c(0L, best_round)
    ),
    error = function(e) {
      tryCatch(
        predict(
          model,
          dmatrix,
          iterationrange = c(1L, best_round + 1L)
        ),
        error = function(e2) {
          predict(model, dmatrix)
        }
      )
    }
  )
}

add_profile_features <- function(source_data, target_data) {
  source_data <- copy(source_data)
  target_data <- copy(target_data)
  target_data[, row_order_internal := .I]

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

  out <- merge(
    target_data,
    station_profile,
    by = c("station_number", "hour", "minute"),
    all.x = TRUE,
    sort = FALSE
  )

  out <- merge(
    out,
    global_profile,
    by = c("hour", "minute"),
    all.x = TRUE,
    sort = FALSE
  )

  overall_mean <- mean(source_data$bikes, na.rm = TRUE)
  overall_median <- median(source_data$bikes, na.rm = TRUE)
  overall_sd <- sd(source_data$bikes, na.rm = TRUE)
  overall_min <- min(source_data$bikes, na.rm = TRUE)
  overall_max <- max(source_data$bikes, na.rm = TRUE)

  if (!is.finite(overall_sd)) overall_sd <- 0

  global_defaults <- c(
    global_hour_minute_mean = overall_mean,
    global_hour_minute_median = overall_median,
    global_hour_minute_sd = overall_sd,
    global_hour_minute_min = overall_min,
    global_hour_minute_max = overall_max,
    global_hour_minute_n = 0
  )

  for (feature_name in names(global_defaults)) {
    missing_rows <- which(is.na(out[[feature_name]]))

    if (length(missing_rows) > 0L) {
      set(
        out,
        i = missing_rows,
        j = feature_name,
        value = global_defaults[[feature_name]]
      )
    }
  }

  station_fallbacks <- list(
    station_hour_minute_mean = "global_hour_minute_mean",
    station_hour_minute_median = "global_hour_minute_median",
    station_hour_minute_sd = "global_hour_minute_sd",
    station_hour_minute_min = "global_hour_minute_min",
    station_hour_minute_max = "global_hour_minute_max",
    station_hour_minute_n = "global_hour_minute_n"
  )

  for (feature_name in names(station_fallbacks)) {
    fallback_name <- station_fallbacks[[feature_name]]
    missing_rows <- which(is.na(out[[feature_name]]))

    if (length(missing_rows) > 0L) {
      set(
        out,
        i = missing_rows,
        j = feature_name,
        value = out[[fallback_name]][missing_rows]
      )
    }
  }

  setorder(out, row_order_internal)
  out[, row_order_internal := NULL]

  out
}

prepare_model_matrices <- function(
  train_data,
  valid_data,
  selected_features
) {
  train_frame <- as.data.frame(train_data[, ..selected_features])
  valid_frame <- as.data.frame(valid_data[, ..selected_features])
  n_train <- nrow(train_frame)

  combined_frame <- rbind(train_frame, valid_frame)

  for (feature_name in names(combined_frame)) {
    if (
      is.character(combined_frame[[feature_name]]) ||
      is.factor(combined_frame[[feature_name]])
    ) {
      combined_frame[[feature_name]] <- as.character(
        combined_frame[[feature_name]]
      )

      combined_frame[[feature_name]][
        is.na(combined_frame[[feature_name]])
      ] <- "__MISSING__"

      combined_frame[[feature_name]] <- as.factor(
        combined_frame[[feature_name]]
      )
    }

    if (is.logical(combined_frame[[feature_name]])) {
      combined_frame[[feature_name]] <- as.integer(
        combined_frame[[feature_name]]
      )
    }

    if (
      is.numeric(combined_frame[[feature_name]]) ||
      is.integer(combined_frame[[feature_name]])
    ) {
      train_values <- combined_frame[[feature_name]][seq_len(n_train)]
      fill_value <- median(train_values, na.rm = TRUE)

      if (!is.finite(fill_value)) fill_value <- 0

      combined_frame[[feature_name]][
        is.na(combined_frame[[feature_name]])
      ] <- fill_value
    }
  }

  sparse_all <- sparse.model.matrix(
    ~ . - 1,
    data = combined_frame
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

baseline_config <- data.table(
  config_id = "baseline",
  config_source = "baseline",
  center_id = NA_character_,
  eta = 0.03,
  max_depth = 6L,
  min_child_weight = 10,
  subsample = 0.85,
  colsample_bytree = 0.85,
  gamma = 0,
  lambda = 1,
  alpha = 0
)

reference_config <- data.table(
  config_id = "reference_config",
  config_source = "previous_submission_reference",
  center_id = NA_character_,
  eta = 0.03591504,
  max_depth = 6L,
  min_child_weight = 3.99688001,
  subsample = 0.87071714,
  colsample_bytree = 0.65131130,
  gamma = 0.01718908,
  lambda = 0.31737196,
  alpha = 0.00620258
)

sample_random_configs <- function(n) {
  result <- vector("list", n)

  for (i in seq_len(n)) {
    gamma_value <- if (runif(1) < 0.30) {
      0
    } else {
      log_uniform(1, GAMMA_LOWER_NONZERO, GAMMA_UPPER)
    }

    alpha_value <- if (runif(1) < 0.35) {
      0
    } else {
      log_uniform(1, ALPHA_LOWER_NONZERO, ALPHA_UPPER)
    }

    result[[i]] <- data.table(
      config_id = sprintf("random_%03d", i),
      config_source = "random_search",
      center_id = NA_character_,
      eta = log_uniform(1, ETA_LOWER, ETA_UPPER),
      max_depth = sample(
        MAX_DEPTH_LOWER:MAX_DEPTH_UPPER,
        size = 1L
      ),
      min_child_weight = log_uniform(
        1,
        MIN_CHILD_WEIGHT_LOWER,
        MIN_CHILD_WEIGHT_UPPER
      ),
      subsample = runif(
        1,
        SUBSAMPLE_LOWER,
        SUBSAMPLE_UPPER
      ),
      colsample_bytree = runif(
        1,
        COLSAMPLE_LOWER,
        COLSAMPLE_UPPER
      ),
      gamma = gamma_value,
      lambda = log_uniform(
        1,
        LAMBDA_LOWER,
        LAMBDA_UPPER
      ),
      alpha = alpha_value
    )
  }

  rbindlist(result)
}

add_signatures <- function(configs) {
  configs <- copy(configs)

  configs[
    ,
    signature := mapply(
      config_signature,
      eta,
      max_depth,
      min_child_weight,
      subsample,
      colsample_bytree,
      gamma,
      lambda,
      alpha,
      USE.NAMES = FALSE
    )
  ]

  configs
}

build_fold_cache <- function(data, split_row) {
  fold_id <- split_row$fold_id[1]
  train_end <- as.Date(split_row$train_end[1])
  valid_start <- as.Date(split_row$valid_start[1])
  valid_end <- as.Date(split_row$valid_end[1])

  source_data <- copy(data[date <= train_end])
  valid_data <- copy(
    data[date >= valid_start & date <= valid_end]
  )

  train_model <- add_profile_features(
    source_data,
    source_data
  )

  valid_model <- add_profile_features(
    source_data,
    valid_data
  )

  matrices <- prepare_model_matrices(
    train_model,
    valid_model,
    FINAL_SET
  )

  dtrain <- xgb.DMatrix(
    data = matrices$train,
    label = log1p(train_model$bikes),
    missing = NA
  )

  dvalid <- xgb.DMatrix(
    data = matrices$valid,
    label = log1p(valid_model$bikes),
    missing = NA
  )

  list(
    fold_id = fold_id,
    dtrain = dtrain,
    dvalid = dvalid,
    actual_valid = valid_model$bikes,
    y_valid_log = log1p(valid_model$bikes),
    train_rows = nrow(train_model),
    valid_rows = nrow(valid_model),
    n_columns = matrices$n_columns
  )
}

fit_one_config <- function(config_row, fold_cache) {
  params <- config_row_to_params(config_row)
  start_time <- Sys.time()

  model <- tryCatch(
    xgb.train(
      params = params,
      data = fold_cache$dtrain,
      nrounds = MAX_NROUNDS,
      evals = list(
        train = fold_cache$dtrain,
        valid = fold_cache$dvalid
      ),
      early_stopping_rounds = EARLY_STOPPING_ROUNDS,
      print_every_n = PRINT_EVERY_N,
      verbose = 0
    ),
    error = function(e) {
      xgb.train(
        params = params,
        data = fold_cache$dtrain,
        nrounds = MAX_NROUNDS,
        watchlist = list(
          train = fold_cache$dtrain,
          valid = fold_cache$dvalid
        ),
        early_stopping_rounds = EARLY_STOPPING_ROUNDS,
        print_every_n = PRINT_EVERY_N,
        verbose = 0
      )
    }
  )

  best_round <- get_best_nrounds(model)
  pred_log <- predict_xgb_at_best(
    model,
    fold_cache$dvalid,
    best_round
  )

  pred_bikes <- pmax(0, expm1(pred_log))

  result <- list(
    valid_rmsle = rmsle(
      fold_cache$actual_valid,
      pred_bikes
    ),
    valid_rmse_log = sqrt(
      mean(
        (pred_log - fold_cache$y_valid_log)^2,
        na.rm = TRUE
      )
    ),
    best_nrounds = best_round,
    elapsed_seconds = as.numeric(
      difftime(
        Sys.time(),
        start_time,
        units = "secs"
      )
    )
  )

  rm(model, pred_log, pred_bikes)
  gc(verbose = FALSE)

  result
}

FOLD_RESULTS_FILE <- file.path(
  RESULTS_DIR,
  "hpo_fold_scores_checkpoint.csv"
)

if (RESUME && file.exists(FOLD_RESULTS_FILE)) {
  fold_results <- fread(FOLD_RESULTS_FILE)
  fold_results <- unique(
    fold_results,
    by = c("config_id", "fold_id"),
    fromLast = TRUE
  )
} else {
  fold_results <- data.table()
}

run_config_batch_on_fold <- function(
  configs,
  split_row,
  phase,
  current_results
) {
  configs <- unique(copy(configs), by = "config_id")
  fold_id_value <- as.character(split_row$fold_id[1])

  completed_ids <- character(0)

  if (nrow(current_results) > 0L) {
    completed_ids <- current_results[
      fold_id == fold_id_value &
        status == "ok",
      unique(config_id)
    ]
  }

  configs_to_run <- configs[
    !config_id %in% completed_ids
  ]

  if (nrow(configs_to_run) == 0L) {
    return(current_results)
  }

  fold_cache <- build_fold_cache(train, split_row)

  for (i in seq_len(nrow(configs_to_run))) {
    config_row <- configs_to_run[i]

    fit_result <- tryCatch(
      {
        fitted <- fit_one_config(
          config_row,
          fold_cache
        )

        data.table(
          config_id = config_row$config_id[1],
          config_source = config_row$config_source[1],
          center_id = config_row$center_id[1],
          fold_id = fold_id_value,
          phase = phase,
          status = "ok",
          error_message = NA_character_,
          valid_rmsle = fitted$valid_rmsle,
          valid_rmse_log = fitted$valid_rmse_log,
          best_nrounds = fitted$best_nrounds,
          elapsed_seconds = fitted$elapsed_seconds,
          train_rows = fold_cache$train_rows,
          valid_rows = fold_cache$valid_rows,
          n_columns = fold_cache$n_columns,
          eta = config_row$eta[1],
          max_depth = config_row$max_depth[1],
          min_child_weight = config_row$min_child_weight[1],
          subsample = config_row$subsample[1],
          colsample_bytree = config_row$colsample_bytree[1],
          gamma = config_row$gamma[1],
          lambda = config_row$lambda[1],
          alpha = config_row$alpha[1]
        )
      },
      error = function(e) {
        data.table(
          config_id = config_row$config_id[1],
          config_source = config_row$config_source[1],
          center_id = config_row$center_id[1],
          fold_id = fold_id_value,
          phase = phase,
          status = "error",
          error_message = conditionMessage(e),
          valid_rmsle = NA_real_,
          valid_rmse_log = NA_real_,
          best_nrounds = NA_integer_,
          elapsed_seconds = NA_real_,
          train_rows = fold_cache$train_rows,
          valid_rows = fold_cache$valid_rows,
          n_columns = fold_cache$n_columns,
          eta = config_row$eta[1],
          max_depth = config_row$max_depth[1],
          min_child_weight = config_row$min_child_weight[1],
          subsample = config_row$subsample[1],
          colsample_bytree = config_row$colsample_bytree[1],
          gamma = config_row$gamma[1],
          lambda = config_row$lambda[1],
          alpha = config_row$alpha[1]
        )
      }
    )

    if (nrow(current_results) > 0L) {
      current_results <- current_results[
        !(
          config_id == config_row$config_id[1] &
          fold_id == fold_id_value
        )
      ]
    }

    current_results <- rbindlist(
      list(current_results, fit_result),
      use.names = TRUE,
      fill = TRUE
    )

    setorder(current_results, fold_id, config_id)
    fwrite(current_results, FOLD_RESULTS_FILE)
  }

  rm(fold_cache)
  gc(verbose = FALSE)

  current_results
}

rank_configurations <- function(
  configs,
  required_fold_ids,
  current_results
) {
  configs <- unique(copy(configs), by = "config_id")

  score_rows <- current_results[
    status == "ok" &
      config_id %in% configs$config_id &
      fold_id %in% required_fold_ids
  ]

  summary <- score_rows[
    ,
    .(
      n_folds = uniqueN(fold_id),
      mean_rmsle = mean(valid_rmsle),
      sd_rmsle = if (.N > 1L) sd(valid_rmsle) else 0,
      worst_fold_rmsle = max(valid_rmsle),
      best_fold_rmsle = min(valid_rmsle),
      median_best_nrounds = median(best_nrounds, na.rm = TRUE),
      mean_elapsed_seconds = mean(elapsed_seconds, na.rm = TRUE)
    ),
    by = config_id
  ]

  summary <- summary[
    n_folds == length(required_fold_ids)
  ]

  ranking <- merge(
    configs,
    summary,
    by = "config_id",
    all = FALSE
  )

  setorder(
    ranking,
    mean_rmsle,
    sd_rmsle,
    worst_fold_rmsle,
    config_id
  )

  ranking[, rank := seq_len(.N)]
  ranking
}

select_survivors <- function(ranking) {
  if (nrow(ranking) == 0L) {
    stop("No successful configurations are available.")
  }

  keep_n <- max(
    3L,
    ceiling(nrow(ranking) / SUCCESSIVE_HALVING_ETA)
  )

  ranking[seq_len(min(keep_n, nrow(ranking)))]
}

generate_local_configs <- function(
  center_rows,
  n_configs,
  existing_configs
) {
  center_rows <- copy(center_rows)
  n_centers <- min(
    N_LOCAL_CENTERS,
    nrow(center_rows)
  )

  center_rows <- center_rows[seq_len(n_centers)]
  seen_signatures <- unique(existing_configs$signature)

  generated <- list()
  generated_n <- 0L
  attempts <- 0L
  max_attempts <- max(1000L, n_configs * 200L)

  while (
    generated_n < n_configs &&
      attempts < max_attempts
  ) {
    attempts <- attempts + 1L
    center <- center_rows[
      ((attempts - 1L) %% n_centers) + 1L
    ]

    eta_value <- clamp(
      center$eta[1] *
        exp(runif(1, log(0.60), log(1.50))),
      ETA_LOWER,
      ETA_UPPER
    )

    depth_value <- as.integer(
      clamp(
        center$max_depth[1] +
          sample(-1L:1L, 1L),
        MAX_DEPTH_LOWER,
        MAX_DEPTH_UPPER
      )
    )

    child_value <- clamp(
      center$min_child_weight[1] *
        exp(runif(1, log(0.50), log(2.00))),
      MIN_CHILD_WEIGHT_LOWER,
      MIN_CHILD_WEIGHT_UPPER
    )

    subsample_value <- clamp(
      center$subsample[1] +
        runif(1, -0.10, 0.10),
      SUBSAMPLE_LOWER,
      SUBSAMPLE_UPPER
    )

    colsample_value <- clamp(
      center$colsample_bytree[1] +
        runif(1, -0.10, 0.10),
      COLSAMPLE_LOWER,
      COLSAMPLE_UPPER
    )

    gamma_value <- if (runif(1) < 0.25) {
      0
    } else if (center$gamma[1] <= 0) {
      log_uniform(1, GAMMA_LOWER_NONZERO, 1)
    } else {
      clamp(
        center$gamma[1] *
          exp(runif(1, log(0.30), log(3.00))),
        GAMMA_LOWER_NONZERO,
        GAMMA_UPPER
      )
    }

    lambda_value <- clamp(
      center$lambda[1] *
        exp(runif(1, log(0.30), log(3.00))),
      LAMBDA_LOWER,
      LAMBDA_UPPER
    )

    alpha_value <- if (runif(1) < 0.30) {
      0
    } else if (center$alpha[1] <= 0) {
      log_uniform(1, ALPHA_LOWER_NONZERO, 1)
    } else {
      clamp(
        center$alpha[1] *
          exp(runif(1, log(0.30), log(3.00))),
        ALPHA_LOWER_NONZERO,
        ALPHA_UPPER
      )
    }

    signature_value <- config_signature(
      eta_value,
      depth_value,
      child_value,
      subsample_value,
      colsample_value,
      gamma_value,
      lambda_value,
      alpha_value
    )

    if (!signature_value %in% seen_signatures) {
      generated_n <- generated_n + 1L

      generated[[generated_n]] <- data.table(
        config_id = sprintf("local_%03d", generated_n),
        config_source = "local_refinement",
        center_id = center$config_id[1],
        eta = eta_value,
        max_depth = depth_value,
        min_child_weight = child_value,
        subsample = subsample_value,
        colsample_bytree = colsample_value,
        gamma = gamma_value,
        lambda = lambda_value,
        alpha = alpha_value,
        signature = signature_value
      )

      seen_signatures <- c(
        seen_signatures,
        signature_value
      )
    }
  }

  if (generated_n < n_configs) {
    stop("Could not generate enough unique local configurations.")
  }

  rbindlist(generated)
}

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

train[, date := as.Date(datetime_utc, tz = "UTC")]
train[, hour := hour(datetime_utc)]
train[, minute := minute(datetime_utc)]
train[, weekday := wday(datetime_utc, week_start = 1)]
train[, is_weekend := as.integer(weekday %in% c(6L, 7L))]
train[, month := month(datetime_utc)]
train[, day := day(datetime_utc)]
train[, yday := yday(datetime_utc)]
train[, week := isoweek(datetime_utc)]

setorder(train, datetime_utc, station_number, row_id_original)

validation_splits <- make_validation_splits(train)

fwrite(
  validation_splits,
  file.path(RESULTS_DIR, "validation_splits.csv")
)

fwrite(
  data.table(
    feature_order = seq_along(FINAL_SET),
    feature = FINAL_SET
  ),
  file.path(RESULTS_DIR, "selected_23_features.csv")
)

inner_splits <- validation_splits[
  split_type == "inner_development"
]

outer_split <- validation_splits[
  split_type == "outer_confirmation"
]

random_configs <- sample_random_configs(
  N_RANDOM_CONFIGS
)

initial_configs <- add_signatures(
  rbindlist(
    list(
      baseline_config,
      reference_config,
      random_configs
    ),
    use.names = TRUE,
    fill = TRUE
  )
)

initial_configs <- unique(
  initial_configs,
  by = "signature"
)

fwrite(
  initial_configs,
  file.path(RESULTS_DIR, "initial_candidates.csv")
)

for (fold_id_value in inner_splits$fold_id) {
  fold_results <- run_config_batch_on_fold(
    baseline_config,
    inner_splits[fold_id == fold_id_value],
    "baseline",
    fold_results
  )
}

stage1_fold_ids <- tail(inner_splits$fold_id, 1L)

for (fold_id_value in stage1_fold_ids) {
  fold_results <- run_config_batch_on_fold(
    initial_configs,
    inner_splits[fold_id == fold_id_value],
    "successive_halving_stage1",
    fold_results
  )
}

stage1_ranking <- rank_configurations(
  initial_configs,
  stage1_fold_ids,
  fold_results
)

stage1_survivors <- initial_configs[
  config_id %in%
    select_survivors(stage1_ranking)$config_id
]

fwrite(
  stage1_ranking,
  file.path(RESULTS_DIR, "stage1_ranking.csv")
)

stage2_fold_ids <- tail(inner_splits$fold_id, 2L)

for (fold_id_value in stage2_fold_ids) {
  fold_results <- run_config_batch_on_fold(
    stage1_survivors,
    inner_splits[fold_id == fold_id_value],
    "successive_halving_stage2",
    fold_results
  )
}

stage2_ranking <- rank_configurations(
  stage1_survivors,
  stage2_fold_ids,
  fold_results
)

stage2_survivors <- stage1_survivors[
  config_id %in%
    select_survivors(stage2_ranking)$config_id
]

fwrite(
  stage2_ranking,
  file.path(RESULTS_DIR, "stage2_ranking.csv")
)

for (fold_id_value in inner_splits$fold_id) {
  fold_results <- run_config_batch_on_fold(
    stage2_survivors,
    inner_splits[fold_id == fold_id_value],
    "successive_halving_stage3",
    fold_results
  )
}

stage3_ranking <- rank_configurations(
  stage2_survivors,
  inner_splits$fold_id,
  fold_results
)

if (nrow(stage3_ranking) == 0L) {
  stop("No configuration completed all three inner folds.")
}

fwrite(
  stage3_ranking,
  file.path(RESULTS_DIR, "stage3_ranking.csv")
)

local_configs <- generate_local_configs(
  stage3_ranking[
    seq_len(
      min(N_LOCAL_CENTERS, nrow(stage3_ranking))
    )
  ],
  N_LOCAL_CONFIGS,
  initial_configs
)

fwrite(
  local_configs,
  file.path(RESULTS_DIR, "local_candidates.csv")
)

for (fold_id_value in inner_splits$fold_id) {
  fold_results <- run_config_batch_on_fold(
    local_configs,
    inner_splits[fold_id == fold_id_value],
    "local_refinement",
    fold_results
  )
}

local_ranking <- rank_configurations(
  local_configs,
  inner_splits$fold_id,
  fold_results
)

fwrite(
  local_ranking,
  file.path(RESULTS_DIR, "local_ranking.csv")
)

complete_configs <- unique(
  rbindlist(
    list(
      baseline_config,
      reference_config,
      stage2_survivors,
      local_configs
    ),
    use.names = TRUE,
    fill = TRUE
  ),
  by = "config_id"
)

complete_configs <- add_signatures(complete_configs)

complete_inner_ranking <- rank_configurations(
  complete_configs,
  inner_splits$fold_id,
  fold_results
)

if (nrow(complete_inner_ranking) == 0L) {
  stop("No configuration completed the full inner comparison.")
}

setorder(
  complete_inner_ranking,
  mean_rmsle,
  sd_rmsle,
  worst_fold_rmsle
)

selected_inner_config <- copy(
  complete_inner_ranking[1]
)

selected_inner_config[
  ,
  selection_basis :=
    "Lowest mean RMSLE across the three chronological inner folds; SD and worst-fold RMSLE used as tie-breakers"
]

fwrite(
  fold_results,
  file.path(RESULTS_DIR, "hpo_fold_scores.csv")
)

fwrite(
  complete_inner_ranking,
  file.path(RESULTS_DIR, "hpo_summary.csv")
)

selected_source_config_id <- selected_inner_config$config_id[1]

selected_hpo_config <- copy(selected_inner_config)
selected_hpo_config[
  ,
  `:=`(
    configuration_name = "hpo_config",
    source_config_id = selected_source_config_id
  )
]

fwrite(
  selected_hpo_config,
  file.path(RESULTS_DIR, "selected_hpo_config.csv")
)

selected_config_id <- selected_source_config_id

selected_config <- complete_configs[
  config_id == selected_config_id
]

outer_report_configs <- unique(
  rbindlist(
    list(
      baseline_config,
      selected_config
    ),
    use.names = TRUE,
    fill = TRUE
  ),
  by = "config_id"
)

fold_results <- run_config_batch_on_fold(
  outer_report_configs,
  outer_split,
  "outer_confirmation_reporting_only",
  fold_results
)

outer_confirmation <- fold_results[
  status == "ok" &
    fold_id == "outer_48d" &
    config_id %in% outer_report_configs$config_id
]

outer_confirmation[
  ,
  selected_from_inner_folds :=
    config_id == selected_config_id
]

outer_confirmation[
  ,
  outer_used_for_selection := FALSE
]

fwrite(
  outer_confirmation,
  file.path(RESULTS_DIR, "outer_confirmation.csv")
)

selected_inner_rounds <- fold_results[
  status == "ok" &
    config_id == selected_config_id &
    fold_id %in% inner_splits$fold_id,
  best_nrounds
]

recommended_nrounds <- as.integer(
  round(
    median(
      selected_inner_rounds,
      na.rm = TRUE
    )
  )
)

if (
  !is.finite(recommended_nrounds) ||
    recommended_nrounds < 1L
) {
  recommended_nrounds <- 300L
}

hpo_config <- selected_inner_config[
  ,
  .(
    configuration_name = "hpo_config",
    source_config_id = config_id,
    config_source,
    center_id,
    eta,
    max_depth,
    min_child_weight,
    subsample,
    colsample_bytree,
    gamma,
    lambda,
    alpha,
    recommended_nrounds,
    inner_mean_rmsle = mean_rmsle,
    inner_sd_rmsle = sd_rmsle,
    inner_worst_fold_rmsle = worst_fold_rmsle
  )
]

final_parameters <- copy(hpo_config)

fwrite(
  final_parameters,
  file.path(RESULTS_DIR, "hpo_config.csv")
)

run_configuration <- data.table(
  setting = c(
    "timezone",
    "target",
    "features",
    "inner_folds",
    "inner_validation_days",
    "outer_validation_days",
    "random_configs",
    "local_configs",
    "successive_halving_eta",
    "max_nrounds",
    "early_stopping_rounds",
    "seed",
    "threads",
    "selection_basis",
    "outer_role",
    "post_processing"
  ),
  value = c(
    "UTC",
    "log1p(bikes)",
    as.character(length(FINAL_SET)),
    as.character(INNER_FOLDS),
    as.character(INNER_VALID_DAYS),
    as.character(OUTER_VALID_DAYS),
    as.character(N_RANDOM_CONFIGS),
    as.character(N_LOCAL_CONFIGS),
    as.character(SUCCESSIVE_HALVING_ETA),
    as.character(MAX_NROUNDS),
    as.character(EARLY_STOPPING_ROUNDS),
    as.character(SEED),
    as.character(N_THREADS),
    "inner three-fold mean RMSLE",
    "reporting-only confirmation",
    "none"
  )
)

fwrite(
  run_configuration,
  file.path(RESULTS_DIR, "run_configuration.csv")
)

print(validation_splits)
print(complete_inner_ranking[1:min(10L, .N)])
print(selected_hpo_config)
print(outer_confirmation)
print(hpo_config)
