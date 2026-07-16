library(data.table)
library(lubridate)
library(xgboost)
library(Matrix)

set.seed(123)

DATA_DIR <- "."
TRAIN_FILE <- file.path(DATA_DIR, "train.csv")
RESULTS_DIR <- file.path("results", "feature_set_comparison")

OUTER_VALID_DAYS <- 48L
INNER_FOLDS <- 3L
INNER_VALID_DAYS <- 14L
MIN_INNER_TRAIN_DAYS <- 45L

NROUNDS <- 1200L
EARLY_STOPPING_ROUNDS <- 50L
PRINT_EVERY_N <- 100L
SEED <- 123L
RMSLE_TOLERANCE <- 0.001

available_cores <- parallel::detectCores()
if (is.na(available_cores)) available_cores <- 2L
N_THREADS <- max(1L, available_cores - 1L)

XGB_PARAMS <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.03,
  max_depth = 6L,
  min_child_weight = 10,
  subsample = 0.85,
  colsample_bytree = 0.85,
  lambda = 1,
  alpha = 0,
  tree_method = "hist",
  nthread = N_THREADS,
  seed = SEED
)

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

add_raw_time_features <- function(data) {
  out <- copy(data)
  out[, date := as.Date(datetime_utc, tz = "UTC")]
  out[, hour := hour(datetime_utc)]
  out[, minute := minute(datetime_utc)]
  out[, halfhour := hour * 2 + minute / 30]
  out[, weekday := wday(datetime_utc, week_start = 1)]
  out[, is_weekend := as.integer(weekday %in% c(6L, 7L))]
  out[, month := month(datetime_utc)]
  out[, day := day(datetime_utc)]
  out[, yday := yday(datetime_utc)]
  out[, week := isoweek(datetime_utc)]
  out
}

add_safe_engineered_features <- function(source_data, target_data) {
  out <- copy(target_data)

  out[, sin_halfhour := sin(2 * pi * halfhour / 48)]
  out[, cos_halfhour := cos(2 * pi * halfhour / 48)]
  out[, sin_weekday := sin(2 * pi * weekday / 7)]
  out[, cos_weekday := cos(2 * pi * weekday / 7)]
  out[, sin_month := sin(2 * pi * month / 12)]
  out[, cos_month := cos(2 * pi * month / 12)]

  source_start_date <- min(source_data$date, na.rm = TRUE)
  out[, days_since_start := as.numeric(date - source_start_date)]
  out[, weeks_since_start := days_since_start / 7]

  center_lat <- mean(source_data$lat, na.rm = TRUE)
  center_lng <- mean(source_data$lng, na.rm = TRUE)
  lat_scale <- 111
  lng_scale <- 111 * cos(center_lat * pi / 180)

  out[, lat_centered := lat - center_lat]
  out[, lng_centered := lng - center_lng]
  out[, distance_to_center := sqrt(
    (lat_centered * lat_scale)^2 +
      (lng_centered * lng_scale)^2
  )]
  out[, lat_lng_interaction := lat * lng]
  out[, lat_sq := lat^2]
  out[, lng_sq := lng^2]

  if ("name" %in% names(out)) {
    out[, station_name := as.character(name)]
    out[is.na(station_name), station_name := "unknown"]
    out[, name_length := nchar(station_name)]
    out[, name_word_count := lengths(strsplit(station_name, "\\s+"))]

    station_name_lower <- tolower(out$station_name)
    out[, name_has_bahnhof := as.integer(
      grepl("bahnhof|hbf|bf", station_name_lower)
    )]
    out[, name_has_uni := as.integer(
      grepl("uni|universität|universitaet|tu ", station_name_lower)
    )]
    out[, name_has_strasse := as.integer(
      grepl("straße|strasse|str\\.", station_name_lower)
    )]
    out[, name_has_platz := as.integer(
      grepl("platz", station_name_lower)
    )]
    out[, name_has_center := as.integer(
      grepl("zentrum|center|centre|markt", station_name_lower)
    )]
  } else {
    out[, station_name := "unknown"]
    out[, name_length := 0]
    out[, name_word_count := 0]
    out[, name_has_bahnhof := 0]
    out[, name_has_uni := 0]
    out[, name_has_strasse := 0]
    out[, name_has_platz := 0]
    out[, name_has_center := 0]
  }

  out
}

add_profile_features <- function(source_data, target_data) {
  source_data <- copy(source_data)
  target_data <- copy(target_data)

  station_hour_minute_profile <- source_data[
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

  station_hour_minute_profile[
    is.na(station_hour_minute_sd),
    station_hour_minute_sd := 0
  ]

  global_hour_minute_profile <- source_data[
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

  global_hour_minute_profile[
    is.na(global_hour_minute_sd),
    global_hour_minute_sd := 0
  ]

  station_static_profile <- source_data[
    ,
    .(
      station_mean = mean(bikes, na.rm = TRUE),
      station_median = median(bikes, na.rm = TRUE),
      station_sd = sd(bikes, na.rm = TRUE),
      station_min = min(bikes, na.rm = TRUE),
      station_max = max(bikes, na.rm = TRUE),
      station_q25 = as.numeric(quantile(bikes, 0.25, na.rm = TRUE)),
      station_q75 = as.numeric(quantile(bikes, 0.75, na.rm = TRUE)),
      station_zero_rate = mean(bikes == 0, na.rm = TRUE),
      station_low_rate = mean(bikes <= 2, na.rm = TRUE),
      station_high_rate = mean(bikes >= 15, na.rm = TRUE),
      station_n = .N
    ),
    by = station_number
  ]

  station_static_profile[is.na(station_sd), station_sd := 0]

  station_weekday_hour_profile <- source_data[
    ,
    .(
      station_weekday_hour_mean = mean(bikes, na.rm = TRUE),
      station_weekday_hour_median = median(bikes, na.rm = TRUE),
      station_weekday_hour_sd = sd(bikes, na.rm = TRUE),
      station_weekday_hour_n = .N
    ),
    by = .(station_number, weekday, hour)
  ]

  station_weekday_hour_profile[
    is.na(station_weekday_hour_sd),
    station_weekday_hour_sd := 0
  ]

  global_weekday_hour_profile <- source_data[
    ,
    .(
      global_weekday_hour_mean = mean(bikes, na.rm = TRUE),
      global_weekday_hour_median = median(bikes, na.rm = TRUE),
      global_weekday_hour_sd = sd(bikes, na.rm = TRUE),
      global_weekday_hour_n = .N
    ),
    by = .(weekday, hour)
  ]

  global_weekday_hour_profile[
    is.na(global_weekday_hour_sd),
    global_weekday_hour_sd := 0
  ]

  station_month_profile <- source_data[
    ,
    .(
      station_month_mean = mean(bikes, na.rm = TRUE),
      station_month_median = median(bikes, na.rm = TRUE),
      station_month_sd = sd(bikes, na.rm = TRUE),
      station_month_n = .N
    ),
    by = .(station_number, month)
  ]

  station_month_profile[
    is.na(station_month_sd),
    station_month_sd := 0
  ]

  global_month_profile <- source_data[
    ,
    .(
      global_month_mean = mean(bikes, na.rm = TRUE),
      global_month_median = median(bikes, na.rm = TRUE),
      global_month_sd = sd(bikes, na.rm = TRUE),
      global_month_n = .N
    ),
    by = month
  ]

  global_month_profile[
    is.na(global_month_sd),
    global_month_sd := 0
  ]

  target_data[, row_order_internal := .I]

  out <- merge(
    target_data,
    station_hour_minute_profile,
    by = c("station_number", "hour", "minute"),
    all.x = TRUE,
    sort = FALSE
  )

  out <- merge(
    out,
    global_hour_minute_profile,
    by = c("hour", "minute"),
    all.x = TRUE,
    sort = FALSE
  )

  out <- merge(
    out,
    station_static_profile,
    by = "station_number",
    all.x = TRUE,
    sort = FALSE
  )

  out <- merge(
    out,
    station_weekday_hour_profile,
    by = c("station_number", "weekday", "hour"),
    all.x = TRUE,
    sort = FALSE
  )

  out <- merge(
    out,
    global_weekday_hour_profile,
    by = c("weekday", "hour"),
    all.x = TRUE,
    sort = FALSE
  )

  out <- merge(
    out,
    station_month_profile,
    by = c("station_number", "month"),
    all.x = TRUE,
    sort = FALSE
  )

  out <- merge(
    out,
    global_month_profile,
    by = "month",
    all.x = TRUE,
    sort = FALSE
  )

  overall_mean <- mean(source_data$bikes, na.rm = TRUE)
  overall_median <- median(source_data$bikes, na.rm = TRUE)
  overall_sd <- sd(source_data$bikes, na.rm = TRUE)
  overall_min <- min(source_data$bikes, na.rm = TRUE)
  overall_max <- max(source_data$bikes, na.rm = TRUE)

  if (!is.finite(overall_sd)) overall_sd <- 0

  fill_defaults <- c(
    global_hour_minute_mean = overall_mean,
    global_hour_minute_median = overall_median,
    global_hour_minute_sd = overall_sd,
    global_hour_minute_min = overall_min,
    global_hour_minute_max = overall_max,
    global_hour_minute_n = 0,
    global_weekday_hour_mean = overall_mean,
    global_weekday_hour_median = overall_median,
    global_weekday_hour_sd = overall_sd,
    global_weekday_hour_n = 0,
    global_month_mean = overall_mean,
    global_month_median = overall_median,
    global_month_sd = overall_sd,
    global_month_n = 0
  )

  for (feature_name in names(fill_defaults)) {
    set(
      out,
      i = which(is.na(out[[feature_name]])),
      j = feature_name,
      value = fill_defaults[[feature_name]]
    )
  }

  station_fallbacks <- list(
    station_hour_minute_mean = "global_hour_minute_mean",
    station_hour_minute_median = "global_hour_minute_median",
    station_hour_minute_sd = "global_hour_minute_sd",
    station_hour_minute_min = "global_hour_minute_min",
    station_hour_minute_max = "global_hour_minute_max",
    station_hour_minute_n = "global_hour_minute_n",
    station_weekday_hour_mean = "global_weekday_hour_mean",
    station_weekday_hour_median = "global_weekday_hour_median",
    station_weekday_hour_sd = "global_weekday_hour_sd",
    station_weekday_hour_n = "global_weekday_hour_n",
    station_month_mean = "global_month_mean",
    station_month_median = "global_month_median",
    station_month_sd = "global_month_sd",
    station_month_n = "global_month_n"
  )

  for (feature_name in names(station_fallbacks)) {
    fallback_name <- station_fallbacks[[feature_name]]
    missing_rows <- which(is.na(out[[feature_name]]))
    if (length(missing_rows) > 0) {
      set(
        out,
        i = missing_rows,
        j = feature_name,
        value = out[[fallback_name]][missing_rows]
      )
    }
  }

  station_static_features <- setdiff(
    names(station_static_profile),
    "station_number"
  )

  for (feature_name in station_static_features) {
    fallback_value <- mean(
      station_static_profile[[feature_name]],
      na.rm = TRUE
    )
    if (!is.finite(fallback_value)) fallback_value <- 0
    set(
      out,
      i = which(is.na(out[[feature_name]])),
      j = feature_name,
      value = fallback_value
    )
  }

  setorder(out, row_order_internal)
  out[, row_order_internal := NULL]

  out
}

add_interaction_keys <- function(data) {
  out <- copy(data)
  out[, station_halfhour_key := paste(station_number, halfhour, sep = "_")]
  out[, station_weekday_key := paste(station_number, weekday, sep = "_")]
  out[, station_month_key := paste(station_number, month, sep = "_")]
  out
}

build_full_feature_dataset <- function(source_data, target_data) {
  out <- add_safe_engineered_features(source_data, target_data)
  out <- add_profile_features(source_data, out)
  out <- add_interaction_keys(out)
  out
}

C0 <- c(
  "station_number",
  "hour",
  "minute",
  "halfhour",
  "weekday",
  "is_weekend",
  "month",
  "day",
  "yday",
  "week"
)

C1_ADDED <- c(
  "sin_halfhour",
  "cos_halfhour",
  "sin_weekday",
  "cos_weekday",
  "sin_month",
  "cos_month",
  "days_since_start",
  "weeks_since_start"
)

C2_ADDED <- c(
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

C3_ADDED <- c(
  "lat",
  "lng",
  "lat_centered",
  "lng_centered",
  "distance_to_center",
  "lat_lng_interaction",
  "lat_sq",
  "lng_sq"
)

C4_ADDED <- c(
  "station_name",
  "name_length",
  "name_word_count",
  "name_has_bahnhof",
  "name_has_uni",
  "name_has_strasse",
  "name_has_platz",
  "name_has_center"
)

C5_ADDED <- c(
  "station_mean",
  "station_median",
  "station_sd",
  "station_min",
  "station_max",
  "station_q25",
  "station_q75",
  "station_zero_rate",
  "station_low_rate",
  "station_high_rate",
  "station_n",
  "station_weekday_hour_mean",
  "station_weekday_hour_median",
  "station_weekday_hour_sd",
  "station_weekday_hour_n",
  "global_weekday_hour_mean",
  "global_weekday_hour_median",
  "global_weekday_hour_sd",
  "global_weekday_hour_n",
  "station_month_mean",
  "station_month_median",
  "station_month_sd",
  "station_month_n",
  "global_month_mean",
  "global_month_median",
  "global_month_sd",
  "global_month_n",
  "station_halfhour_key",
  "station_weekday_key",
  "station_month_key"
)

COMPACT_C2 <- c(
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

FEATURE_SETS <- list(
  C0 = C0,
  C1 = unique(c(C0, C1_ADDED)),
  C2 = unique(c(C0, C1_ADDED, C2_ADDED)),
  C3 = unique(c(C0, C1_ADDED, C2_ADDED, C3_ADDED)),
  C4 = unique(c(C0, C1_ADDED, C2_ADDED, C3_ADDED, C4_ADDED)),
  C5 = unique(c(
    C0,
    C1_ADDED,
    C2_ADDED,
    C3_ADDED,
    C4_ADDED,
    C5_ADDED
  )),
  COMPACT_C2_23 = COMPACT_C2
)

feature_definitions <- rbindlist(
  lapply(names(FEATURE_SETS), function(feature_set_name) {
    data.table(
      feature_set = feature_set_name,
      feature_order = seq_along(FEATURE_SETS[[feature_set_name]]),
      feature = FEATURE_SETS[[feature_set_name]],
      n_features = length(FEATURE_SETS[[feature_set_name]])
    )
  })
)

fwrite(
  feature_definitions,
  file.path(RESULTS_DIR, "feature_set_definitions.csv")
)

make_validation_splits <- function(data) {
  max_date <- max(data$date, na.rm = TRUE)
  min_date <- min(data$date, na.rm = TRUE)

  outer_valid_start <- max_date - OUTER_VALID_DAYS + 1L
  outer_train_end <- outer_valid_start - 1L
  development_data <- data[date <= outer_train_end]
  development_max_date <- max(development_data$date, na.rm = TRUE)

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
      stop("The requested chronological inner folds cannot be created.")
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
    ]
  )
}

get_best_round <- function(model) {
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

  best_ntreelimit <- tryCatch(
    as.integer(model$best_ntreelimit),
    error = function(e) NA_integer_
  )

  if (length(best_ntreelimit) == 1L && is.finite(best_ntreelimit)) {
    return(best_ntreelimit)
  }

  NROUNDS
}

predict_at_best_round <- function(model, data, best_round) {
  tryCatch(
    predict(
      model,
      data,
      iterationrange = c(1L, best_round + 1L)
    ),
    error = function(e) {
      tryCatch(
        predict(model, data, ntreelimit = best_round),
        error = function(e2) predict(model, data)
      )
    }
  )
}

fit_feature_set <- function(
  train_model,
  valid_model,
  feature_set_name,
  selected_features,
  fold_id
) {
  missing_train <- setdiff(selected_features, names(train_model))
  missing_valid <- setdiff(selected_features, names(valid_model))

  if (length(missing_train) > 0L || length(missing_valid) > 0L) {
    stop(paste(
      "Feature construction failed for",
      feature_set_name
    ))
  }

  matrices <- prepare_model_matrices(
    train_model,
    valid_model,
    selected_features
  )

  y_train_log <- log1p(train_model$bikes)
  y_valid_log <- log1p(valid_model$bikes)

  dtrain <- xgb.DMatrix(
    data = matrices$train,
    label = y_train_log,
    missing = NA
  )

  dvalid <- xgb.DMatrix(
    data = matrices$valid,
    label = y_valid_log,
    missing = NA
  )

  start_time <- Sys.time()

  model <- tryCatch(
    xgb.train(
      params = XGB_PARAMS,
      data = dtrain,
      nrounds = NROUNDS,
      evals = list(train = dtrain, valid = dvalid),
      early_stopping_rounds = EARLY_STOPPING_ROUNDS,
      print_every_n = PRINT_EVERY_N,
      verbose = 0
    ),
    error = function(e) {
      xgb.train(
        params = XGB_PARAMS,
        data = dtrain,
        nrounds = NROUNDS,
        watchlist = list(train = dtrain, valid = dvalid),
        early_stopping_rounds = EARLY_STOPPING_ROUNDS,
        print_every_n = PRINT_EVERY_N,
        verbose = 0
      )
    }
  )

  train_seconds <- as.numeric(
    difftime(Sys.time(), start_time, units = "secs")
  )

  best_round <- get_best_round(model)
  pred_log <- predict_at_best_round(model, dvalid, best_round)
  pred_bikes <- pmax(0, expm1(pred_log))

  result <- data.table(
    fold_id = fold_id,
    feature_set = feature_set_name,
    n_features = length(selected_features),
    n_matrix_columns = ncol(matrices$train),
    valid_rmsle = rmsle(valid_model$bikes, pred_bikes),
    valid_rmse_log = sqrt(
      mean((pred_log - y_valid_log)^2, na.rm = TRUE)
    ),
    best_iteration = best_round,
    train_seconds = train_seconds
  )

  rm(model, matrices, dtrain, dvalid, pred_log, pred_bikes)
  gc(verbose = FALSE)

  result
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
  stop(paste(
    "Missing required columns:",
    paste(missing_columns, collapse = ", ")
  ))
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

train <- add_raw_time_features(train)
setorder(train, datetime_utc, station_number, row_id_original)

validation_splits <- make_validation_splits(train)

fwrite(
  validation_splits,
  file.path(RESULTS_DIR, "validation_splits.csv")
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

  train_model <- build_full_feature_dataset(
    source_data,
    source_data
  )

  valid_model <- build_full_feature_dataset(
    source_data,
    valid_data
  )

  one_fold_results <- vector("list", length(FEATURE_SETS))

  for (feature_index in seq_along(FEATURE_SETS)) {
    feature_set_name <- names(FEATURE_SETS)[feature_index]

    one_fold_results[[feature_index]] <- fit_feature_set(
      train_model = train_model,
      valid_model = valid_model,
      feature_set_name = feature_set_name,
      selected_features = FEATURE_SETS[[feature_set_name]],
      fold_id = split_row$fold_id
    )
  }

  fold_results[[fold_index]] <- rbindlist(one_fold_results)

  rm(
    source_data,
    valid_data,
    train_model,
    valid_model,
    one_fold_results
  )
  gc(verbose = FALSE)
}

feature_set_fold_scores <- rbindlist(fold_results)

feature_set_summary <- feature_set_fold_scores[
  ,
  .(
    n_features = first(n_features),
    mean_rmsle = mean(valid_rmsle),
    sd_rmsle = sd(valid_rmsle),
    worst_fold_rmsle = max(valid_rmsle),
    best_fold_rmsle = min(valid_rmsle),
    mean_best_iteration = mean(best_iteration),
    median_best_iteration = median(best_iteration),
    mean_train_seconds = mean(train_seconds),
    total_train_seconds = sum(train_seconds)
  ),
  by = feature_set
]

setorder(
  feature_set_summary,
  mean_rmsle,
  sd_rmsle,
  n_features
)

feature_set_summary[, rank := seq_len(.N)]
feature_set_summary[
  ,
  delta_vs_best := mean_rmsle - min(mean_rmsle)
]

best_mean_rmsle <- min(feature_set_summary$mean_rmsle)

eligible_sets <- feature_set_summary[
  mean_rmsle <= best_mean_rmsle + RMSLE_TOLERANCE
]

setorder(
  eligible_sets,
  n_features,
  mean_rmsle,
  sd_rmsle
)

feature_set_recommendation <- copy(eligible_sets[1])
feature_set_recommendation[
  ,
  selection_rule := paste0(
    "Lowest feature count within ",
    RMSLE_TOLERANCE,
    " RMSLE of the best three-fold mean"
  )
]

fwrite(
  feature_set_fold_scores,
  file.path(RESULTS_DIR, "feature_set_fold_scores.csv")
)

fwrite(
  feature_set_summary,
  file.path(RESULTS_DIR, "feature_set_summary.csv")
)

fwrite(
  feature_set_recommendation,
  file.path(RESULTS_DIR, "feature_set_recommendation.csv")
)

outer_split <- validation_splits[fold_id == "outer_48d"]

outer_source_data <- copy(train[date <= outer_split$train_end])
outer_valid_data <- copy(train[
  date >= outer_split$valid_start &
    date <= outer_split$valid_end
])

outer_train_model <- build_full_feature_dataset(
  outer_source_data,
  outer_source_data
)

outer_valid_model <- build_full_feature_dataset(
  outer_source_data,
  outer_valid_data
)

selected_feature_set <- feature_set_recommendation$feature_set[1]

outer_confirmation <- fit_feature_set(
  train_model = outer_train_model,
  valid_model = outer_valid_model,
  feature_set_name = selected_feature_set,
  selected_features = FEATURE_SETS[[selected_feature_set]],
  fold_id = "outer_48d"
)

outer_confirmation[
  ,
  inner_mean_rmsle :=
    feature_set_recommendation$mean_rmsle[1]
]

outer_confirmation[
  ,
  inner_sd_rmsle :=
    feature_set_recommendation$sd_rmsle[1]
]

fwrite(
  outer_confirmation,
  file.path(RESULTS_DIR, "outer_confirmation.csv")
)

run_configuration <- data.table(
  setting = c(
    "timezone",
    "target",
    "inner_folds",
    "inner_validation_days",
    "outer_validation_days",
    "nrounds",
    "early_stopping_rounds",
    "rmsle_tolerance",
    "seed",
    "post_processing"
  ),
  value = c(
    "UTC",
    "log1p(bikes)",
    as.character(INNER_FOLDS),
    as.character(INNER_VALID_DAYS),
    as.character(OUTER_VALID_DAYS),
    as.character(NROUNDS),
    as.character(EARLY_STOPPING_ROUNDS),
    as.character(RMSLE_TOLERANCE),
    as.character(SEED),
    "none"
  )
)

fwrite(
  run_configuration,
  file.path(RESULTS_DIR, "run_configuration.csv")
)

print(validation_splits)
print(feature_set_summary)
print(feature_set_recommendation)
print(outer_confirmation)
