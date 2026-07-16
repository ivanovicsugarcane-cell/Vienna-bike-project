required_packages <- c("tidyverse", "lubridate", "xgboost", "Matrix")

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(paste("Missing packages:", paste(missing_packages, collapse = ", ")))
}

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(xgboost)
  library(Matrix)
})

set.seed(123)

TRAIN_FILE <- "train.csv"
TEST_FILE <- "test.csv"
SUBMISSION_FILE <- "submission.csv"

FINAL_NROUNDS <- 448L

detected_cores <- parallel::detectCores()
if (is.na(detected_cores)) detected_cores <- 2L
N_THREADS <- max(1L, detected_cores - 1L)

FINAL_XGB_PARAMS <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 6L,
  eta = 0.03591504,
  min_child_weight = 3.99688001,
  subsample = 0.87071714,
  colsample_bytree = 0.65131130,
  gamma = 0.01718908,
  lambda = 0.31737196,
  alpha = 0.00620258,
  nthread = N_THREADS,
  seed = 123L
)

FINAL_FEATURES <- c(
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

safe_parse_datetime <- function(x) {
  if (inherits(x, "POSIXct") || inherits(x, "POSIXt")) {
    return(as.POSIXct(x, tz = "UTC"))
  }

  x_chr <- as.character(x)
  parsed <- suppressWarnings(ymd_hms(x_chr, tz = "UTC", quiet = TRUE))
  bad <- is.na(parsed) & !is.na(x_chr) & x_chr != ""

  if (any(bad)) {
    parsed[bad] <- suppressWarnings(ymd_hm(x_chr[bad], tz = "UTC", quiet = TRUE))
  }

  bad <- is.na(parsed) & !is.na(x_chr) & x_chr != ""

  if (any(bad)) {
    parsed[bad] <- suppressWarnings(as.POSIXct(x_chr[bad], tz = "UTC"))
  }

  parsed
}

add_time_features <- function(df) {
  df %>%
    mutate(
      hour = hour(datetime),
      minute = minute(datetime),
      weekday = wday(datetime, week_start = 1),
      is_weekend = if_else(weekday %in% c(6, 7), 1, 0),
      month = month(datetime),
      day = day(datetime),
      yday = yday(datetime),
      week = isoweek(datetime)
    )
}

make_final_features <- function(train_data, target_data) {
  station_profile <- train_data %>%
    group_by(station_number, hour, minute) %>%
    summarise(
      station_hour_minute_mean = mean(bikes, na.rm = TRUE),
      station_hour_minute_median = median(bikes, na.rm = TRUE),
      station_hour_minute_sd = sd(bikes, na.rm = TRUE),
      station_hour_minute_min = min(bikes, na.rm = TRUE),
      station_hour_minute_max = max(bikes, na.rm = TRUE),
      station_hour_minute_n = n(),
      .groups = "drop"
    ) %>%
    mutate(station_hour_minute_sd = coalesce(station_hour_minute_sd, 0))

  global_profile <- train_data %>%
    group_by(hour, minute) %>%
    summarise(
      global_hour_minute_mean = mean(bikes, na.rm = TRUE),
      global_hour_minute_median = median(bikes, na.rm = TRUE),
      global_hour_minute_sd = sd(bikes, na.rm = TRUE),
      global_hour_minute_min = min(bikes, na.rm = TRUE),
      global_hour_minute_max = max(bikes, na.rm = TRUE),
      global_hour_minute_n = n(),
      .groups = "drop"
    ) %>%
    mutate(global_hour_minute_sd = coalesce(global_hour_minute_sd, 0))

  add_profiles <- function(df) {
    df %>%
      left_join(
        station_profile,
        by = c("station_number", "hour", "minute")
      ) %>%
      left_join(
        global_profile,
        by = c("hour", "minute")
      ) %>%
      arrange(.row_id)
  }

  list(
    train = add_profiles(train_data),
    test = add_profiles(target_data)
  )
}

if (!file.exists(TRAIN_FILE)) {
  stop("train.csv was not found in the current working directory.")
}

if (!file.exists(TEST_FILE)) {
  stop("test.csv was not found in the current working directory.")
}

train <- read_csv(TRAIN_FILE, show_col_types = FALSE) %>%
  mutate(
    .row_id = row_number(),
    datetime = safe_parse_datetime(datetime),
    station_number = as.character(station_number)
  )

test <- read_csv(TEST_FILE, show_col_types = FALSE) %>%
  mutate(
    .row_id = row_number(),
    datetime = safe_parse_datetime(datetime),
    station_number = as.character(station_number)
  )

required_train_columns <- c("datetime", "station_number", "bikes", "lat", "lng")
required_test_columns <- c("datetime", "station_number", "lat", "lng")

missing_train_columns <- setdiff(required_train_columns, names(train))
missing_test_columns <- setdiff(required_test_columns, names(test))

if (length(missing_train_columns) > 0) {
  stop(paste("Missing train columns:", paste(missing_train_columns, collapse = ", ")))
}

if (length(missing_test_columns) > 0) {
  stop(paste("Missing test columns:", paste(missing_test_columns, collapse = ", ")))
}

if (anyNA(train$datetime)) {
  stop("Some train datetime values could not be parsed.")
}

if (anyNA(test$datetime)) {
  stop("Some test datetime values could not be parsed.")
}

if (anyNA(train$bikes) || any(train$bikes < 0)) {
  stop("The bikes target contains missing or negative values.")
}

train_fe <- add_time_features(train)
test_fe <- add_time_features(test)

engineered <- make_final_features(train_fe, test_fe)

train_model <- engineered$train
test_model <- engineered$test

missing_train_features <- setdiff(FINAL_FEATURES, names(train_model))
missing_test_features <- setdiff(FINAL_FEATURES, names(test_model))

if (length(missing_train_features) > 0) {
  stop(paste("Missing train features:", paste(missing_train_features, collapse = ", ")))
}

if (length(missing_test_features) > 0) {
  stop(paste("Missing test features:", paste(missing_test_features, collapse = ", ")))
}

train_xgb_df <- train_model %>%
  select(all_of(FINAL_FEATURES)) %>%
  mutate(station_number = as.factor(station_number))

station_levels <- levels(train_xgb_df$station_number)

test_xgb_df <- test_model %>%
  select(all_of(FINAL_FEATURES)) %>%
  mutate(
    station_number = factor(
      station_number,
      levels = station_levels
    )
  )

if (anyNA(test_xgb_df$station_number)) {
  stop("test.csv contains station numbers not found in train.csv.")
}

train_xgb_matrix <- sparse.model.matrix(~ . - 1, data = train_xgb_df)
test_xgb_matrix <- sparse.model.matrix(~ . - 1, data = test_xgb_df)

train_columns <- colnames(train_xgb_matrix)

missing_test_matrix_columns <- setdiff(
  train_columns,
  colnames(test_xgb_matrix)
)

if (length(missing_test_matrix_columns) > 0) {
  zero_matrix <- Matrix(
    0,
    nrow = nrow(test_xgb_matrix),
    ncol = length(missing_test_matrix_columns),
    sparse = TRUE
  )

  colnames(zero_matrix) <- missing_test_matrix_columns
  test_xgb_matrix <- cbind(test_xgb_matrix, zero_matrix)
}

extra_test_matrix_columns <- setdiff(
  colnames(test_xgb_matrix),
  train_columns
)

if (length(extra_test_matrix_columns) > 0) {
  test_xgb_matrix <- test_xgb_matrix[
    ,
    setdiff(colnames(test_xgb_matrix), extra_test_matrix_columns),
    drop = FALSE
  ]
}

test_xgb_matrix <- test_xgb_matrix[, train_columns, drop = FALSE]

dtrain <- xgb.DMatrix(
  data = train_xgb_matrix,
  label = log1p(train_model$bikes),
  missing = NA
)

dtest <- xgb.DMatrix(
  data = test_xgb_matrix,
  missing = NA
)

final_xgb_model <- xgb.train(
  params = FINAL_XGB_PARAMS,
  data = dtrain,
  nrounds = FINAL_NROUNDS,
  verbose = 1
)

pred_log <- predict(final_xgb_model, dtest)
pred_raw <- pmax(expm1(pred_log), 0)

prediction_scale <- case_when(
  pred_raw < 3 ~ 0.8925,
  pred_raw < 8 ~ 0.7800,
  pred_raw < 15 ~ 0.6300,
  pred_raw < 25 ~ 0.5800,
  TRUE ~ 0.4200
)

pred_final <- as.integer(
  pmax(
    floor(pred_raw * prediction_scale + 0.05),
    0
  )
)

submission_id <- if ("id" %in% names(test_model)) {
  as.character(test_model$id)
} else {
  paste0(
    format(
      test_model$datetime,
      "%Y-%m-%d %H:%M:%S",
      tz = "UTC"
    ),
    "_",
    test_model$station_number
  )
}

submission <- tibble(
  id = submission_id,
  bikes = pred_final
)

if (nrow(submission) != nrow(test)) {
  stop("Submission row count does not match test.csv.")
}

if (anyNA(submission$id) || anyNA(submission$bikes)) {
  stop("Submission contains missing values.")
}

if (any(submission$bikes < 0)) {
  stop("Submission contains negative predictions.")
}

if (anyDuplicated(submission$id)) {
  stop("Submission contains duplicated IDs.")
}

write_csv(submission, SUBMISSION_FILE)

if (!file.exists(SUBMISSION_FILE)) {
  stop("submission.csv was not created.")
}
cat("Rows:", nrow(submission), "\n")
cat("Missing IDs:", sum(is.na(submission$id)), "\n")
cat("Missing predictions:", sum(is.na(submission$bikes)), "\n")
cat("Duplicate IDs:", anyDuplicated(submission$id), "\n")
cat("Minimum prediction:", min(submission$bikes), "\n")
cat("Maximum prediction:", max(submission$bikes), "\n")
cat("CSV:", normalizePath(SUBMISSION_FILE), "\n")
