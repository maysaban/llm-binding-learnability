# LLM Binding Learnability — linear mixed-effects analysis
#
# Proposal: one lme4 model per experiment per LLM.
#   Exp 1: surprisal ~ local_match * matrix_match + (1 | item_id)
#   Exp 2: surprisal ~ head_match * distractor_match + (1 | item_id)
#   Exp 3: surprisal ~ local_match * context_type + (1 | item_id)
#
# Factors use ±0.5 sum coding so coefficients are surprisal differences
# (match minus mismatch), averaged over the other factor. Negative = the
# named "match" level is more expected (lower surprisal).
#
# Run from the repo root or from analysis/:
#   Rscript analysis/run_analysis.R

options(warn = 1)
options(contrasts = c("contr.sum", "contr.poly"))

CRAN_PACKAGES <- c(
  "lme4",
  "lmerTest",
  "emmeans",
  "ggplot2",
  "dplyr",
  "tidyr",
  "readr",
  "purrr",
  "tibble",
  "stringr"
)

ensure_packages <- function(packages) {
  installed <- rownames(installed.packages())
  missing <- setdiff(packages, installed)
  if (length(missing)) {
    message("Installing CRAN packages: ", paste(missing, collapse = ", "))
    install.packages(missing, repos = "https://cloud.r-project.org")
  }
  invisible(lapply(packages, library, character.only = TRUE))
}

ensure_packages(CRAN_PACKAGES)
emmeans::emm_options(lmer.df = "satterthwaite")


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  if (sys.nframe() > 0 && !is.null(sys.frame(1)$ofile)) {
    return(dirname(normalizePath(sys.frame(1)$ofile)))
  }
  normalizePath("analysis")
})()

ROOT <- normalizePath(file.path(script_dir, ".."))
RESULTS_DIR <- file.path(ROOT, "results")
OUTPUT_DIR <- file.path(script_dir, "output")
TABLE_DIR <- file.path(OUTPUT_DIR, "tables")
FIGURE_DIR <- file.path(OUTPUT_DIR, "figures")
MODEL_DIR <- file.path(OUTPUT_DIR, "model_summaries")

for (path in c(OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, MODEL_DIR)) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

MODEL_ORDER <- c(
  "babyllama",
  "qwen-2.5-7b",
  "llama-3.1-8b",
  "qwen-2.5-72b",
  "llama-3.1-70b"
)

MODEL_LABELS <- c(
  "babyllama" = "BabyLlama",
  "qwen-2.5-7b" = "Qwen-2.5-7B",
  "llama-3.1-8b" = "Llama-3.1-8B",
  "qwen-2.5-72b" = "Qwen-2.5-72B",
  "llama-3.1-70b" = "Llama-3.1-70B"
)

EXPERIMENT_LABELS <- c(
  "1" = "Exp 1: Locality domains",
  "2" = "Exp 2: Structural hierarchy",
  "3" = "Exp 3: Logophoric diagnostic"
)


# -----------------------------------------------------------------------------
# Factor coding
# -----------------------------------------------------------------------------

sum_code_05 <- function(x, level_order) {
  x <- factor(as.character(x), levels = level_order)
  stopifnot(nlevels(x) == 2)
  contrasts(x) <- matrix(c(-0.5, 0.5), ncol = 1)
  x
}

annotate_experiment <- function(df, experiment) {
  condition <- as.character(df$condition)
  df$item_id <- factor(df$item_id)
  df$surprisal <- as.numeric(df$surprisal)
  df$experiment <- as.integer(experiment)
  df$target <- as.character(df$target)

  if (experiment == 1) {
    df$local_match <- sum_code_05(
      ifelse(startsWith(condition, "+Local"), "match", "mismatch"),
      c("mismatch", "match")
    )
    df$matrix_match <- sum_code_05(
      ifelse(endsWith(condition, "+Matrix"), "match", "mismatch"),
      c("mismatch", "match")
    )
  } else if (experiment == 2) {
    df$head_match <- sum_code_05(
      ifelse(startsWith(condition, "+Head"), "match", "mismatch"),
      c("mismatch", "match")
    )
    df$distractor_match <- sum_code_05(
      ifelse(endsWith(condition, "+Distractor"), "match", "mismatch"),
      c("mismatch", "match")
    )
  } else if (experiment == 3) {
    df$local_match <- sum_code_05(
      ifelse(startsWith(condition, "+Local"), "match", "mismatch"),
      c("mismatch", "match")
    )
    df$context_type <- sum_code_05(
      ifelse(grepl("Coarg", condition, fixed = TRUE), "coarg", "picture_np"),
      c("coarg", "picture_np")
    )
  } else {
    stop("Unknown experiment: ", experiment)
  }
  df
}

discover_result_files <- function(results_dir) {
  files <- list.files(
    results_dir,
    pattern = "\\.csv$",
    recursive = TRUE,
    full.names = TRUE
  )
  tibble::tibble(path = files) %>%
    dplyr::mutate(
      rel = stringr::str_replace(
        normalizePath(path, winslash = "/"),
        paste0(normalizePath(results_dir, winslash = "/"), "/"),
        ""
      ),
      parts = strsplit(rel, "/")
    ) %>%
    dplyr::filter(purrr::map_int(parts, length) >= 2) %>%
    dplyr::mutate(
      model = purrr::map_chr(parts, 1),
      experiment_dir = purrr::map_chr(parts, 2),
      experiment = as.integer(stringr::str_extract(experiment_dir, "\\d+"))
    ) %>%
    dplyr::filter(model %in% names(MODEL_LABELS), experiment %in% 1:3)
}

validate_dataset <- function(df, model, experiment) {
  n <- nrow(df)
  n_items <- dplyr::n_distinct(df$item_id)
  n_conditions <- dplyr::n_distinct(df$condition)
  if (n != 128) {
    warning(model, " exp", experiment, ": expected 128 rows, got ", n)
  }
  if (n_items != 32) {
    warning(model, " exp", experiment, ": expected 32 items, got ", n_items)
  }
  if (n_conditions != 4) {
    warning(model, " exp", experiment, ": expected 4 conditions, got ", n_conditions)
  }
  if (anyNA(df$surprisal)) {
    stop(model, " exp", experiment, ": missing surprisal values")
  }
  invisible(df)
}


# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------

file_index <- discover_result_files(RESULTS_DIR)
if (nrow(file_index) == 0) {
  stop("No result CSVs found under ", RESULTS_DIR)
}

datasets <- purrr::pmap(
  file_index,
  function(path, rel, parts, model, experiment_dir, experiment) {
    df <- readr::read_csv(path, show_col_types = FALSE)
    required <- c("item_id", "condition", "sentence", "target", "surprisal")
    missing <- setdiff(required, names(df))
    if (length(missing)) {
      stop("Missing columns in ", path, ": ", paste(missing, collapse = ", "))
    }
    df <- annotate_experiment(df, experiment)
    df$model <- model
    df$model_label <- unname(MODEL_LABELS[model])
    validate_dataset(df, model, experiment)
    df
  }
)

names(datasets) <- paste0(
  file_index$model,
  "_exp",
  file_index$experiment
)

all_data <- dplyr::bind_rows(datasets)
all_data$model <- factor(all_data$model, levels = MODEL_ORDER)
all_data$model_label <- factor(
  all_data$model_label,
  levels = unname(MODEL_LABELS[MODEL_ORDER])
)


# -----------------------------------------------------------------------------
# Descriptives: by-item cell means, then mean / SE across items
# -----------------------------------------------------------------------------

item_cells <- all_data %>%
  dplyr::group_by(model, model_label, experiment, item_id, condition) %>%
  dplyr::summarise(surprisal = mean(surprisal), .groups = "drop")

cell_means <- item_cells %>%
  dplyr::group_by(model, model_label, experiment, condition) %>%
  dplyr::summarise(
    n_items = dplyr::n(),
    mean = mean(surprisal),
    sd = sd(surprisal),
    se = sd / sqrt(n_items),
    ci95_lo = mean - 1.96 * se,
    ci95_hi = mean + 1.96 * se,
    .groups = "drop"
  ) %>%
  dplyr::arrange(experiment, model, condition)

readr::write_csv(cell_means, file.path(TABLE_DIR, "cell_means.csv"))


# -----------------------------------------------------------------------------
# Mixed models
# -----------------------------------------------------------------------------

fit_lmer <- function(df, formula, model, experiment) {
  fit <- lmerTest::lmer(formula, data = df, REML = TRUE)
  outfile <- file.path(
    MODEL_DIR,
    sprintf("%s_exp%s.txt", model, experiment)
  )
  sink(outfile)
  cat("Model:", MODEL_LABELS[[model]], "\n")
  cat("Experiment:", EXPERIMENT_LABELS[[as.character(experiment)]], "\n")
  cat("Formula:", deparse(formula), "\n\n")
  print(summary(fit))
  cat("\nType III ANOVA (Satterthwaite):\n")
  print(anova(fit, type = 3))
  sink()
  fit
}

tidy_fixed <- function(fit, model, experiment, formula) {
  coefs <- summary(fit)$coefficients
  tibble::tibble(
    model = model,
    model_label = unname(MODEL_LABELS[model]),
    experiment = experiment,
    formula = deparse(formula),
    term = rownames(coefs),
    estimate = coefs[, "Estimate"],
    se = coefs[, "Std. Error"],
    df = coefs[, "df"],
    t = coefs[, "t value"],
    p = coefs[, "Pr(>|t|)"]
  )
}

contrast_to_row <- function(contrast_df, model, experiment, diagnostic, note) {
  tibble::tibble(
    model = model,
    model_label = unname(MODEL_LABELS[model]),
    experiment = experiment,
    diagnostic = diagnostic,
    contrast = as.character(contrast_df$contrast[1]),
    estimate = contrast_df$estimate[1],
    se = contrast_df$SE[1],
    df = contrast_df$df[1],
    t = contrast_df$t.ratio[1],
    p = contrast_df$p.value[1],
    note = note
  )
}

# match minus mismatch, given factor levels (mismatch, match)
match_effect <- list(`match - mismatch` = c(-1, 1))
# mismatch minus match: the surprisal penalty for mismatching
mismatch_penalty <- list(`mismatch - match` = c(1, -1))

diagnostics_exp1 <- function(fit, model) {
  local_penalty <- contrast(
    emmeans(fit, ~ local_match),
    method = mismatch_penalty
  )
  rescue <- contrast(
    emmeans(fit, ~ matrix_match | local_match),
    method = match_effect
  )
  rescue_df <- as.data.frame(rescue)
  rescue_at_mismatch <- rescue_df[rescue_df$local_match == "mismatch", , drop = FALSE]
  rescue_at_match <- rescue_df[rescue_df$local_match == "match", , drop = FALSE]

  dplyr::bind_rows(
    contrast_to_row(
      as.data.frame(local_penalty),
      model, 1,
      "local_mismatch_penalty",
      "Baseline: surprisal(-Local) - surprisal(+Local). Positive = local sensitivity."
    ),
    contrast_to_row(
      rescue_at_mismatch,
      model, 1,
      "matrix_rescue",
      "Critical: surprisal(+Matrix) - surprisal(-Matrix) at -Local. Negative = illicit rescue."
    ),
    contrast_to_row(
      rescue_at_match,
      model, 1,
      "matrix_effect_at_local_match",
      "Control: matrix match effect when the local subject already matches."
    )
  )
}

diagnostics_exp2 <- function(fit, model) {
  head_penalty <- contrast(
    emmeans(fit, ~ head_match),
    method = mismatch_penalty
  )
  trap <- contrast(
    emmeans(fit, ~ distractor_match | head_match),
    method = match_effect
  )
  trap_df <- as.data.frame(trap)
  trap_at_mismatch <- trap_df[trap_df$head_match == "mismatch", , drop = FALSE]
  trap_at_match <- trap_df[trap_df$head_match == "match", , drop = FALSE]

  dplyr::bind_rows(
    contrast_to_row(
      as.data.frame(head_penalty),
      model, 2,
      "head_mismatch_penalty",
      "Baseline: surprisal(-Head) - surprisal(+Head). Positive = structural-head sensitivity."
    ),
    contrast_to_row(
      trap_at_mismatch,
      model, 2,
      "proximity_trap",
      "Critical: surprisal(+Distractor) - surprisal(-Distractor) at -Head. Negative = linear trap."
    ),
    contrast_to_row(
      trap_at_match,
      model, 2,
      "distractor_effect_at_head_match",
      "Control: distractor match effect when the head already matches."
    )
  )
}

diagnostics_exp3 <- function(fit, model) {
  penalties <- contrast(
    emmeans(fit, ~ local_match | context_type),
    method = mismatch_penalty
  )
  penalty_df <- as.data.frame(penalties)
  coarg <- penalty_df[penalty_df$context_type == "coarg", , drop = FALSE]
  picture <- penalty_df[penalty_df$context_type == "picture_np", , drop = FALSE]

  emm_obj <- emmeans(fit, ~ local_match * context_type)
  cells <- as.data.frame(emm_obj)
  vcov_emm <- vcov(emm_obj)
  weights <- rep(0, nrow(cells))
  weights[cells$local_match == "mismatch" & cells$context_type == "coarg"] <- 1
  weights[cells$local_match == "match" & cells$context_type == "coarg"] <- -1
  weights[cells$local_match == "mismatch" & cells$context_type == "picture_np"] <- -1
  weights[cells$local_match == "match" & cells$context_type == "picture_np"] <- 1
  estimate <- sum(cells$emmean * weights)
  se <- sqrt(as.numeric(t(weights) %*% vcov_emm %*% weights))
  df_val <- mean(cells$df)
  t_val <- estimate / se
  p_val <- 2 * pt(-abs(t_val), df = df_val)

  dplyr::bind_rows(
    contrast_to_row(
      coarg, model, 3,
      "mismatch_penalty_coarg",
      "surprisal(-Local) - surprisal(+Local) in co-argument contexts."
    ),
    contrast_to_row(
      picture, model, 3,
      "mismatch_penalty_picture",
      "surprisal(-Local) - surprisal(+Local) in picture-NP contexts."
    ),
    tibble::tibble(
      model = model,
      model_label = unname(MODEL_LABELS[model]),
      experiment = 3L,
      diagnostic = "logophoric_exemption",
      contrast = "(mismatch-match | coarg) - (mismatch-match | picture-NP)",
      estimate = estimate,
      se = se,
      df = df_val,
      t = t_val,
      p = p_val,
      note = "Positive = larger local-mismatch penalty in co-argument than picture-NP (exemption)."
    )
  )
}

FORMULAS <- list(
  `1` = surprisal ~ local_match * matrix_match + (1 | item_id),
  `2` = surprisal ~ head_match * distractor_match + (1 | item_id),
  `3` = surprisal ~ local_match * context_type + (1 | item_id)
)

fits <- list()
fixed_rows <- list()
diagnostic_rows <- list()
emm_rows <- list()

for (i in seq_len(nrow(file_index))) {
  model <- file_index$model[[i]]
  experiment <- file_index$experiment[[i]]
  key <- paste0(model, "_exp", experiment)
  df <- datasets[[key]]
  formula <- FORMULAS[[as.character(experiment)]]
  message("Fitting ", MODEL_LABELS[[model]], " / experiment ", experiment)

  fit <- fit_lmer(df, formula, model, experiment)
  fits[[key]] <- fit
  fixed_rows[[key]] <- tidy_fixed(fit, model, experiment, formula)

  diagnostic_rows[[key]] <- switch(
    as.character(experiment),
    "1" = diagnostics_exp1(fit, model),
    "2" = diagnostics_exp2(fit, model),
    "3" = diagnostics_exp3(fit, model)
  )

  specs <- switch(
    as.character(experiment),
    "1" = ~ local_match * matrix_match,
    "2" = ~ head_match * distractor_match,
    "3" = ~ local_match * context_type
  )
  emm <- as.data.frame(emmeans(fit, specs))
  emm$model <- model
  emm$model_label <- unname(MODEL_LABELS[model])
  emm$experiment <- experiment
  emm_rows[[key]] <- emm
}

fixed_effects <- dplyr::bind_rows(fixed_rows) %>%
  dplyr::mutate(
    model = factor(model, levels = MODEL_ORDER),
    significant = p < 0.05
  ) %>%
  dplyr::arrange(experiment, model, term)

diagnostics <- dplyr::bind_rows(diagnostic_rows) %>%
  dplyr::mutate(
    model = factor(model, levels = MODEL_ORDER),
    significant = p < 0.05,
    supports_heuristic = dplyr::case_when(
      diagnostic == "matrix_rescue" ~ estimate < 0 & p < 0.05,
      diagnostic == "proximity_trap" ~ estimate < 0 & p < 0.05,
      diagnostic == "logophoric_exemption" ~ estimate > 0 & p < 0.05,
      diagnostic %in% c("local_mismatch_penalty", "head_mismatch_penalty") ~ estimate > 0 & p < 0.05,
      TRUE ~ NA
    )
  ) %>%
  dplyr::arrange(experiment, diagnostic, model)

readr::write_csv(fixed_effects, file.path(TABLE_DIR, "lme_fixed_effects.csv"))
readr::write_csv(diagnostics, file.path(TABLE_DIR, "diagnostics.csv"))
readr::write_csv(dplyr::bind_rows(emm_rows), file.path(TABLE_DIR, "emmeans_cells.csv"))


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

palette_two <- c("mismatch" = "#D55E00", "match" = "#0072B2")
palette_context <- c("coarg" = "#D55E00", "picture_np" = "#009E73")

theme_binding <- function() {
  ggplot2::theme_bw(base_size = 12) +
    ggplot2::theme(
      strip.background = ggplot2::element_rect(fill = "grey95", colour = NA),
      legend.position = "bottom",
      panel.grid.minor = ggplot2::element_blank()
    )
}

plot_exp1 <- function() {
  means <- cell_means %>% dplyr::filter(experiment == 1) %>%
    dplyr::mutate(
      local_match = ifelse(startsWith(condition, "+Local"), "match", "mismatch"),
      matrix_match = ifelse(endsWith(condition, "+Matrix"), "match", "mismatch")
    )
  ggplot2::ggplot(
    means,
    ggplot2::aes(x = local_match, y = mean, colour = matrix_match, group = matrix_match)
  ) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::geom_point(size = 2.4) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = ci95_lo, ymax = ci95_hi),
      width = 0.08,
      linewidth = 0.5
    ) +
    ggplot2::scale_colour_manual(values = palette_two, name = "Matrix match") +
    ggplot2::scale_x_discrete(limits = c("mismatch", "match"), labels = c("−Local", "+Local")) +
    ggplot2::facet_wrap(~ model_label, nrow = 1) +
    ggplot2::labs(
      title = "Experiment 1: Locality domains",
      subtitle = "Matrix rescue = lower surprisal for +Matrix when Local mismatches",
      x = "Local subject",
      y = "Mean surprisal at reflexive"
    ) +
    theme_binding()
}

plot_exp2 <- function() {
  means <- cell_means %>% dplyr::filter(experiment == 2) %>%
    dplyr::mutate(
      head_match = ifelse(startsWith(condition, "+Head"), "match", "mismatch"),
      distractor_match = ifelse(endsWith(condition, "+Distractor"), "match", "mismatch")
    )
  ggplot2::ggplot(
    means,
    ggplot2::aes(x = head_match, y = mean, colour = distractor_match, group = distractor_match)
  ) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::geom_point(size = 2.4) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = ci95_lo, ymax = ci95_hi),
      width = 0.08,
      linewidth = 0.5
    ) +
    ggplot2::scale_colour_manual(values = palette_two, name = "Distractor match") +
    ggplot2::scale_x_discrete(limits = c("mismatch", "match"), labels = c("−Head", "+Head")) +
    ggplot2::facet_wrap(~ model_label, nrow = 1) +
    ggplot2::labs(
      title = "Experiment 2: Structural hierarchy",
      subtitle = "Proximity trap = lower surprisal for +Distractor when Head mismatches",
      x = "Head noun",
      y = "Mean surprisal at reflexive"
    ) +
    theme_binding()
}

plot_exp3 <- function() {
  means <- cell_means %>% dplyr::filter(experiment == 3) %>%
    dplyr::mutate(
      local_match = ifelse(startsWith(condition, "+Local"), "match", "mismatch"),
      context_type = ifelse(grepl("Coarg", condition, fixed = TRUE), "coarg", "picture_np")
    )
  ggplot2::ggplot(
    means,
    ggplot2::aes(x = local_match, y = mean, colour = context_type, group = context_type)
  ) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::geom_point(size = 2.4) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = ci95_lo, ymax = ci95_hi),
      width = 0.08,
      linewidth = 0.5
    ) +
    ggplot2::scale_colour_manual(
      values = palette_context,
      name = "Context",
      labels = c("coarg" = "Co-argument", "picture_np" = "Picture-NP")
    ) +
    ggplot2::scale_x_discrete(limits = c("mismatch", "match"), labels = c("−Local", "+Local")) +
    ggplot2::facet_wrap(~ model_label, nrow = 1) +
    ggplot2::labs(
      title = "Experiment 3: Logophoric diagnostic",
      subtitle = "Exemption = smaller local-mismatch penalty in picture-NP than co-argument",
      x = "Local subject",
      y = "Mean surprisal at reflexive"
    ) +
    theme_binding()
}

plot_diagnostics <- function() {
  key <- diagnostics %>%
    dplyr::filter(
      diagnostic %in% c(
        "matrix_rescue",
        "proximity_trap",
        "logophoric_exemption",
        "local_mismatch_penalty",
        "head_mismatch_penalty"
      )
    ) %>%
    dplyr::mutate(
      diagnostic = factor(
        diagnostic,
        levels = c(
          "local_mismatch_penalty",
          "matrix_rescue",
          "head_mismatch_penalty",
          "proximity_trap",
          "logophoric_exemption"
        ),
        labels = c(
          "Exp1 local penalty",
          "Exp1 matrix rescue",
          "Exp2 head penalty",
          "Exp2 proximity trap",
          "Exp3 logophoric exemption"
        )
      )
    )
  ggplot2::ggplot(
    key,
    ggplot2::aes(x = estimate, y = model_label, colour = significant)
  ) +
    ggplot2::geom_vline(xintercept = 0, linewidth = 0.4, colour = "grey50") +
    ggplot2::geom_point(size = 2.6) +
    ggplot2::geom_errorbarh(
      ggplot2::aes(xmin = estimate - 1.96 * se, xmax = estimate + 1.96 * se),
      height = 0.2
    ) +
    ggplot2::scale_colour_manual(
      values = c("TRUE" = "#0072B2", "FALSE" = "grey55"),
      name = "p < .05"
    ) +
    ggplot2::facet_wrap(~ diagnostic, scales = "free_x", ncol = 1) +
    ggplot2::labs(
      title = "Key LME diagnostics across models",
      subtitle = "Error bars = 95% CI. Rescue/trap estimates < 0 support surface heuristics; exemption > 0 supports picture-NP relaxation.",
      x = "Estimate (surprisal difference)",
      y = NULL
    ) +
    theme_binding()
}

save_figure <- function(plot, name, width, height) {
  ggplot2::ggsave(
    file.path(FIGURE_DIR, paste0(name, ".pdf")),
    plot,
    width = width,
    height = height
  )
  ggplot2::ggsave(
    file.path(FIGURE_DIR, paste0(name, ".png")),
    plot,
    width = width,
    height = height,
    dpi = 150
  )
}

save_figure(plot_exp1(), "fig1_locality", 13, 4.2)
save_figure(plot_exp2(), "fig2_hierarchy", 13, 4.2)
save_figure(plot_exp3(), "fig3_logophor", 13, 4.2)
save_figure(plot_diagnostics(), "fig4_diagnostics", 8.5, 10)


# -----------------------------------------------------------------------------
# Human-readable summary
# -----------------------------------------------------------------------------

fmt_p <- function(p) {
  ifelse(p < 0.001, "< .001", sprintf("%.3f", p))
}

summary_lines <- c(
  "# LME analysis summary",
  "",
  "Each LLM was analysed separately, as specified in the research proposal.",
  "Random effect: intercept for lexical item (`item_id`).",
  "p-values: Satterthwaite approximation via lmerTest / emmeans.",
  "",
  "## Key diagnostics",
  ""
)

for (diag_name in c(
  "local_mismatch_penalty",
  "matrix_rescue",
  "head_mismatch_penalty",
  "proximity_trap",
  "mismatch_penalty_coarg",
  "mismatch_penalty_picture",
  "logophoric_exemption"
)) {
  block <- diagnostics %>% dplyr::filter(diagnostic == diag_name)
  if (nrow(block) == 0) next
  summary_lines <- c(summary_lines, paste0("### ", diag_name), "")
  for (r in seq_len(nrow(block))) {
    row <- block[r, ]
    summary_lines <- c(
      summary_lines,
      sprintf(
        "- %s: estimate = %.3f, t(%.1f) = %.2f, p = %s%s",
        row$model_label,
        row$estimate,
        row$df,
        row$t,
        fmt_p(row$p),
        ifelse(isTRUE(row$supports_heuristic), "  [supports predicted heuristic/exemption]", "")
      )
    )
  }
  summary_lines <- c(summary_lines, "")
}

writeLines(summary_lines, file.path(OUTPUT_DIR, "summary.md"))

sink(file.path(OUTPUT_DIR, "session_info.txt"))
cat("Project: llm-binding-learnability\n")
cat("Script: analysis/run_analysis.R\n\n")
print(sessionInfo())
sink()

message("Wrote tables to ", TABLE_DIR)
message("Wrote figures to ", FIGURE_DIR)
message("Done.")
