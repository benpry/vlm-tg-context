theme_set(theme_bw() +
  theme(
    panel.grid = element_blank(),
    strip.background = element_blank()
  ))

COL_SCALE <- scale_color_manual(values = c(
  "human_original" = "#f56942",
  "human_naive" = "#e89f46",
  "model_qwen" = "#32d97a",
  "model_gemma" = "#2cd9e6",
  "model_llama" = "#4388e8"
))

condition_order <- c(
  "yoked", "shuffled", "backward", "ablated",
  "other-within", "other-across", "no context"
)

make_accuracy_plot <- function(df, repnum_type = "matcher") {
  p <- df |>
    mutate(
      repnum = if (repnum_type == "original") orig_repNum + 1 else matcher_repNum + 1,
      trialnum = if (repnum_type == "original") orig_trialNum + 1 else matcher_trialNum + 1,
      condition = factor(condition, levels = condition_order)
    ) |>
    ggplot(aes(x = repnum, y = accuracy, col = type)) +
    geom_hline(yintercept = 1 / 12, lty = "dashed") +
    # geom_point(position = position_jitter(width = .2), alpha = .05) +
    stat_summary(
      aes(group = interaction(gameId, type, trialnum)),
      fun = mean, geom = "point", alpha = .05,
      position = position_jitter(width = .2)
    ) +
    stat_summary(
      aes(group = interaction(gameId, type)),
      fun = mean, geom = "line", alpha = .2
    ) +
    # geom_smooth(method = "glm", formula = y ~ log(x)) +
    geom_smooth(method = "loess") +
    stat_summary(
      fun.data = mean_cl_boot, geom = "pointrange"
    ) +
    scale_x_continuous(breaks = 1:6) +
    COL_SCALE +
    labs(
      x = glue("{str_to_sentence(repnum_type)} repetition number"),
      y = "Accuracy", col = "Type"
    )

  if (n_distinct(df$condition) > 1) {
    p <- p +
      facet_wrap(
        ~condition,
        nrow = 2,
        labeller = as_labeller(str_to_sentence)
      )
  }

  p
}
