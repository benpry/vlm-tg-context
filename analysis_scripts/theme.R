theme_set(theme_bw() +
  theme(panel.grid = element_blank()))

COL_SCALE <- scale_color_manual(values = c(
  "human_original" = "#f56942",
  "human_naive" = "#e89f46",
  "model_qwen" = "#32d97a",
  "model_gemma" = "#2cd9e6",
  "model_llama" = "#4388e8"
))

make_accuracy_plot <- function(df) {
  df |>
    mutate(matcher_repNum = matcher_repNum + 1) |>
    ggplot(aes(x = matcher_repNum, y = accuracy, col = type)) +
    geom_hline(yintercept = 1 / 12, lty = "dashed") +
    geom_point(position = position_jitter(width = .2), alpha = .1) +
    stat_summary(
      aes(group = interaction(gameId, type)),
      fun = mean, geom = "line", alpha = .2
    ) +
    # geom_line(aes(group = interaction(gameId, tangram, type)), alpha = .2) +
    geom_smooth(method = "glm", formula = y ~ log(x)) +
    scale_x_continuous(breaks = 1:6) +
    COL_SCALE +
    labs(x = "Repetition number", y = "Accuracy", col = "Type")
}
