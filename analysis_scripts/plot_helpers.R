theme_set(theme_bw() +
  theme(panel.grid = element_blank()))

COL_SCALE <- scale_color_manual(values = c(
  "human_original" = "#f56942",
  "human_naive" = "#e89f46",
  "model_qwen" = "#32d97a",
  "model_gemma" = "#2cd9e6",
  "model_llama" = "#4388e8"
))
