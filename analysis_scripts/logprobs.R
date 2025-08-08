logprobs_to_long <- function(logprobs) {
  logprobs_cleaned <- logprobs |>
    mutate(model_logprobs = map(model_logprobs, \(l) {
      l |>
        str_replace_all("'", '"') |>
        fromJSON()
    }))
  logprobs_res <- logprobs_cleaned$model_logprobs |>
    map(\(l) {
      l |>
        t() |>
        as_tibble()
    }) |>
    list_rbind()

  logprobs_res_out <- logprobs_res |>
    select(order(colnames(logprobs_res))) |>
    rowwise() |>
    mutate(across(everything(), exp))
  logprobs_combined <- logprobs_cleaned |>
    select(gameId, trialNum, repNum, target, condition) |>
    cbind(logprobs_res_out) |>
    pivot_longer(
      cols = c(A:L),
      names_to = "tangram",
      values_to = "logprob"
    ) |>
    filter(target == tangram) |>
    rename(accuracy = logprob) |>
    select(gameId, condition, trialNum, repNum, target, accuracy)

  logprobs_combined
}
