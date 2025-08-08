logprobs_to_long <- function(logprobs) {
  if (!"orig_trialNum" %in% colnames(logprobs)) {
    logprobs <- logprobs |>
      mutate(
        orig_trialNum = trialNum,
        orig_repNum = repNum,
        matcher_trialNum = trialNum,
        matcher_repNum = repNum
      )
  }

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
    select(
      gameId, orig_trialNum, orig_repNum,
      matcher_trialNum, matcher_repNum, target, condition
    ) |>
    cbind(logprobs_res_out) |>
    pivot_longer(
      cols = c(A:L),
      names_to = "tangram",
      values_to = "logprob"
    ) |>
    filter(target == tangram) |>
    rename(accuracy = logprob) |>
    select(
      gameId, condition, orig_trialNum, orig_repNum,
      matcher_trialNum, matcher_repNum, target, accuracy
    )

  logprobs_combined
}
