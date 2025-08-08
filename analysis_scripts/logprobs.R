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
    mutate(
      model_logprobs = map(model_logprobs, \(l) {
        l |>
          str_replace_all("'", '"') |>
          fromJSON()
      }),
      message = map(message, fromJSON),
      message_history = message_history |>
        replace_na("[]") |>
        map(fromJSON)
    )

  logprobs_msglens <- logprobs_cleaned$message |>
    map_int(\(m) {
      m$text |>
        replace_na("") |>
        str_c(collapse = " ") |>
        str_count("\\S+")
    })
  logprobs_ctxlens <- logprobs_cleaned$message_history |>
    map_int(\(m) {
      if (length(m) == 0) {
        return(0L)
      }

      m |>
        list_rbind() |>
        pull(text) |>
        replace_na("") |>
        str_c(collapse = " ") |>
        str_count("\\S+")
    })

  logprobs_res <- logprobs_cleaned$model_logprobs |>
    map(\(l) {
      l |>
        t() |>
        as_tibble()
    }) |>
    list_rbind() |>
    unnest(cols = everything())
  logprobs_res_out <- logprobs_res |>
    select(order(colnames(logprobs_res))) |>
    rowwise() |>
    mutate(
      across(everything(), exp),
      across(everything(), \(x) x / sum(c_across(everything())))
    )

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
    ) |>
    mutate(
      message_length = logprobs_msglens,
      context_length = logprobs_ctxlens
    )

  logprobs_combined
}

get_all_logprobs <- function(model_name) {
  yoked <- read_csv(here(OUTPUT_LOC, glue("yoked_{model_name}_logprobs.csv"))) |>
    logprobs_to_long()
  shuffled <- read_csv(here(OUTPUT_LOC, glue("shuffled_{model_name}_logprobs.csv"))) |>
    logprobs_to_long()
  backward <- read_csv(here(OUTPUT_LOC, glue("backward_{model_name}_logprobs.csv"))) |>
    logprobs_to_long() |>
    mutate(
      orig_trialNum = 71 - orig_trialNum,
      orig_repNum = 5 - orig_repNum
    )
  ablated <- read_csv(here(OUTPUT_LOC, glue("ablated_{model_name}_logprobs.csv"))) |>
    logprobs_to_long()
  other_within <- read_csv(here(OUTPUT_LOC, glue("wrong_within_{model_name}_logprobs.csv"))) |>
    logprobs_to_long() |>
    mutate(condition = "other-within")
  other_across <- read_csv(here(OUTPUT_LOC, glue("wrong_across_{model_name}_logprobs.csv"))) |>
    logprobs_to_long() |>
    mutate(condition = "other-across")

  bind_rows(
    yoked,
    shuffled,
    backward,
    ablated,
    other_within,
    other_across
  ) |>
    mutate(
      type = glue("model_{str_extract(model_name, '^[a-z]+') |> str_to_lower()}"),
      condition = factor(condition, levels = c(
        "yoked", "shuffled", "backward",
        "ablated", "other-within", "other-across"
      ))
    )
}
