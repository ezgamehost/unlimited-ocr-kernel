use uocr::generation::{GenerationOptions, select_token};

#[test]
fn windowed_ngram_blocks_only_recent_matching_prefixes() {
    let options = GenerationOptions {
        max_tokens: 10,
        ngram_size: 3,
        ngram_window: 6,
    };
    let history = [1, 2, 3, 9, 1, 2];
    let mut scores = vec![0.; 10];
    scores[3] = 10.;
    scores[4] = 9.;
    assert_eq!(select_token(&history, &scores, &options).unwrap(), 4);
    let short = GenerationOptions {
        ngram_window: 3,
        ..options
    };
    assert_eq!(select_token(&history, &scores, &short).unwrap(), 3);
}

#[test]
fn greedy_ties_choose_first_and_invalid_logits_fail() {
    let options = GenerationOptions {
        max_tokens: 10,
        ngram_size: 0,
        ngram_window: 0,
    };
    assert_eq!(select_token(&[], &[1., 3., 3.], &options).unwrap(), 1);
    assert!(select_token(&[], &[f32::NAN], &options).is_err());
    assert!(select_token(&[], &[f32::NEG_INFINITY], &options).is_err());
}
