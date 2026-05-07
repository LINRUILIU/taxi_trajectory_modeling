# Onsite Checklist

1. Confirm `test_input_8.pkl` and `test_input_16.pkl` are readable.
2. Primary submission path: run `selector_mix` for both datasets.
3. Check selector metadata for selection rate, probability quantiles, NaN count, and clip count.
4. Run `scripts/smoke_check.py` without GT/OSM on selector outputs.
5. If selector fails or drifts badly, immediately fall back to `b28_compat`.
6. Run `scripts/make_submission.py` for the primary selector bundle.
7. Prepare a second `scripts/make_submission.py` bundle for the fallback `b28_compat` outputs.
8. Verify `pred_test_8.pkl` and `pred_test_16.pkl` exist in the chosen submission bundle.

## Rollback Triggers

- selector selection rate is far outside the expected OOF/full-val band
- selector metadata reports unexpected feature clipping or large NaN counts
- selector model or feature schema file mismatch
- selector output contains NaN/Inf or changes known points
- route candidate debug generation fails
