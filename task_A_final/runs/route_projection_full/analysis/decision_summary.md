# Gap-Level Decision Summary

- 1/8 official MAE: 84.7593 m
- 1/16 official MAE: 158.6299 m
- Global official threshold: 139.6601 m
- Global shape threshold: 44.8576 m
- high_official_high_shape: 36786
- high_official_low_shape: 25254
- low_official_high_shape: 25254
- low_official_low_shape: 160865

## Suggested Focus
- `high_official_high_shape`: prioritize path/routing errors.
- `high_official_low_shape`: prioritize phase/timing errors.
- `low_official_high_shape`: inspect metric-cheat or unrealistic path geometry.
