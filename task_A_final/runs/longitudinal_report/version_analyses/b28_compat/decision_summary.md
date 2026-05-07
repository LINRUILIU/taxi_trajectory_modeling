# Gap-Level Decision Summary

- 1/8 official MAE: 81.7358 m
- 1/16 official MAE: 142.1400 m
- Global official threshold: 128.7163 m
- Global shape threshold: 40.5192 m
- high_official_high_shape: 32622
- high_official_low_shape: 29418
- low_official_high_shape: 29418
- low_official_low_shape: 156701

## Suggested Focus
- `high_official_high_shape`: prioritize path/routing errors.
- `high_official_low_shape`: prioritize phase/timing errors.
- `low_official_high_shape`: inspect metric-cheat or unrealistic path geometry.
