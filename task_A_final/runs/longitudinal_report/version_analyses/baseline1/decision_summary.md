# Gap-Level Decision Summary

- 1/8 official MAE: 92.0430 m
- 1/16 official MAE: 170.3906 m
- Global official threshold: 154.1489 m
- Global shape threshold: 75.2109 m
- high_official_high_shape: 43043
- high_official_low_shape: 18997
- low_official_high_shape: 18997
- low_official_low_shape: 167122

## Suggested Focus
- `high_official_high_shape`: prioritize path/routing errors.
- `high_official_low_shape`: prioritize phase/timing errors.
- `low_official_high_shape`: inspect metric-cheat or unrealistic path geometry.
