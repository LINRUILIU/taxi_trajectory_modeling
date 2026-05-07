# Gap-Level Decision Summary

- 1/8 official MAE: 84.1953 m
- 1/16 official MAE: 147.4253 m
- Global official threshold: 133.3244 m
- Global shape threshold: 44.1633 m
- high_official_high_shape: 34576
- high_official_low_shape: 27464
- low_official_high_shape: 27464
- low_official_low_shape: 158655

## Suggested Focus
- `high_official_high_shape`: prioritize path/routing errors.
- `high_official_low_shape`: prioritize phase/timing errors.
- `low_official_high_shape`: inspect metric-cheat or unrealistic path geometry.
