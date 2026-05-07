# Gap-Level Decision Summary

- 1/8 official MAE: 78.0982 m
- 1/16 official MAE: 137.8786 m
- Global official threshold: 124.4890 m
- Global shape threshold: 39.2861 m
- high_official_high_shape: 31326
- high_official_low_shape: 30714
- low_official_high_shape: 30714
- low_official_low_shape: 155405

## Suggested Focus
- `high_official_high_shape`: prioritize path/routing errors.
- `high_official_low_shape`: prioritize phase/timing errors.
- `low_official_high_shape`: inspect metric-cheat or unrealistic path geometry.
