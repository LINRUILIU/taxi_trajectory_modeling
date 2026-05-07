# Algorithm Notes

- `pchip_only`: safest Phase 0 baseline; zero OSM dependency.
- `b28_compat`: legacy wrapper to stabilize behavior before deeper module split.
- `route_projection`: project long-gap base interpolation onto routed polyline with monotonic `s`.
- `route_s`: blend projected `s_base` with uniform route-time `s_uniform`.
- `selector_mix`: use `b28_compat` as base candidate, `route_projection` as route candidate, then apply a learned gap selector with fixed threshold to decide which gaps should switch.
