# Presets

Presets are *recommended parameter sets* for a given embodiment. They are not
mandatory — they only pre-fill defaults under `recommended_rules`. Any value you
set explicitly in your own `cleaning_config.yaml` overrides the preset.

Use one via:

```yaml
preset: galaxea_r1pro_dual_arm
```

or on the CLI: `lerobot-cleaner run ./ds --preset galaxea_r1pro_dual_arm --yes`.

| Preset | Embodiment |
|--------|-----------|
| `galaxea_r1pro_dual_arm` | Galaxea R1Pro, 16-dim dual-arm |
| `galaxea_r1pro_single_arm` | Galaxea R1Pro, single arm |

Contribute new presets via PR.
