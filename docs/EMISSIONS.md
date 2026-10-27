# Main-engine emission calculations

## Model and scope

The kernel estimates propulsion main-engine emissions. Auxiliary engines and boilers are outside this kernel. Power, speed, draft and emission factors are supplied for each integration interval.

    LF_raw = (V / Vref)^b * (T / T_SEIM)^a
    LF = clip(LF_raw, 0, load_cap)
    E_p_g = power_kW * LF * LLAF_p(LF) * EF_p_g_kWh * dt_seconds / 3600

SEIM sets T = T_SEIM; VSEIM uses visual draft; REFERENCE uses supplied reference draft. The canonical baseline field is seim_draft_m. The legacy design_draft_m alias is accepted only when it does not conflict with the canonical field.

Exponents, load limits, emission factors and low-load tables require configuration. Their sources must be documented.

## Low-load adjustment

Above the configured threshold, LLAF equals one. Below it, the supplied table is linearly interpolated with endpoint clipping. Zero load yields zero propulsion emissions.

All scenarios use the same table and rule. Their numerical LLAF values can differ because draft changes the calculated load.

## Integration and outputs

Calculate each valid interval before summing passages. Mean speed cannot generally reproduce a nonlinear interval calculation.

Invalid inputs, within-passage overlap and identity conflicts produce issue records. Common-valid intervals are used across scenarios. Overlapping coverage for the same area/vessel across passages raises an error.

Interval outputs use g; aggregated outputs use kg. Results include passage emissions, area totals, vessel-type totals, source shares, rankings and model comparisons.

The inventory comparison uses:

    relative_difference_pct = 100 * (SEIM - VSEIM) / SEIM

A positive value denotes a lower VSEIM estimate. The passage comparison also retains explicitly named signed VSEIM-minus-SEIM columns and reduction_pct with the convention above. Percentages with zero denominators are empty.

## Reference scenario

The reference calculation requires reference_draft_m, reference_source and reference_kind for every valid interval.

The independent-validation command also requires voyage-level verified-draft records with declared manual-verification, authority-record, operator-record or survey provenance. It compares identical complete passages across all three scenarios. The benchmark remains a model-derived estimate conditional on verified draft.
