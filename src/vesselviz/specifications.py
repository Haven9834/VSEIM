"""Ship-type regression for missing technical fields using supplied coefficients."""
import math


def fill_missing(records, models, order):
    """Apply intercept + sum(coefficient * predictor) in explicit target order.

    models maps vessel type to target-specific predictors, coefficients,
    intercept and source. Existing finite positive measurements take priority.
    """
    output, issues = [], []
    for index, original in enumerate(records):
        row = dict(original)
        for target in order:
            current = row.get(target)
            if current is not None and str(current).strip():
                current = float(current)
                if not math.isfinite(current) or current <= 0:
                    raise ValueError("Invalid supplied technical field: " + target)
                continue
            model = models.get(row.get("vessel_type"), {}).get(target)
            if not model:
                issues.append(dict(record=index, field=target, code="missing_regression_model"))
                continue
            names, weights = model.get("predictors", []), model.get("coefficients", [])
            if not names or len(names) != len(weights) or not model.get("source", "").strip():
                raise ValueError("Regression requires aligned predictors, coefficients and source")
            try:
                values = [float(row[name]) for name in names]
            except (KeyError, TypeError, ValueError):
                issues.append(dict(record=index, field=target, code="missing_regression_predictor"))
                continue
            numbers = values+list(weights)+[model["intercept"]]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in numbers):
                raise ValueError("Regression inputs and coefficients must be finite")
            estimate = model["intercept"]+sum(a*b for a, b in zip(values, weights))
            if estimate <= 0:
                issues.append(dict(record=index, field=target, code="nonphysical_regression_estimate"))
                continue
            row[target] = estimate
            row[target+"_source"] = "regression: " + model["source"]
        output.append(row)
    return output, issues
