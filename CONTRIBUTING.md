# Contributing

Keep coordinate transforms, units and grouping rules explicit. Add a unit test for each bug fix and update schema documentation when an input contract changes.

Use small artificial fixtures in tests rather than copying vessel observations. Never commit private AIS records, CCTV footage, credentials or model weights. New example images require separate rights/privacy review.

Changes to emission equations or low-load interpolation require numerical regression against an independently prepared expected result. Do not treat visual similarity of a figure as evidence of numerical equivalence.

Before submitting changes, run `python -m unittest discover -s tests -v` and `python scripts/check_release.py`. Optional GUI interactions should also be checked manually on the target desktop platform.
