# Repository contents

The source archive contains Python modules, configuration schemas, documentation, tests, a CI workflow and three explicitly listed manuscript illustrations in assets/.

It excludes local configuration, credentials, raw footage, AIS records, vessel archives, model weights, generated results and temporary files. The audit permits defined text formats and the three named illustrations, and checks common credential and personal-path patterns.

## Checks and archive

~~~bash
python -m unittest discover -s tests -v
python scripts/check_release.py
python scripts/check_release.py --zip ../vseim-research.zip
~~~

The archive target must be outside the repository and must not already exist. Only audited source files enter the archive.

Example scientific parameters and camera calibration fields are blank. Transport and render settings are software defaults. Each dataset, model and third-party dependency follows its own access and licensing conditions.
