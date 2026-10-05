# Model artefact

The service loads `models/router.joblib` (about 205 KB, scikit-learn 1.8.0).

The artefact is **not committed**. Its TF-IDF vocabulary is derived from Kestrel customer request
text, which ops policy §10 does not allow in a public repository. You can obtain it in either of
two ways:

- copy the privately shared `router.joblib` into this folder; or
- rebuild it from the assignment data placed in the project root:

  ```bash
  python -m scripts.train
  ```

  This uses `reports/selected_spec.json`, or `DEFAULT_SPEC` if that file is absent.

Set `KESTREL_MODEL_PATH` to load the artefact from another location.
