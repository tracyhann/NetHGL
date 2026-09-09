# Data schema

The package does not ship a dataset loader tied to one study. Callers prepare de-identified arrays and metadata under their own governance controls.

## One graph

Required arrays:

| Name | Shape | Type | Meaning |
|---|---:|---|---|
| `roi_by_time` | `n_rois × n_timepoints` | finite float | Parcellated signal in fixed atlas order |
| `roi_to_network` | `n_rois` | integer | Contiguous network index beginning at zero |

Optional pre-parcellation arrays:

| Name | Shape | Type | Meaning |
|---|---:|---|---|
| `samples_by_grayordinate` | `n_timepoints × n_grayordinates` | float, NaN allowed | Imaging signal aligned to the atlas axis |
| `parcel_labels` | `n_grayordinates` | integer | Atlas labels on exactly the same axis; zero is background by default |

Atlas alignment is a preprocessing responsibility. The package does not infer or resample labels across brain-model axes.

## Longitudinal metadata

Statistical functions accept a pandas data frame. Column names are configurable; typical roles are:

| Role | Suggested column | Encoding |
|---|---|---|
| De-identified grouping key | `participant_id` | String or integer; never a name or source-system identifier |
| Visit/order | `visit` | Integer or categorical |
| Time | `days` | Continuous days relative to a documented reference visit |
| Concurrent remission | `remission` | `1` remitted, `0` depressed |
| Treatment | `treatment` | Binary `0/1`; document which condition is the reference |
| Age | `age` | Continuous; raw or standardized as documented |
| Sex covariate | `sex` | Analysis-defined binary/categorical coding documented by the caller |
| Network/dyadic balance | Caller-selected | Continuous, generally expressed per analysis-sample SD |

Do not place names, medical-record numbers, acquisition paths, free-text notes, or re-identification keys in graph objects or release outputs.

## Outcome orientation

The model target is binary depression (`1` depressed, `0` remitted). Statistical remission models use the inverse clinical indicator (`1` remitted, `0` depressed). Consequently:

- Model depression probability: `sigmoid(depression_logit)`.
- Remission-oriented attribution: negative depression-logit attribution.
- Logistic odds ratios above one: higher odds of concurrent remission when the outcome is coded as above.

Always verify local coding before interpreting a sign or odds ratio.

## Repeated graphs

Multiple runs or visits may belong to one participant. Keep the grouping key in private analysis metadata and:

- Assign every graph from a participant to the same train/validation/test partition.
- Cluster binary-model standard errors by participant.
- Use participant random intercepts for repeated continuous outcomes.
- Permute participant-level treatment labels as whole blocks.
- Bootstrap participants rather than treating repeated graphs as independent when participant-level inference is intended.

## Placeholder configuration

`configs/example.yaml` uses `<DATA_ROOT>`, `<OUTPUT_ROOT>`, `<ATLAS_LABELS>`, and related placeholders. Replace them only in an untracked local configuration. Never commit the populated file.

