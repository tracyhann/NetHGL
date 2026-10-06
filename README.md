# [MODELNAME]

[MODELNAME] is a hierarchical graph neural network that classifies fMRI scans as concurrent remission or depression from ROI graphs. It first learns representations within atlas functional networks at ROI resolution, pools ROIs into network tokens with learned weights, and then models directed communication among the network tokens. The repository also contains the graph-construction, attribution, perturbation, repeated-measures statistical, and evaluation functions used to study the trained model.

This is a code-only release. It contains no participant records, imaging data, derived connectivity, graph files, split manifests, checkpoints, predictions, or study results.

## Method at a glance

![[MODELNAME] pipeline](docs/figures/model_pipeline.png)

**A. Graph construction.** Each resting-state fMRI scan (participant $`u`$, visit $`v`$, scan $`m`$) is preprocessed and parcellated into 450 ROIs (400 Schaefer cortical and 50 Tian subcortical) assigned to 24 functional networks. Each ROI's time series is summarized by 32 temporal features under a within-ROI and a whole-brain normalization (64 features per node), and ROIs are connected only within their own network. The scan takes the concurrent MADRS label of its visit: remitted (MADRS ≤ 10) or depressed (MADRS > 10).

**B. Network-aware hierarchical graph learning.** Two GraphSAGE layers (64→256→64) pass messages within networks; learned, scan-invariant weights pool the ROI embeddings into one embedding per network; two GATv2 layers (64→256→64) pass messages over the complete directed network graph; and a network-weighted linear ensemble produces the depression logit. Gradient-weighted attention on the final network edges supports the downstream interpretation.

## Repository contents

- `src/tms_gnn/graph`: atlas-agnostic parcellation, pairwise FC, 64-dimensional ROI features, and directed graph construction.
- `src/tms_gnn/models`: the primary hierarchical GraphSAGE/ROI-pooling/GATv2 model.
- `src/tms_gnn/training`: participant-level splitting, binary metrics, threshold selection, early stopping, and prediction.
- `src/tms_gnn/interpretability`: gradient×attention attribution, network/dyadic balance, and removal/retain-only perturbations.
- `src/tms_gnn/analysis`: clustered logistic regression, treatment moderation, random-intercept mixed models, and multiplicity correction.
- `examples/synthetic_workflow.py`: an end-to-end run using generated arrays only.
- `docs/`: method equations, the input data schema, and the pipeline figure.

See [Methods](docs/methods.md) for equations and [Data schema](docs/data_schema.md) for the caller-supplied input contract.

## Installation

Create a clean Python 3.10 or newer environment, then install the full package:

```bash
python -m pip install -e ".[all]"
```

For development (linting):

```bash
python -m pip install -e ".[all,dev]"
ruff check .
```

The base install omits PyTorch Geometric and statsmodels. Use the `gnn` extra for model execution, the `analysis` extra for statistical models, or `all` for both.

## Quick start with synthetic data

```bash
python examples/synthetic_workflow.py
```

The example generates two small ROI time-series graphs, runs the hierarchical model, differentiates the depression logit with respect to final-layer GAT attention, reverses the sign to remission orientation, and reports outgoing-minus-incoming network balance. It does not read or write study data.

The central library calls are:

```python
from tms_gnn.config import GraphConfig, ModelConfig
from tms_gnn.graph import build_graph
from tms_gnn.models import HierarchicalBrainGNN

graph_config = GraphConfig()
graph = build_graph(roi_by_time, roi_to_network, graph_config)

model_config = ModelConfig(
    input_dim=graph_config.feature_dim,
    num_rois=roi_by_time.shape[0],
    num_networks=int(roi_to_network.max()) + 1,
)
model = HierarchicalBrainGNN(model_config)
```

`roi_by_time` and `roi_to_network` are caller-owned arrays. No path, cohort, or atlas asset is encoded in the package.

## Graph construction

1. Align an atlas label vector with the grayordinate/voxel axis in the governed preprocessing environment.
2. Use `parcellate_timeseries` to average finite samples inside each non-background parcel, or supply an already parcellated ROI×time array.
3. Use `pairwise_pearson_connectivity` for pairwise-deletion Pearson FC.
4. Use `extract_roi_timeseries_features` for the temporal feature matrix.
5. Use `build_graph` to create the FC, node features, community labels, and directed ROI edge index together.

The primary graph retains all directed ROI pairs within the same network, including self-pairs. `GraphConfig` can instead allow cross-network ROI edges or select only positive/negative FC pairs. The later network graph contains every ordered network pair, including self-edges.

Project-specific CIFTI atlas alignment is intentionally not embedded: users must provide labels already aligned to their imaging axis. This prevents silent atlas mismatch and keeps licensed or governed atlas assets outside the release.

## Model output and interpretation

The binary model emits a single depression logit. Higher values indicate greater modeled evidence for depression; the corresponding probability is `sigmoid(logit)`. Remission-oriented attribution is therefore the negative of depression-logit attribution.

The forward pass returns:

- `logit`: one graph-level depression logit.
- `roi_pool_weights`: learned ROI weights, normalized within each network.
- `network_embeddings`: final network tokens.
- `network_attention`: final GATv2 edge attention.
- `network_edge_index`: the directed batched network edges aligned with attention.

Attention alone is not treated as feature importance. The released analysis uses attention multiplied by the derivative of the selected logit with respect to attention. Network balance is outgoing minus incoming attribution after removing self-edges; directed dyadic balance is A→B minus B→A.

These quantities characterize the trained model's prediction mechanism. They are not estimates of biological causality or treatment mechanism.

## Training and evaluation

`participant_level_split` performs seeded random splitting on unique participants and then maps every repeated graph back to the same partition. The default fractions give 29 training, 6 validation, and 7 test participants for a 42-participant cohort. It does not balance age or sex by default. An optional participant-level stratification variable can be supplied when the cohort supports it.

The training helper uses AdamW (learning rate 0.001, weight decay 0.005), batch size 16, binary cross-entropy with logits, optional positive-class weighting, no dropout, at most 50 epochs, and early stopping with patience 15. By default the checkpoint and its decision threshold maximize the validation minimum class recall (`selection_metric="min_class_recall"`); `selection_metric="validation_loss"` selects on validation loss instead. Apply the returned validation threshold, unchanged, to the test set. Reported metrics include accuracy, balanced accuracy, ROC AUC, specificity, recall, precision, F1, and a two-class confusion matrix.

No research split or seed list is included. Supply and document release-appropriate seeds in your own run configuration.

## Statistical analyses

The analysis package exposes:

- `fit_clustered_logistic`: concurrent binary remission associations with participant-clustered sandwich standard errors.
- `fit_treatment_moderation_logistic`: balance×treatment interaction plus treatment-specific simple slopes.
- `fit_random_intercept_lme`: Gaussian repeated-outcome analysis with a participant random intercept.
- `benjamini_hochberg`, `holm_adjust`, and `max_stat_fwer`: multiplicity control.
- `permute_participant_blocks`: participant-level label permutation that preserves repeated graphs.

Predictors are not automatically standardized. If effects should be reported per standard deviation, estimate scale parameters from the intended analysis population, divide continuous predictors by those SDs, and record that choice. Mean-centering is not required to express a one-SD slope, but it changes the interpretation of lower-order terms in interaction models.

## Perturbation analysis

`component_mask` selects a target network's outgoing, incoming, or complete incident non-self edge component. `apply_perturbation` supports:

- `remove`: delete the selected component to test necessity.
- `retain`: keep only the selected component and all self-edges to test sufficiency.

Target masks should be compared with size- and topology-matched network controls. Positive signed effects mean that target removal is more damaging than matched removal, or that the target retained component preserves the original output better than the matched retained component. Use participant-cluster bootstrap intervals and max-stat FWER when screening a family of components.

## Configuration

Copy `configs/example.yaml` and replace only the angle-bracket placeholders. Keep private paths and identifiers outside version control. The Python APIs do not require this YAML file; it is a documented run-config template for downstream scripts.

## Citation

Replace this block with the final publication citation before archival release:

```text
[AUTHORS]. [MODELNAME]: [MANUSCRIPT TITLE]. [VENUE], [YEAR].
```

## License

Released under the MIT License. See [LICENSE](LICENSE).

