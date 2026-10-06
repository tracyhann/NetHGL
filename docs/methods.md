# Methods and equations

## 1. Parcellation and functional connectivity

Let \(X_{tg}\) be the signal at time \(t\) and grayordinate \(g\), and let \(G_r\) contain grayordinates assigned to ROI \(r\). The parcel signal is the finite-sample mean

\[
x_{rt}=\frac{1}{|G_{rt}|}\sum_{g\in G_{rt}}X_{tg},
\]

where \(G_{rt}\subseteq G_r\) contains finite values at that time. Pairwise FC is Pearson correlation between ROI time series using timepoints finite for both ROIs. Pairs below `min_overlap` or with zero variance receive zero; the diagonal is one.

## 2. ROI temporal features

Two standardized views are formed:

\[
z^{\mathrm{ROI}}_{rt}=\frac{x_{rt}-\bar{x}_{r\cdot}}{s_{r\cdot}},
\qquad
z^{\mathrm{time}}_{rt}=\frac{x_{rt}-\bar{x}_{\cdot t}}{s_{\cdot t}}.
\]

Zero SD is replaced by one, producing zeros for a constant centered vector. For each ROI in each view, the extractor calculates:

- Mean, sample SD, median, IQR, minimum, maximum, unbiased skewness, and excess kurtosis.
- Normalized autocorrelation at lags 1, 2, 3, 4, 5, 10, 20, and 30.
- Sixteen ridge-stabilized Yule–Walker autoregressive coefficients.

Each view has \(8+8+16=32\) features; concatenating the views produces the 64-dimensional study representation.

## 3. ROI graph

Each ROI is a node. In the primary configuration, the directed edge set is

\[
E_{\mathrm{ROI}}=\{(i,j): c_i=c_j\},
\]

including \(i=j\), where \(c_i\) is the atlas network assignment. The graph builder can include cross-network ROI pairs or filter pairs by FC sign. FC determines edge selection in signed variants; the GraphSAGE message-passing implementation is unweighted, matching the primary model.

For layer \(\ell\), mean-aggregating GraphSAGE has the conceptual form

\[
h_i^{(\ell+1)}=\phi\!\left(W_{\mathrm{self}}h_i^{(\ell)}+
W_{\mathrm{neigh}}\operatorname{mean}_{j\in\mathcal N(i)}h_j^{(\ell)}\right).
\]

Two layers map 64→256→64 dimensions with GELU and no dropout.

## 4. Learned ROI-to-network pooling

Let \(m_{rn}=1\) when ROI \(r\) belongs to network \(n\). A global learned logit \(a_{rn}\) is normalized only over allowed ROIs:

\[
\alpha_{rn}=\frac{m_{rn}\exp(a_{rn})}{\sum_q m_{qn}\exp(a_{qn})},
\qquad
z_n=\sum_r\alpha_{rn}h_r.
\]

Thus, weights sum to one across the ROIs in each network. The ROI order and community mapping must remain identical across graphs.

## 5. Directed network GATv2

The network graph contains every ordered pair \((n,m)\), including self-edges. Two GATv2 layers map 64→256→64 dimensions. A simplified single-head attention update is

\[
e_{nm}=a^\top\operatorname{LeakyReLU}(W_s z_n+W_t z_m),
\qquad
\alpha_{nm}=\operatorname{softmax}_{n\in\mathcal N(m)}(e_{nm}),
\]

followed by an attention-weighted source aggregation at destination \(m\). The implementation delegates the exact operator to PyTorch Geometric `GATv2Conv` with `concat=False` and no additional self-loop insertion because self-edges are already explicit. Neither attention dropout nor post-layer dropout is applied.

## 6. Graph readout

Each final network token produces a scalar depression contribution \(q_n=w^\top z_n+b\). A learned linear combination produces the graph logit

\[
\eta=\beta_0+\sum_n\beta_n q_n,
\qquad
P(Y_{\mathrm{depression}}=1)=\operatorname{sigmoid}(\eta).
\]

Training uses binary cross-entropy with logits and AdamW (learning rate 0.001, weight decay 0.005), batch size 16, and at most 50 epochs. After each epoch the decision threshold is chosen on the validation partition to maximize the smaller of the two class recalls; the checkpoint with the highest such validation minimum class recall (ties broken by lower validation loss) is kept with its threshold, and training stops early after 15 epochs without improvement. The test partition remains untouched until final evaluation. Participants are split 29/6/7 into training, validation, and test sets.

## 7. Gradient×attention attribution

For final-layer attention \(\alpha_e\) on network edge \(e\), depression-oriented attribution is

\[
A^{\mathrm{dep}}_e=\alpha_e\frac{\partial\eta}{\partial\alpha_e}.
\]

Because the model emits a depression logit, remission orientation is

\[
A^{\mathrm{rem}}_e=-A^{\mathrm{dep}}_e.
\]

With self-edges excluded, network balance is

\[
B_n=\sum_{e:n\rightarrow j}A^{\mathrm{rem}}_e-
\sum_{e:i\rightarrow n}A^{\mathrm{rem}}_e.
\]

Positive values mean greater outgoing than incoming remission-oriented contribution; negative values mean relatively greater incoming contribution. Directed dyadic balance is

\[
D_{A,B}=A^{\mathrm{rem}}_{A\rightarrow B}-A^{\mathrm{rem}}_{B\rightarrow A}.
\]

The orientation is defined by graph edge indices and the selected model logit, not by ground-truth or predicted class labels.

## 8. Perturbation: necessity and sufficiency

For a target network, component masks cover outgoing, incoming, or all incident non-self edges. Self-edges remain under every retain-only perturbation.

Removal necessity compares the output reduction under targeted removal with a size/topology-matched control:

\[
S_{\mathrm{necessity}}=\Delta_{\mathrm{target\ remove}}-
\Delta_{\mathrm{matched\ remove}}.
\]

Retain-only sufficiency compares output-reconstruction errors with reversed sign:

\[
S_{\mathrm{sufficiency}}=E_{\mathrm{matched\ retain}}-
E_{\mathrm{target\ retain}}.
\]

Positive values indicate greater target-specific necessity or sufficiency. The conclusion concerns reliance of the fitted model, not causal necessity or sufficiency in the brain.

Participant-cluster bootstrap intervals first average repeated graphs within participant and resample participants. Empirical matched-control tests use the finite-sample plus-one correction. When screening many components, max-stat permutation FWER uses the maximum statistic over the complete test family on every permutation.

## 9. Concurrent remission association

For participant \(i\) and graph/visit \(v\), a generic network-balance model is

\[
\operatorname{logit}P(R_{iv}=1)=\beta_0+\beta_1 Z(B_{iv})+
\beta_2 t_{iv}+\beta_3 Z(\mathrm{age}_i)+\beta_4\mathrm{sex}_i.
\]

`fit_clustered_logistic` fits the logistic mean model and computes participant-clustered sandwich standard errors. The caller chooses visit filtering and time origin. If \(B\) is divided by its analysis-sample SD without mean-centering, \(\exp(\beta_1)\) is the odds ratio per one-SD increase; the intercept remains anchored at raw zero.

## 10. Treatment moderation

The moderation model is

\[
\operatorname{logit}P(R_{iv}=1)=\beta_0+\beta_1 Z(B_{iv})+
\beta_2 T_i+\beta_3 Z(B_{iv})T_i+\gamma^\top C_{iv}.
\]

Here \(T=0\) is the caller-defined reference condition and \(T=1\) the comparison. The balance slope is \(\beta_1\) at \(T=0\) and \(\beta_1+\beta_3\) at \(T=1\). The same model applies to network balance or directed dyadic balance. The function returns the interaction coefficients and both simple slopes with odds ratios and confidence intervals.

## 11. Continuous repeated outcomes

For a continuous response, `fit_random_intercept_lme` fits

\[
Y_{iv}=\beta_0+\beta^\top X_{iv}+u_i+\epsilon_{iv},
\qquad u_i\sim\mathcal N(0,\sigma_u^2),
\]

with a participant random intercept. This model is distinct from the binary remission regressions.

## 12. Multiplicity and longitudinal splitting

The package provides Benjamini–Hochberg FDR, Holm FWER, and max-stat permutation FWER. Participant-level variables such as treatment are permuted as intact blocks so repeated graphs retain their within-participant dependence.

Train/validation/test splitting is seeded and random over unique participants. Repeated graphs never cross partitions. No age or sex balancing occurs unless the caller supplies an explicit participant-level stratification variable.

