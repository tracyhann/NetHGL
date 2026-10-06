"""Tidy wrappers for the project's repeated-measures statistical models."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError as error:
    raise ImportError(
        "Statistical model functions require statsmodels. "
        "Install the project with: pip install -e '.[analysis]'"
    ) from error


_COLUMN_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ModerationResult:
    """Interaction-model coefficients and treatment-specific balance slopes."""

    coefficients: pd.DataFrame
    simple_slopes: pd.DataFrame


def _validated_frame(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    participant_col: str,
) -> pd.DataFrame:
    columns = [outcome, *predictors, participant_col]
    if not all(_COLUMN_NAME.fullmatch(column) for column in columns):
        raise ValueError("column names must contain only letters, numbers, and underscores")
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    clean = data[columns].dropna().copy()
    if clean.empty:
        raise ValueError("no complete observations remain for the model")
    if clean[participant_col].nunique() < 2:
        raise ValueError("at least two participants are required")
    return clean


def _tidy_result(
    result,
    *,
    logistic: bool,
    n_observations: int,
    n_participants: int,
    formula: str,
    terms: list[str] | None = None,
) -> pd.DataFrame:
    selected_terms = terms or list(result.params.index)
    confidence = result.conf_int().loc[selected_terms]
    frame = pd.DataFrame(
        {
            "term": selected_terms,
            "estimate": result.params.loc[selected_terms].to_numpy(dtype=float),
            "std_error": result.bse.loc[selected_terms].to_numpy(dtype=float),
            "statistic": result.tvalues.loc[selected_terms].to_numpy(dtype=float),
            "p_value": result.pvalues.loc[selected_terms].to_numpy(dtype=float),
            "ci_lower": confidence.iloc[:, 0].to_numpy(dtype=float),
            "ci_upper": confidence.iloc[:, 1].to_numpy(dtype=float),
        }
    )
    if logistic:
        frame["odds_ratio"] = np.exp(frame["estimate"])
        frame["odds_ratio_ci_lower"] = np.exp(frame["ci_lower"])
        frame["odds_ratio_ci_upper"] = np.exp(frame["ci_upper"])
    frame["n_observations"] = int(n_observations)
    frame["n_participants"] = int(n_participants)
    frame["formula"] = formula
    return frame


def _fit_clustered_formula(
    data: pd.DataFrame,
    formula: str,
    participant_col: str,
) -> tuple[object, pd.DataFrame]:
    result = smf.glm(formula, data=data, family=sm.families.Binomial()).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[participant_col]},
    )
    tidy = _tidy_result(
        result,
        logistic=True,
        n_observations=len(data),
        n_participants=data[participant_col].nunique(),
        formula=formula,
    )
    return result, tidy


def fit_clustered_logistic(
    data: pd.DataFrame,
    *,
    outcome: str,
    predictors: list[str],
    participant_col: str,
) -> pd.DataFrame:
    """Fit logistic regression with participant-clustered sandwich SEs."""

    if not predictors:
        raise ValueError("predictors cannot be empty")
    clean = _validated_frame(data, outcome, predictors, participant_col)
    if not set(clean[outcome].unique()).issubset({0, 1}):
        raise ValueError("logistic outcome must be coded 0/1")
    formula = f"{outcome} ~ " + " + ".join(predictors)
    _, tidy = _fit_clustered_formula(clean, formula, participant_col)
    return tidy


def fit_treatment_moderation_logistic(
    data: pd.DataFrame,
    *,
    outcome: str,
    balance_col: str,
    treatment_col: str,
    participant_col: str,
    covariates: list[str] | None = None,
) -> ModerationResult:
    """Fit remission ~ balance * treatment + covariates with clustered SEs."""

    covariate_columns = list(covariates or [])
    predictors = [balance_col, treatment_col, *covariate_columns]
    clean = _validated_frame(data, outcome, predictors, participant_col)
    if not set(clean[outcome].unique()).issubset({0, 1}):
        raise ValueError("logistic outcome must be coded 0/1")
    if not set(clean[treatment_col].unique()).issubset({0, 1}):
        raise ValueError("treatment must be coded 0/1")
    right_hand_side = f"{balance_col} * {treatment_col}"
    if covariate_columns:
        right_hand_side += " + " + " + ".join(covariate_columns)
    formula = f"{outcome} ~ {right_hand_side}"
    result, coefficients = _fit_clustered_formula(clean, formula, participant_col)

    interaction_term = f"{balance_col}:{treatment_col}"
    covariance = result.cov_params()
    balance_estimate = float(result.params[balance_col])
    interaction_estimate = float(result.params[interaction_term])
    rows = []
    for treatment_value in (0.0, 1.0):
        estimate = balance_estimate + treatment_value * interaction_estimate
        variance = (
            float(covariance.loc[balance_col, balance_col])
            + treatment_value**2 * float(covariance.loc[interaction_term, interaction_term])
            + 2
            * treatment_value
            * float(covariance.loc[balance_col, interaction_term])
        )
        std_error = float(np.sqrt(max(variance, 0.0)))
        statistic = estimate / std_error if std_error > 0 else np.nan
        p_value = float(2 * norm.sf(abs(statistic))) if np.isfinite(statistic) else np.nan
        lower = estimate - norm.ppf(0.975) * std_error
        upper = estimate + norm.ppf(0.975) * std_error
        rows.append(
            {
                "treatment_value": treatment_value,
                "estimate": estimate,
                "std_error": std_error,
                "statistic": statistic,
                "p_value": p_value,
                "odds_ratio": float(np.exp(estimate)),
                "odds_ratio_ci_lower": float(np.exp(lower)),
                "odds_ratio_ci_upper": float(np.exp(upper)),
            }
        )
    return ModerationResult(coefficients=coefficients, simple_slopes=pd.DataFrame(rows))


def fit_random_intercept_lme(
    data: pd.DataFrame,
    *,
    outcome: str,
    predictors: list[str],
    participant_col: str,
    reml: bool = False,
) -> pd.DataFrame:
    """Fit a Gaussian linear mixed model with participant random intercept."""

    if not predictors:
        raise ValueError("predictors cannot be empty")
    clean = _validated_frame(data, outcome, predictors, participant_col)
    formula = f"{outcome} ~ " + " + ".join(predictors)
    result = smf.mixedlm(formula, clean, groups=clean[participant_col]).fit(
        reml=reml,
        method="lbfgs",
    )
    terms = list(result.fe_params.index)
    return _tidy_result(
        result,
        logistic=False,
        n_observations=len(clean),
        n_participants=clean[participant_col].nunique(),
        formula=formula,
        terms=terms,
    )

