"""Label-free evaluation (retrieval, probes, domain shift, identity null) and evaluation levels.

    from nflbowl.evaluation import evaluate_representation
    res = evaluate_representation("F4", E, meta, phys, folds, cfg.science, cfg.evaluation, level="standard")
"""

from .domain_diagnostics import (collapse_indicators, cross_domain_alignment, domain_diagnostics, domain_mean_gap, embedding_summary,
                                 grouped_domain_auc, positive_pair_coverage)
from .evaluator import EvalResult, evaluate_representation, evaluation_row, inductive_fold_retrieval
from .null_tests import shuffled_identity_null
from .paired_inductive import (CandidatePoolMismatch, InductiveComparison, athlete_retrieval_scores, coverage_table, paired_bootstrap)
from .probes import domain_shift, probe_r2
from .retrieval import cross_domain_retrieval, matched_retrieval, within_domain_retrieval

__all__ = ["CandidatePoolMismatch", "EvalResult", "InductiveComparison", "athlete_retrieval_scores", "collapse_indicators", "coverage_table",
           "cross_domain_alignment", "cross_domain_retrieval", "domain_diagnostics", "domain_mean_gap", "domain_shift", "embedding_summary",
           "evaluate_representation", "evaluation_row", "grouped_domain_auc", "inductive_fold_retrieval", "matched_retrieval", "paired_bootstrap",
           "positive_pair_coverage", "probe_r2", "shuffled_identity_null", "within_domain_retrieval"]
