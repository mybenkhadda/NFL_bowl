"""Label-free evaluation (retrieval, probes, domain shift, identity null) and evaluation levels.

    from nflbowl.evaluation import evaluate_representation
    res = evaluate_representation("F4", E, meta, phys, folds, cfg.science, cfg.evaluation, level="standard")
"""

from .evaluator import EvalResult, evaluate_representation, evaluation_row, inductive_fold_retrieval
from .null_tests import shuffled_identity_null
from .probes import domain_shift, probe_r2
from .retrieval import cross_domain_retrieval, matched_retrieval, within_domain_retrieval

__all__ = ["EvalResult", "cross_domain_retrieval", "domain_shift", "evaluate_representation", "evaluation_row", "inductive_fold_retrieval",
           "matched_retrieval", "probe_r2", "shuffled_identity_null", "within_domain_retrieval"]
