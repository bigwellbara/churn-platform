import numpy as np

from churn.training.evaluate import compute_metrics, find_best_threshold, save_confusion_matrix


def test_perfect_predictions_score_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, p, threshold=0.5)
    assert m["f1"] == m["roc_auc"] == m["pr_auc"] == m["accuracy"] == 1.0
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 0, 0, 2)


def test_constant_predictor_has_zero_f1_without_crashing():
    y = np.array([0, 0, 0, 1])
    m = compute_metrics(y, np.zeros(4), threshold=0.5)
    assert m["f1"] == 0.0
    assert m["roc_auc"] == 0.5


def test_best_threshold_beats_default_on_imbalanced_scores():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.25).astype(int)
    # churners score higher on average, but most scores sit below 0.5
    p = np.clip(0.15 + 0.2 * y + rng.normal(0, 0.1, y.size), 0, 1)
    t = find_best_threshold(y, p)
    assert 0 < t < 0.5
    assert compute_metrics(y, p, t)["f1"] > compute_metrics(y, p, 0.5)["f1"]


def test_save_confusion_matrix_writes_png(tmp_path):
    path = save_confusion_matrix([0, 1, 1], [0.2, 0.7, 0.4], 0.5, tmp_path / "cm.png", "t")
    assert path.exists() and path.stat().st_size > 0
