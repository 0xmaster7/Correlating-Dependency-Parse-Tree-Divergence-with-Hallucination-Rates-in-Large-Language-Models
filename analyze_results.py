import json


def missing_dependency_error(package_name, install_name=None, extra_note=""):
    install_target = install_name or package_name
    message = [
        f"Missing required package: {package_name}",
        f"Install it with: pip install {install_target}",
    ]
    if extra_note:
        message.append(extra_note)
    raise SystemExit("\n".join(message))


try:
    import matplotlib
except ModuleNotFoundError:
    missing_dependency_error("matplotlib")

matplotlib.use("Agg")

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    missing_dependency_error("matplotlib")

try:
    import numpy as np
except ModuleNotFoundError:
    missing_dependency_error("numpy")

try:
    import pandas as pd
except ModuleNotFoundError:
    missing_dependency_error("pandas")

try:
    import spacy
except ModuleNotFoundError:
    missing_dependency_error("spacy")

try:
    from scipy import stats
except ModuleNotFoundError:
    missing_dependency_error("scipy")

try:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    missing_dependency_error(
        "sklearn",
        install_name="scikit-learn",
        extra_note="Your requirements file may also need `scikit-learn` added if it is missing there.",
    )

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise SystemExit(
        "Missing spaCy model: en_core_web_sm\n"
        "Install it with: python -m spacy download en_core_web_sm"
    )

FEATURE_COLUMNS = [
    "depth",
    "branching",
    "modifier_density",
    "passive_count",
    "modal_hedge_count",
    "subord_density",
    "neg_count",
    "ent_density",
    "avg_sent_len",
    "coord_density",
]

FEATURE_LABELS = {
    "depth": "Tree Depth",
    "branching": "Branching",
    "modifier_density": "Modifier Density",
    "passive_count": "Passive Count",
    "modal_hedge_count": "Modal/Hedge Count",
    "subord_density": "Subord Density",
    "neg_count": "Negation Count",
    "ent_density": "Entity Density",
    "avg_sent_len": "Avg Sentence Length",
    "coord_density": "Coord Density",
}

REFUSAL_PHRASES = [
    "i don't know",
    "i cannot",
    "i'm not able",
    "as an ai",
    "i don't have information",
    "cannot provide",
    "i'm not sure",
    "impossible to",
]

MODAL_LIST = {"may", "might", "could", "would"}
EPISTEMIC_ADVERB_LIST = {
    "possibly",
    "perhaps",
    "probably",
    "apparently",
    "seemingly",
    "allegedly",
    "reportedly",
}
SUBORD_DEPS = {"ccomp", "advcl", "relcl"}


def is_refusal(text):
    lower_text = text.lower()
    return any(phrase in lower_text for phrase in REFUSAL_PHRASES)


def safe_mean(values):
    return float(np.mean(values)) if values else np.nan


def safe_ttest(group_a, group_b):
    if len(group_a) < 2 or len(group_b) < 2:
        return np.nan, np.nan
    t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
    return float(t_stat), float(p_val)


def get_structural_metrics(text):
    if not text or len(text.strip()) < 10:
        return None

    doc = nlp(text)
    if len(doc) == 0:
        return None

    depths = []
    for token in doc:
        current_depth = 0
        head = token
        while head.head != head:
            current_depth += 1
            head = head.head
        depths.append(current_depth)
    max_depth = max(depths) if depths else 0

    children_counts = [len(list(token.children)) for token in doc]
    avg_branching = sum(children_counts) / len(children_counts) if children_counts else 0.0

    modifiers = [token for token in doc if token.pos_ in {"ADJ", "ADV"}]
    modifier_density = len(modifiers) / len(doc) if len(doc) > 0 else 0.0

    passive_count = len([token for token in doc if token.dep_ in {"nsubjpass", "auxpass"}])

    modal_hedge_count = sum(
        token.lower_ in MODAL_LIST or token.lower_ in EPISTEMIC_ADVERB_LIST for token in doc
    )

    sentences = list(doc.sents)
    sentence_count = len(sentences)
    subord_count = len([token for token in doc if token.dep_ in SUBORD_DEPS])
    subord_density = subord_count / sentence_count if sentence_count else 0.0

    neg_count = len([token for token in doc if token.dep_ == "neg"])

    ent_density = len(doc.ents) / len(doc) if len(doc) > 0 else 0.0

    avg_sent_len = safe_mean([len(sent) for sent in sentences]) if sentences else 0.0

    coord_density = len([token for token in doc if token.dep_ == "cc"]) / len(doc) if len(doc) > 0 else 0.0

    return {
        "depth": max_depth,
        "branching": avg_branching,
        "modifier_density": modifier_density,
        "passive_count": passive_count,
        "modal_hedge_count": modal_hedge_count,
        "subord_density": subord_density,
        "neg_count": neg_count,
        "ent_density": ent_density,
        "avg_sent_len": avg_sent_len,
        "coord_density": coord_density,
    }


def collect_records(model_name, responses):
    rows = []
    skipped_refusals = 0

    for label_name, label_value in (("truth", 0), ("lie", 1)):
        for text in responses.get(label_name, []):
            if is_refusal(text):
                skipped_refusals += 1
                continue
            metrics = get_structural_metrics(text)
            if metrics is None:
                continue
            row = {
                "model": model_name,
                "label": label_name,
                "y": label_value,
                "response_length": len(text.split()),
            }
            row.update(metrics)
            rows.append(row)

    return rows, skipped_refusals


def save_histogram_grid(df_model, model_name):
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(f"Syntactic Feature Distributions: Truth vs Hallucination - {model_name}", fontsize=16)
    axes = axes.flatten()

    truth_df = df_model[df_model["label"] == "truth"]
    lie_df = df_model[df_model["label"] == "lie"]

    for index, feature in enumerate(FEATURE_COLUMNS):
        axes[index].hist(
            truth_df[feature],
            bins=20,
            alpha=0.5,
            label="Truth",
            color="green",
            density=True,
        )
        axes[index].hist(
            lie_df[feature],
            bins=20,
            alpha=0.5,
            label="Hallucination",
            color="red",
            density=True,
        )
        axes[index].set_title(FEATURE_LABELS[feature])
        axes[index].set_xlabel("Value")
        axes[index].set_ylabel("Density")
        axes[index].legend()

    plt.tight_layout()
    plt.savefig(f"graph_{model_name.replace('/', '_')}.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_pca_plot(X_scaled, y, model_name):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 7))
    plt.scatter(
        components[y == 0, 0],
        components[y == 0, 1],
        color="green",
        alpha=0.5,
        label="Truth",
    )
    plt.scatter(
        components[y == 1, 0],
        components[y == 1, 1],
        color="red",
        alpha=0.5,
        label="Hallucination",
    )
    plt.title(f"Syntactic Fingerprint Space: Truth vs Hallucination - {model_name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.savefig(f"pca_{model_name.replace('/', '_')}.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_feature_importance_chart(importances, model_name):
    sorted_items = sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True)
    labels = [FEATURE_LABELS[name] for name, _ in sorted_items]
    values = [value for _, value in sorted_items]
    colors = ["red" if value > 0 else "green" for value in values]

    plt.figure(figsize=(12, 7))
    plt.barh(labels, values, color=colors)
    plt.gca().invert_yaxis()
    plt.title(f"Feature Importance for Hallucination Detection - {model_name}")
    plt.xlabel("Logistic Regression Coefficient")
    plt.ylabel("Feature")
    plt.savefig(f"feature_importance_{model_name.replace('/', '_')}.png", dpi=200, bbox_inches="tight")
    plt.close()


def analyze_dataset(df_model, model_name, final_stats):
    truth_df = df_model[df_model["label"] == "truth"]
    lie_df = df_model[df_model["label"] == "lie"]

    for feature in FEATURE_COLUMNS:
        truth_vals = truth_df[feature].tolist()
        lie_vals = lie_df[feature].tolist()
        _, p_val = safe_ttest(truth_vals, lie_vals)

        final_stats.append(
            {
                "Model": model_name,
                "Metric": FEATURE_LABELS[feature],
                "Avg Truth": round(safe_mean(truth_vals), 4),
                "Avg Lie": round(safe_mean(lie_vals), 4),
                "Difference": round(safe_mean(lie_vals) - safe_mean(truth_vals), 4),
                "Significant?": "YES" if pd.notna(p_val) and p_val < 0.05 else "NO",
                "P-Value": f"{p_val:.5f}" if pd.notna(p_val) else "nan",
            }
        )

    X = df_model[FEATURE_COLUMNS]
    y = df_model["y"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000)
    cv_folds = int(min(5, y.value_counts().min()))
    if cv_folds >= 2:
        auc_scores = cross_val_score(lr, X_scaled, y, cv=cv_folds, scoring="roc_auc")
        mean_auc = float(np.mean(auc_scores))
    else:
        mean_auc = np.nan

    if "response_length" in df_model.columns and cv_folds >= 2:
        X_baseline = df_model[["response_length"]].fillna(0)
        baseline_scaler = StandardScaler()
        X_baseline_scaled = baseline_scaler.fit_transform(X_baseline)
        baseline_scores = cross_val_score(
            LogisticRegression(max_iter=1000),
            X_baseline_scaled,
            y,
            cv=cv_folds,
            scoring="roc_auc",
        )
        baseline_auc = float(np.mean(baseline_scores))
    else:
        baseline_auc = np.nan

    lr.fit(X_scaled, y)
    coef_series = pd.Series(lr.coef_[0], index=FEATURE_COLUMNS)
    top_features = coef_series.abs().sort_values(ascending=False).head(3)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    rf_importances = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)

    save_histogram_grid(df_model, model_name)
    save_pca_plot(X_scaled, y.to_numpy(), model_name)
    save_feature_importance_chart(coef_series.to_dict(), model_name)

    print("\n" + "=" * 90)
    print(f"MODEL: {model_name}")
    print("=" * 90)
    if pd.notna(mean_auc):
        print(f"AUC (Logistic, {cv_folds}-fold CV): {mean_auc:.4f}")
    else:
        print("AUC (Logistic CV): not available; each class needs at least 2 samples.")
    if pd.notna(baseline_auc):
        print(f"AUC (Baseline - length only): {baseline_auc:.4f}")
        print("AUC (Random): 0.5000")
        print(f"Improvement over baseline: {mean_auc - baseline_auc:+.4f}")
    else:
        print("AUC (Baseline - length only): N/A")
    print("\nTop 3 logistic features by |coefficient|:")
    for feature_name, value in top_features.items():
        direction = "hallucination" if coef_series[feature_name] > 0 else "truth"
        print(f"  {FEATURE_LABELS[feature_name]}: {coef_series[feature_name]:.4f} ({direction} signal)")

    print("\nTop 3 random forest features:")
    for feature_name, value in rf_importances.head(3).items():
        print(f"  {FEATURE_LABELS[feature_name]}: {value:.4f}")

    summary_table = pd.DataFrame(
        [
            {
                "Metric": FEATURE_LABELS[feature],
                "Truth Mean": round(truth_df[feature].mean(), 4),
                "Lie Mean": round(lie_df[feature].mean(), 4),
            }
            for feature in FEATURE_COLUMNS
        ]
    )
    print("\nPer-metric means:")
    print(summary_table.to_string(index=False))


print("Loading your experiment data...")
with open("final_cloud_results.json", "r") as file:
    data = json.load(file)

all_rows = []
final_stats = []

for model_name, responses in data.items():
    print(f"\nAnalyzing {model_name}...")
    model_rows, skipped_refusals = collect_records(model_name, responses)
    all_rows.extend(model_rows)

    if not model_rows:
        print(f"  No usable responses remained after filtering. Skipped refusals: {skipped_refusals}")
        continue

    df_model = pd.DataFrame(model_rows)
    if df_model["y"].nunique() < 2:
        print(f"  Need both truth and lie examples for classification. Skipped refusals: {skipped_refusals}")
        continue

    print(f"  Retained {len(df_model)} responses after filtering. Skipped refusals: {skipped_refusals}")
    analyze_dataset(df_model, model_name, final_stats)

if all_rows:
    combined_df = pd.DataFrame(all_rows)
    if combined_df["y"].nunique() == 2:
        analyze_dataset(combined_df, "combined", final_stats)

print("\n" + "=" * 110)
print("FINAL SCIENTIFIC RESULTS")
print("=" * 110)
results_df = pd.DataFrame(final_stats)
if not results_df.empty:
    print(results_df.to_string(index=False))
else:
    print("No statistical results to display.")

print("\nGraphs and classifier visualizations saved to your folder.")
