"""Confronto statistico Sani (HE) vs Malati (PE): boxplot e test di Mann-Whitney."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

METADATA_COLUMNS = {"nome_file", "paziente_id", "tipo_test", "visita",
                    "percorso_file", "angolo_rotazione", "Gruppo", "Classe"}
PALETTE = {"Sani (HE)": "#2ecc71", "Malati (PE)": "#e74c3c"}


def stato_paziente(nome_file):
    """Ricava il gruppo dal nome del file (schema V##HE## / V##PE##)."""
    match = re.search(r"V\d+(HE|PE)\d+", str(nome_file), re.IGNORECASE)
    if match:
        return "Sani (HE)" if match.group(1).upper() == "HE" else "Malati (PE)"
    return "Non classificato"


def feature_columns(df):
    return [c for c in df.columns
            if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]


def mann_whitney_table(df):
    """Test di Mann-Whitney bilaterale per ogni feature, ordinato per p-value."""
    df = df.copy()
    df["Gruppo"] = df["nome_file"].apply(stato_paziente)
    rows = []
    for feature in feature_columns(df):
        sani = df.loc[df["Gruppo"] == "Sani (HE)", feature].dropna()
        malati = df.loc[df["Gruppo"] == "Malati (PE)", feature].dropna()
        _, p_val = mannwhitneyu(sani, malati, alternative="two-sided")
        rows.append({
            "Feature": feature,
            "Mediana Sani (HE)": sani.median(),
            "Mediana Malati (PE)": malati.median(),
            "P-Value": p_val,
            "Significativa (p<0.05)": p_val < 0.05,
        })
    return pd.DataFrame(rows).sort_values("P-Value").reset_index(drop=True)


def save_boxplots(df, output_dir, chunk_size=6):
    """Salva i boxplot Sani vs Malati in griglie da `chunk_size` feature."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["Gruppo"] = df["nome_file"].apply(stato_paziente)
    df = df[df["Gruppo"] != "Non classificato"]
    features = feature_columns(df)

    sns.set_theme(style="whitegrid", palette="muted")
    paths = []
    for i in range(0, len(features), chunk_size):
        chunk = features[i:i + chunk_size]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Confronto Sani vs Malati - Gruppo {i // chunk_size + 1}",
                     fontsize=18, fontweight="bold", y=1.02)
        axes_flat = axes.flatten()
        for idx, feature in enumerate(chunk):
            ax = axes_flat[idx]
            sns.boxplot(data=df, x="Gruppo", y=feature, hue="Gruppo", legend=False, ax=ax,
                        palette=PALETTE, width=0.5, showmeans=True,
                        meanprops={"marker": "o", "markerfacecolor": "white",
                                   "markeredgecolor": "black", "markersize": "7"})
            sns.stripplot(data=df, x="Gruppo", y=feature, ax=ax,
                          color="black", alpha=0.4, jitter=True)
            ax.set_title(feature, fontsize=12, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Valore")
        for j in range(len(chunk), len(axes_flat)):
            fig.delaxes(axes_flat[j])
        plt.tight_layout()
        path = output_dir / f"boxplot_gruppo{i // chunk_size + 1}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
