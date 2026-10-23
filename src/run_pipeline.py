"""Esegue l'intera pipeline: preprocessing -> feature -> statistica.

Uso:
    python src/run_pipeline.py --input data/raw --output results

La cartella di input deve contenere le scansioni delle spirali (.png/.jpg)
nominate secondo lo schema V##HE## / V##PE## (es. V01HE01.png).
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from analysis import mann_whitney_table, save_boxplots
from features import process_and_extract_features
from preprocessing import process_image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def sort_key(path):
    """Ordina per soggetto, gruppo e numero del disegno."""
    nome = path.stem.replace("debug_", "")
    match = re.search(r"V(\d+)(HE|PE)(\d+)", nome, re.IGNORECASE)
    if match:
        return (int(match.group(1)), match.group(2).upper(), int(match.group(3)))
    return (9999, nome, 0)


def list_images(folder):
    return sorted((f for f in Path(folder).iterdir()
                   if f.is_file() and not f.name.startswith(".")
                   and f.suffix.lower() in SUPPORTED_EXTENSIONS), key=sort_key)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="cartella con le scansioni originali")
    parser.add_argument("--output", default="results", help="cartella dei risultati")
    parser.add_argument("--plots", type=int, default=0,
                        help="salva i grafici di linearizzazione per le prime N immagini")
    args = parser.parse_args()

    out = Path(args.output)
    preproc_dir = out / "preprocessed"
    plots_dir = out / "linearizzazione"
    preproc_dir.mkdir(parents=True, exist_ok=True)
    if args.plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Preprocessing
    images = list_images(args.input)
    print(f"Preprocessing di {len(images)} immagini...")
    for image_path in images:
        try:
            angle, _, final_size = process_image(image_path, preproc_dir)
            print(f"  {image_path.name:<20} rotazione {angle:>7.2f} gradi, dimensione finale {final_size}")
        except Exception as exc:  # una scansione illeggibile non blocca le altre
            print(f"  ERRORE su {image_path.name}: {exc}")

    # 2. Estrazione delle feature
    print("\nEstrazione delle feature...")
    records = []
    for idx, path in enumerate(list_images(preproc_dir)):
        save_plot = plots_dir / f"{path.stem}.png" if idx < args.plots else None
        try:
            feats = process_and_extract_features(path, save_plot=save_plot)
        except Exception as exc:
            print(f"  ERRORE su {path.name}: {exc}")
            continue
        if feats is None:
            print(f"  scartata (tracciato troppo corto): {path.name}")
            continue
        records.append({"nome_file": path.name, **feats})

    if not records:
        print("Nessuna feature estratta.")
        return

    df = pd.DataFrame(records)
    df.to_csv(out / "features.csv", index=False)
    print(f"  {len(df)} immagini -> {out / 'features.csv'}")

    # 3. Statistica
    stats = mann_whitney_table(df)
    stats.to_csv(out / "mann_whitney.csv", index=False)
    save_boxplots(df, out / "boxplot")
    print(f"\nTest di Mann-Whitney -> {out / 'mann_whitney.csv'}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
