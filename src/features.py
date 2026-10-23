"""Linearizzazione del tracciato ed estrazione delle 21 feature (Sezioni 5 e 6 della relazione)."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct
from scipy.signal import medfilt, welch
from scipy.stats import kurtosis
from skimage.morphology import skeletonize
from skimage.util import img_as_bool


# ============================================================================
# 1. FUNZIONI MATEMATICHE PER LE FEATURE
# ============================================================================

def katz_fd(signal):
    """Dimensione frattale di Katz, vincolata all'intervallo [1, 2]."""
    signal = np.array(signal)
    n = len(signal) - 1
    if n < 2:
        return 1.0

    L = np.sum(np.abs(np.diff(signal)))
    d = np.max(np.abs(signal - signal[0]))

    if d == 0 or L == 0 or L <= d:
        return 1.0

    log_n = np.log10(n)
    denom = log_n + np.log10(d / L)
    if denom == 0:
        return 1.0

    return float(np.clip(log_n / denom, 1.0, 2.0))


def robust_bowley_skewness(signal):
    """Skewness robusta di Bowley: (Q3 + Q1 - 2*Q2) / (Q3 - Q1), vincolata a [-1, 1]."""
    signal = np.array(signal)
    if len(signal) < 4:
        return 0.0

    q25, q50, q75 = np.percentile(signal, [25, 50, 75])
    iqr = q75 - q25
    if iqr == 0:
        return 0.0

    return float(np.clip((q75 + q25 - 2 * q50) / iqr, -1.0, 1.0))


def approximate_entropy(U, m=2, r=None):
    """Entropia approssimata (ApEn) con tolleranza r = 0.2 * std."""
    U = np.array(U)
    if r is None:
        r = 0.2 * np.std(U)
    if len(U) <= m + 1 or r == 0:
        return 0.0

    def _phi(m):
        N = len(U)
        x = np.array([U[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None, :] - x[None, :, :]), axis=-1) <= r, axis=0) / (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1)

    return np.abs(_phi(m) - _phi(m + 1))


def willison_amplitude(signal, threshold=None):
    """Ampiezza di Willison: numero di variazioni consecutive oltre la soglia."""
    if threshold is None:
        threshold = 0.1 * np.std(signal)
    return np.sum(np.abs(np.diff(signal)) > threshold)


# ============================================================================
# 2. TRACCIAMENTO DELLA SPIRALE
# ============================================================================

def traccia_spirale_esterno_interno(skel, cx, cy):
    """Ordina i pixel dello scheletro partendo dal punto più lontano dal centro.

    A ogni passo si sceglie il vicino non visitato più prossimo (intorno 3x3,
    poi 5x5 per superare piccole interruzioni).
    """
    y_coords, x_coords = np.where(skel > 0)
    if len(x_coords) == 0:
        return np.array([])

    dists = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
    start_idx = np.argmax(dists)
    start_pt = (x_coords[start_idx], y_coords[start_idx])

    visited = {start_pt}
    path = [start_pt]
    curr = start_pt

    while True:
        possible_nexts = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = curr[0] + dx, curr[1] + dy
                if (nx, ny) not in visited and 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                    if skel[ny, nx] > 0:
                        possible_nexts.append((nx, ny))

        if not possible_nexts:
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    nx, ny = curr[0] + dx, curr[1] + dy
                    if (nx, ny) not in visited and 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                        if skel[ny, nx] > 0:
                            possible_nexts.append((nx, ny))

        if not possible_nexts:
            break

        best_next = min(possible_nexts, key=lambda p: (p[0] - curr[0]) ** 2 + (p[1] - curr[1]) ** 2)
        visited.add(best_next)
        path.append(best_next)
        curr = best_next

    return np.array(path)


# ============================================================================
# 3. LINEARIZZAZIONE ED ESTRAZIONE
# ============================================================================

def process_and_extract_features(image_path, plot_graphs=False, save_plot=None):
    """Linearizza la spirale e calcola le 21 feature.

    Restituisce un dizionario {nome_feature: valore}, oppure None se
    l'immagine non è leggibile o il tracciato è troppo corto.
    """
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    skel = skeletonize(img_as_bool(binary)).astype(np.uint8)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

    h, w = image.shape
    cx, cy = w // 2, h // 2

    ordered_pts = traccia_spirale_esterno_interno(skel, cx, cy)
    if len(ordered_pts) < 50:
        return None

    ordered_pts = ordered_pts[::-1]
    x_final = ordered_pts[:, 0]
    y_final = ordered_pts[:, 1]

    # Coordinate polari e srotolamento dell'angolo
    dx = x_final - cx
    dy = y_final - cy
    r_final = np.sqrt(dx ** 2 + dy ** 2)
    theta_unwrapped = np.unwrap(np.arctan2(dy, dx))

    # Filtro di monotonicità: scarta i punti in cui il tratto torna indietro
    valid_idx = np.ones(len(theta_unwrapped), dtype=bool)
    max_theta = -np.inf
    for i in range(len(theta_unwrapped)):
        if theta_unwrapped[i] >= max_theta:
            max_theta = theta_unwrapped[i]
        else:
            valid_idx[i] = False

    theta_clean = theta_unwrapped[valid_idx]
    r_clean = r_final[valid_idx]
    x_clean = x_final[valid_idx]
    y_clean = y_final[valid_idx]

    if len(theta_clean) < 30:
        return None

    # Spirale di Archimede ideale r = m*theta + c (minimi quadrati)
    A = np.vstack([theta_clean, np.ones(len(theta_clean))]).T
    m, c = np.linalg.lstsq(A, r_clean, rcond=None)[0]
    ideal_radius = m * theta_clean + c

    # Segnale di errore radiale filtrato con mediana a 3 campioni
    error_signal = medfilt(r_clean - ideal_radius, kernel_size=3)

    # Profilo di spessore campionato sulla mappa di distanza
    pressure_signal = dist_transform[y_clean, x_clean]

    freqs, psd = welch(error_signal, fs=1.0, nperseg=min(256, len(error_signal)))
    dct_signal = dct(error_signal, type=2, norm="ortho")

    lunghezza_spirale_reale = np.sum(np.sqrt(np.diff(x_clean) ** 2 + np.diff(y_clean) ** 2))

    features = {
        # --- Ampiezza e lunghezza ---
        "error_distance_mean": float(np.mean(np.abs(error_signal))) if len(error_signal) > 0 else 0.0,
        "tremore_rms": float(np.sqrt(np.mean(error_signal ** 2))) if len(error_signal) > 0 else 0.0,
        "tremore_std": float(np.std(error_signal)) if len(error_signal) > 0 else 0.0,

        # --- Spettro (Welch) ---
        "frequenza_dominante_rad": float(freqs[np.argmax(psd)]) if len(psd) > 0 and np.max(psd) > 0 else 0.0,
        "mean_frequency_mnf_rad": float(np.sum(freqs * psd) / np.sum(psd)) if len(psd) > 0 and np.sum(psd) > 0 else 0.0,
        "psd_tremor_band_energy_rad": float(np.sum(psd)) if len(psd) > 0 else 0.0,

        # --- Forma e complessità ---
        "kurtosis": float(kurtosis(error_signal)) if len(error_signal) > 3 else 0.0,
        "skewness": robust_bowley_skewness(error_signal),
        "mean_absolute_value_mav": float(np.mean(np.abs(error_signal))) if len(error_signal) > 0 else 0.0,
        "fractal_dimension_katz": katz_fd(error_signal),
        "approximate_entropy": approximate_entropy(error_signal) if len(error_signal) > 2 else 0.0,
        "lunghezza_spirale_reale": float(lunghezza_spirale_reale),

        # --- DCT ---
        "dct_mnf": float(np.sum(np.abs(dct_signal) * np.arange(len(dct_signal))) / np.sum(np.abs(dct_signal)))
        if len(dct_signal) > 0 and np.sum(np.abs(dct_signal)) > 0 else 0.0,
        "dct_wamp": int(willison_amplitude(dct_signal)),
        "dct_fractal_dimension": katz_fd(dct_signal),

        # --- Spessore del tratto ---
        "pressione_media_px": float(np.mean(pressure_signal)) if len(pressure_signal) > 0 else 0.0,
        "pressione_std_px": float(np.std(pressure_signal)) if len(pressure_signal) > 0 else 0.0,
        "pressione_cv": float(np.std(pressure_signal) / np.mean(pressure_signal))
        if len(pressure_signal) > 0 and np.mean(pressure_signal) > 0 else 0.0,
        "pressione_range_px": float(np.ptp(pressure_signal)) if len(pressure_signal) > 0 else 0.0,
        "pressione_fractal_dimension_katz": katz_fd(pressure_signal),
        "pressione_num_variazioni_significative": int(willison_amplitude(
            pressure_signal,
            threshold=0.2 * np.std(pressure_signal) if len(pressure_signal) > 0 else 0.5,
        )),
    }

    if plot_graphs or save_plot:
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))

        axs[0].plot(theta_clean, r_clean, label="Reale", color="blue", linewidth=1.5)
        axs[0].plot(theta_clean, ideal_radius, label="Ideale", color="red", linestyle="--")
        axs[0].set_title(f"Linearizzazione: {image_path.name}")
        axs[0].set_xlabel("Angolo srotolato (monotono)")
        axs[0].set_ylabel("Raggio")
        axs[0].legend(loc="upper right")
        axs[0].grid(True)

        axs[1].plot(theta_clean, error_signal, color="purple", linewidth=1.2)
        axs[1].axhline(0, color="black", linestyle="--")
        axs[1].set_title("Tremore radiale (errore)")
        axs[1].set_xlabel("Angolo srotolato")
        axs[1].set_ylabel("Errore (px)")
        axs[1].grid(True)

        axs[2].plot(theta_clean, pressure_signal, color="green", linewidth=1.2)
        axs[2].set_title("Profilo di spessore")
        axs[2].set_xlabel("Angolo srotolato")
        axs[2].set_ylabel("Spessore (px)")
        axs[2].grid(True)

        plt.tight_layout()
        if save_plot:
            fig.savefig(save_plot, dpi=100)
        if plot_graphs:
            plt.show()
        plt.close(fig)

    return features
