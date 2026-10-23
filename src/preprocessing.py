"""Pipeline di preprocessing delle scansioni delle spirali di Archimede.

Ogni funzione rimuove una specifica classe di disturbo (vedi Sezione 3 della
relazione). Il codice riproduce quello del notebook `notebooks/SpiraliFINALE.ipynb`.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import dilation, disk, skeletonize
from skimage.util import img_as_bool, img_as_ubyte

TARGET_SIZE = 224


# ============================================================================
# PULIZIA DELL'IMMAGINE IN SCALA DI GRIGI
# ============================================================================

def convert_to_grayscale(image):
    """Converte un'immagine BGR in scala di grigi (Y = 0.299R + 0.587G + 0.114B)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_scan_lines(gray, dark_threshold=50, coverage=0.5):
    """Rimuove le righe orizzontali artefatto dello scanner.

    Una riga con più del `coverage` (50%) di pixel più scuri di `dark_threshold`
    è considerata un artefatto e sostituita con la media della prima riga pulita
    sopra e della prima riga pulita sotto.
    """
    result = gray.copy()
    h, w = gray.shape

    for r in range(h):
        if np.sum(gray[r, :] < dark_threshold) > w * coverage:
            r_above, r_below = r - 1, r + 1
            while r_above >= 0 and np.sum(gray[r_above, :] < dark_threshold) > w * coverage:
                r_above -= 1
            while r_below < h and np.sum(gray[r_below, :] < dark_threshold) > w * coverage:
                r_below += 1
            if r_above >= 0 and r_below < h:
                result[r, :] = ((gray[r_above].astype(int) + gray[r_below].astype(int)) // 2).astype(np.uint8)
            elif r_above >= 0:
                result[r, :] = gray[r_above]
            else:
                result[r, :] = gray[r_below]
    return result


def whiten_background(gray, threshold=230):
    """Porta a bianco puro i pixel più chiari di `threshold` (ombre, trama della carta)."""
    result = gray.copy()
    result[result > threshold] = 255
    return result


def enhance_document(gray):
    """Migliora il contrasto locale con CLAHE (clip limit 2.0, griglia 8x8)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise_image(gray):
    """Filtro gaussiano 3x3 contro il rumore ad alta frequenza."""
    return cv2.GaussianBlur(gray, (3, 3), 0)


# ============================================================================
# BINARIZZAZIONE E MORFOLOGIA
# ============================================================================

def binarize_image(gray):
    """Soglia adattiva gaussiana (finestra 51x51, C = 10)."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=51, C=10
    )


def close_gaps(binary):
    """Chiusura morfologica (ellisse 3x3): ricollega piccole interruzioni del tratto."""
    inverted = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
    return cv2.bitwise_not(closed)


def keep_main_spiral(binary):
    """Mantiene solo la componente connessa (8-connettività) di area maggiore."""
    inv = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if num_labels <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1

    mask = (labels == largest_label).astype(np.uint8) * 255
    result = np.full_like(binary, 255)
    result[mask > 0] = 0
    return result


def smooth_stroke_skeleton(binary, radius=3):
    """Lisciatura, apertura, scheletrizzazione e ridilatazione a spessore uniforme."""
    inv = cv2.bitwise_not(binary)

    # 1. Lisciatura dei contorni
    blurred = cv2.GaussianBlur(inv, (5, 5), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # 2. Apertura morfologica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

    # 3. Scheletrizzazione
    skel = skeletonize(img_as_bool(smoothed))

    # 4. Dilatazione a spessore uniforme
    dilated = dilation(skel, disk(radius))

    return img_as_ubyte(~dilated)


# ============================================================================
# ANALISI GEOMETRICA
# ============================================================================

def find_spiral_center(binary_image):
    """Trova il "centro biologico" della spirale (Sezione 4 della relazione).

    1. estremità libere dello scheletro (filtro 3x3);
    2. estremità più vicina al baricentro dell'inchiostro;
    3. cerchio ai minimi quadrati sulla parte interna della spirale;
    4. validazione del candidato entro 1.5 volte il raggio del cerchio.
    """
    inv = cv2.bitwise_not(binary_image)
    skel = skeletonize(img_as_bool(inv)).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D(skel.astype(np.float32), -1, kernel.astype(np.float32))
    endpoints_mask = (skel == 1) & (neighbor_count == 2)
    endpoint_coords = np.column_stack(np.where(endpoints_mask))

    moments = cv2.moments(inv)
    if moments["m00"] == 0:
        return None
    bary_x = moments["m10"] / moments["m00"]
    bary_y = moments["m01"] / moments["m00"]

    if len(endpoint_coords) == 0:
        return (int(bary_x), int(bary_y))

    dists = np.sqrt((endpoint_coords[:, 1] - bary_x) ** 2 + (endpoint_coords[:, 0] - bary_y) ** 2)
    nearest = np.argmin(dists)
    cy, cx = endpoint_coords[nearest]
    candidate = (int(cx), int(cy))

    skel_coords = np.column_stack(np.where(skel == 1))
    dists_all = np.sqrt((skel_coords[:, 1] - bary_x) ** 2 + (skel_coords[:, 0] - bary_y) ** 2)
    max_radius = dists_all.max()

    for fraction in [0.20, 0.25, 0.30, 0.40]:
        inner_mask = dists_all <= max_radius * fraction
        inner_coords = skel_coords[inner_mask]
        if len(inner_coords) >= 10:
            break

    if len(inner_coords) < 10:
        return candidate

    pts = inner_coords[:, ::-1].astype(np.float32)
    x, y = pts[:, 0], pts[:, 1]

    # Circonferenza x^2 + y^2 + Dx + Ey + F = 0 ai minimi quadrati
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    circle_cx = -D / 2
    circle_cy = -E / 2
    circle_r = np.sqrt(circle_cx ** 2 + circle_cy ** 2 - F)

    dist_to_circle_center = np.sqrt((candidate[0] - circle_cx) ** 2 + (candidate[1] - circle_cy) ** 2)

    if dist_to_circle_center <= circle_r * 1.5:
        return candidate

    dist_ep = np.sqrt((endpoint_coords[:, 1] - circle_cx) ** 2 + (endpoint_coords[:, 0] - circle_cy) ** 2)
    nearest_corrected = np.argmin(dist_ep)
    cy2, cx2 = endpoint_coords[nearest_corrected]
    return (int(cx2), int(cy2))


def detect_rotation_angle(binary_image):
    """Angolo del rettangolo di area minima (cv2.minAreaRect) che contiene la spirale."""
    inverted = cv2.bitwise_not(binary_image)
    coords = np.column_stack(np.where(inverted > 0))
    if len(coords) < 50:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return float(-angle)


def rotate_image(image, angle):
    """Ruota l'immagine allargando la tela per non tagliare il disegno."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        image, matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )


def crop_document(binary_image, padding=10):
    """Ritaglia attorno al disegno con un margine di `padding` pixel."""
    inverted = cv2.bitwise_not(binary_image)
    coords = cv2.findNonZero(inverted)
    if coords is None:
        return binary_image
    x, y, w, h = cv2.boundingRect(coords)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(binary_image.shape[1], x + w + padding)
    y2 = min(binary_image.shape[0], y + h + padding)
    return binary_image[y1:y2, x1:x2]


def normalize_spiral_size(binary_image, target_size=TARGET_SIZE):
    """Ridimensiona su una tela quadrata `target_size` x `target_size` (non usata nella pipeline finale)."""
    h, w = binary_image.shape
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(binary_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size), 255, dtype=np.uint8)
    x = (target_size - new_w) // 2
    y = (target_size - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def center_spiral(binary_image, center):
    """Trasla l'immagine in modo che `center` coincida con il centro della tela (non usata nella pipeline finale)."""
    h, w = binary_image.shape
    cx, cy = center
    M = np.float32([[1, 0, w // 2 - cx], [0, 1, h // 2 - cy]])
    return cv2.warpAffine(binary_image, M, (w, h), borderValue=255)


# ============================================================================
# PIPELINE PRINCIPALE
# ============================================================================

def process_image(image_path, output_dir):
    """Applica la pipeline a una scansione e salva l'immagine di controllo.

    Come nel notebook, viene salvata l'immagine `debug_<nome>` (spirale
    preprocessata con il centro trovato segnato in rosso): è l'immagine su cui
    lavora poi l'estrazione delle feature.

    Restituisce (angolo di rotazione, dimensione originale, dimensione finale).
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Impossibile leggere: {image_path}")

    original_size = (image.shape[1], image.shape[0])

    # Pulizia e contrasto
    gray = convert_to_grayscale(image)
    gray = remove_scan_lines(gray)
    gray = whiten_background(gray, threshold=230)
    enhanced = enhance_document(gray)
    denoised = denoise_image(enhanced)
    binary = binarize_image(denoised)

    # Morfologia
    binary = close_gaps(binary)
    binary = keep_main_spiral(binary)
    binary = smooth_stroke_skeleton(binary)

    # Centro della spirale (immagine di controllo)
    center = find_spiral_center(binary)
    debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if center is not None:
        cv2.circle(debug, center, radius=10, color=(0, 0, 255), thickness=-1)

    # Orientamento e ritaglio
    angle = detect_rotation_angle(binary)
    rotated = rotate_image(binary, angle)
    cropped = crop_document(rotated)
    final_size = (cropped.shape[1], cropped.shape[0])

    Image.fromarray(debug).save(output_dir / f"debug_{image_path.name}")

    return angle, original_size, final_size
