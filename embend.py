# -*- coding: utf-8 -*-
"""
Fixed version of the script: Build weighted reference face embeddings DB with exactly 5 prototypes per person.

Main fix: avoid using Python boolean `or` with numpy arrays (caused ValueError: truth value of an array is ambiguous).
Replaced expressions like `a = centroid(...) or centroid(...)` with explicit None checks.
Minor cleanups and additional comments.

Requirements:
 - insightface
 - numpy, opencv-python, tqdm
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from tqdm import tqdm

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

# -----------------------
# CONFIG
# -----------------------
INPUT_PATHS = [Path("photo")]
NAME = "unknown"
DB_EMBED_DIR = Path(r'project\embend')
EMB_MODEL_NAME = 'buffalo_l'

EMBEDDINGS_FILE = DB_EMBED_DIR / 'db_embeddings.npy'
LABELS_FILE     = DB_EMBED_DIR / 'db_labels.npy'
HAIR_FILE       = DB_EMBED_DIR / 'db_hair.npy'
CANDIDATES_FILE = DB_EMBED_DIR / 'candidates_by_person.npy'

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

DET_CONF_THRESHOLD = 0.45
MIN_FACE_PIXELS    = 50
FRAME_INTERVAL     = 1

USE_GPU = True
VERBOSE = True

# Поза
POSE_BINS = (-0.28, 0.28)  # left < -0.28; right > 0.28; else frontal

# Hairstyle
HAIR_ENABLED  = True
HAIR_BINS     = 16
# Базовые приоритеты для комбинированного скора по типу прототипа
PRIORITIES = {
    "frontal": 1.00,
    "left_profile": 0.80,
    "right_profile": 0.80,
    "back": 0.30,
    "hair": 0.20,
}
# -----------------------
# Utils
# -----------------------

def normalize_vec(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    if n == 0 or np.isnan(n):
        return v
    return (v / n).astype(np.float32)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def is_video_file(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def extract_frames_from_video(video_path: Path, frame_step: int = 30, max_frames: int = 0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        if VERBOSE:
            print(f"  Не удалось открыть видео {video_path}")
        return
    idx = 0
    returned = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            yield idx, frame
            returned += 1
            if max_frames > 0 and returned >= max_frames:
                break
        idx += 1
    cap.release()


def clamp_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))
    return x1, y1, x2, y2


def bbox_sharpness_v_laplacian(frame: np.ndarray, bbox) -> float:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

# -----------------------
# Pose / Hair / Quality
# -----------------------

def pose_score_from_kps(face) -> float:
    """Оценка yaw: отрицательное — влево, положительное — вправо."""
    try:
        kps = np.array(face.kps)  # (5,2)
        left_eye_x = float(kps[0, 0]); right_eye_x = float(kps[1, 0]); nose_x = float(kps[2, 0])
        eye_mid = (left_eye_x + right_eye_x) / 2.0
        eye_dist = max(1.0, abs(right_eye_x - left_eye_x))
        return (nose_x - eye_mid) / eye_dist
    except Exception:
        return 0.0


def compute_hairstyle_descriptor(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
    """Быстрый hair-дескриптор — гистограмма H (HSV) из области над лицом."""
    if frame is None:
        return None
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w_img, h_img)
    face_h = max(1, y2 - y1)
    top = max(0, y1 - face_h // 2)
    roi = frame[top:y1, x1:x2]
    if roi.size == 0:
        return None
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        hist = cv2.calcHist([h_channel], [0], None, [HAIR_BINS], [0, 180]).flatten().astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s
        return hist
    except Exception:
        return None


def calculate_quality_score(face, frame: np.ndarray) -> float:
    try:
        x1, y1, x2, y2 = face.bbox
        area = (x2 - x1) * (y2 - y1)
        score = np.log1p(area) * 100.0
        sharpness = bbox_sharpness_v_laplacian(frame, face.bbox)
        score += sharpness
        score += float(face.det_score) * 100.0
        return max(0.0, float(score))
    except Exception:
        return 0.0


def weighted_avg(embeddings: List[np.ndarray], weights: List[float]) -> Optional[np.ndarray]:
    if not embeddings or not weights or len(embeddings) != len(weights):
        return None
    E = np.array(embeddings, dtype=np.float32)
    W = np.array(weights, dtype=np.float32)
    if W.sum() == 0:
        W = np.ones_like(W)
    return normalize_vec(np.average(E, axis=0, weights=W))

# -----------------------
# Main
# -----------------------

def main():
    if FaceAnalysis is None:
        print("[!] insightface не установлен. pip install insightface")
        return

    os.makedirs(DB_EMBED_DIR, exist_ok=True)

    # Загрузка существующей базы (embeddings/labels/hair) — всё как object
    try:
        existing_embeddings = list(np.load(EMBEDDINGS_FILE, allow_pickle=True)) if EMBEDDINGS_FILE.exists() else []
        existing_labels     = list(np.load(LABELS_FILE,     allow_pickle=True)) if LABELS_FILE.exists()     else []
        existing_hairs      = list(np.load(HAIR_FILE,       allow_pickle=True)) if HAIR_FILE.exists()       else []
    except Exception as e:
        print(f"[!] Ошибка загрузки существующей базы: {e}")
        existing_embeddings, existing_labels, existing_hairs = [], [], []

    print("Загрузка модели InsightFace...")
    try:
        face_app = FaceAnalysis(name=EMB_MODEL_NAME, allowed_modules=['detection', 'recognition'])
        ctx_id = 0 if USE_GPU else -1
        face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    except Exception as e:
        print(f"[!] Не удалось инициализировать модель: {e}")
        return

    # --- Сбор кандидатов ---
    all_candidates: Dict[str, List[Dict[str, Any]]] = {}

    def process_frame(person_name: str, frame_bgr: np.ndarray):
        # Преобразуем в RGB для insightface
        faces = face_app.get(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        h_img, w_img = frame_bgr.shape[:2]
        for face in faces:
            if face.det_score < DET_CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = face.bbox
            if (x2 - x1) * (y2 - y1) < MIN_FACE_PIXELS ** 2:
                continue

            # --- ДОБАВЛЕНО: отступы вокруг лица (padding 20%) и безопасный кроп ---
            bw, bh = int(x2 - x1), int(y2 - y1)
            mx, my = int(bw * 0.15), int(bh * 0.15)
            px1 = max(0, int(x1) - mx)
            py1 = max(0, int(y1) - my)
            px2 = min(w_img, int(x2) + mx)
            py2 = min(h_img, int(y2) + my)
            # безопасный кроп (можно использовать для визуализации или дополнительной обработки)
            face_crop = frame_bgr[py1:py2, px1:px2].copy() if (py2 > py1 and px2 > px1) else None
            # --- КОНЕЦ ДОБАВЛЕНИЯ ---

            quality = calculate_quality_score(face, frame_bgr)
            if quality <= 0:
                continue
            emb = normalize_vec(face.normed_embedding.astype(np.float32))
            pose = pose_score_from_kps(face)
            # hair — можно вычислять как раньше по исходному bbox (в нашем случае используем frame_bgr и bbox без padding)
            hair = compute_hairstyle_descriptor(frame_bgr, face.bbox) if HAIR_ENABLED else None

            # Сохраняем кандидата. Не сохраняем изображение в базу, но держим face_crop в памяти кандидата, если нужно (не рекомендуется сохранять большие массивы в.npy)
            candidate = {
                "emb": emb,
                "weight": float(quality),
                "pose": float(pose),
                "hair": hair,
            }
            # Опционально добавить кроп для локального отладки (не сохраняется в .npy автоматически)
            if face_crop is not None:
                candidate["face_crop_shape"] = face_crop.shape

            all_candidates.setdefault(person_name, []).append(candidate)

    for root_path in INPUT_PATHS:
        root_path = Path(root_path)
        if not root_path.exists():
            print(f"[!] Путь не найден: {root_path}")
            continue

        if root_path.is_dir():
            subdirs = [p for p in sorted(root_path.iterdir()) if p.is_dir()]
            persons_dirs = subdirs if subdirs else [root_path]
            for person_dir in persons_dirs:
                person_name = person_dir.name if person_dir.is_dir() else NAME
                files = sorted([p for p in person_dir.iterdir() if p.is_file() and (is_image_file(p) or is_video_file(p))])
                for file_path in tqdm(files, desc=f"{person_name}", leave=False):
                    if is_image_file(file_path):
                        bgr = cv2.imdecode(np.fromfile(str(file_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                        if bgr is None:
                            continue
                        process_frame(person_name, bgr)
                    elif is_video_file(file_path):
                        for _, frame_bgr in extract_frames_from_video(file_path, frame_step=FRAME_INTERVAL):
                            process_frame(person_name, frame_bgr)
        else:
            # одиночный путь — трактуем как папку одного человека
            person_name = NAME if NAME else root_path.name

    # Сохраняем «сырые» кандидаты для отладки
    try:
        np.save(CANDIDATES_FILE, all_candidates, allow_pickle=True)
        if VERBOSE:
            print(f"Сохранены кандидаты: {CANDIDATES_FILE}")
    except Exception as e:
        print(f"[!] Не удалось сохранить кандидатов: {e}")

    # --- Формируем РОВНО 5 прототипов на человека ---
    new_embeddings: List[Optional[np.ndarray]] = []
    new_labels: List[str] = []
    new_hairs: List[Optional[np.ndarray]] = []

    def hair_average(items: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        hs = [c['hair'] for c in items if c.get('hair') is not None]
        if not hs:
            return None
        return np.mean(np.stack(hs, axis=0), axis=0).astype(np.float32)

    print("\nВычисление 5 прототипов на человека...")
    for person_name, cands in tqdm(all_candidates.items(), desc="Persons"):
        if not cands:
            continue

        # Группы по позе
        left_items, frontal_items, right_items = [], [], []
        for c in cands:
            p = c.get('pose', 0.0)
            if p < POSE_BINS[0]:
                left_items.append(c)
            elif p > POSE_BINS[1]:
                right_items.append(c)
            else:
                frontal_items.append(c)

        # Анфас (главный)
        def centroid(items: List[Dict[str, Any]]) -> Optional[np.ndarray]:
            if not items:
                return None
            return weighted_avg([it['emb'] for it in items], [it['weight'] for it in items])

        # compute frontal once and fallback to centroid(cands) when None
        frontal_emb = centroid(frontal_items)
        if frontal_emb is None:
            frontal_emb = centroid(cands)

        hair_avg_all = hair_average(cands)

        # Левый/правый профиль — если пусто, подставим анфас (чтобы было ровно 5 записей)
        left_emb = centroid(left_items)
        if left_emb is None:
            left_emb = frontal_emb
        right_emb = centroid(right_items)
        if right_emb is None:
            right_emb = frontal_emb

        # Hair для поз (если есть)
        hair_left = hair_average(left_items)
        if hair_left is None:
            hair_left = hair_avg_all
        hair_right = hair_average(right_items)
        if hair_right is None:
            hair_right = hair_avg_all
        hair_frontal = hair_average(frontal_items)
        if hair_frontal is None:
            hair_frontal = hair_avg_all

        # 1) frontal
        if frontal_emb is not None:
            new_embeddings.append(frontal_emb)
        else:
            # Защита: если вообще нет лиц (маловероятно), создадим единичный вектор-заглушку
            new_embeddings.append(np.zeros((512,), dtype=np.float32))
        new_labels.append(f"{person_name}__frontal")
        new_hairs.append(hair_frontal)

        # 2) left_profile
        new_embeddings.append(left_emb if left_emb is not None else new_embeddings[-1])
        new_labels.append(f"{person_name}__left_profile")
        new_hairs.append(hair_left)

        # 3) right_profile
        new_embeddings.append(right_emb if right_emb is not None else new_embeddings[-2])
        new_labels.append(f"{person_name}__right_profile")
        new_hairs.append(hair_right)

        # 4) back (hair-only) — emb=None, hair=hair_avg_all
        new_embeddings.append(None)
        new_labels.append(f"{person_name}__back")
        new_hairs.append(hair_avg_all)

        # 5) hair (hair-only) — emb=None, hair=hair_avg_all (наименьший вес)
        new_embeddings.append(None)
        new_labels.append(f"{person_name}__hair")
        new_hairs.append(hair_avg_all)

    # --- Объединение со старой базой, удаляя старые записи того же person root ---
    existing_map = list(zip(existing_labels, existing_embeddings, existing_hairs))
    new_roots = set([lab.split('__')[0] for lab in new_labels])

    filtered_existing_labels, filtered_existing_embeddings, filtered_existing_hairs = [], [], []
    for lab, emb, hair in existing_map:
        root = str(lab).split('__')[0]
        if root in new_roots:
            continue
        filtered_existing_labels.append(lab)
        filtered_existing_embeddings.append(emb)
        filtered_existing_hairs.append(hair)

    all_labels     = filtered_existing_labels + new_labels
    all_embeddings = filtered_existing_embeddings + new_embeddings
    all_hairs      = filtered_existing_hairs + new_hairs

    if not all_labels:
        print("[!] Нечего сохранять.")
        return

    # --- Очистка конфликтов/неуникальности только по face-эмбеддингам (hair-only пропускаем) ---
    def split_face_indices(embs: List[Optional[np.ndarray]]) -> Tuple[List[int], List[int]]:
        face_idxs, hair_only_idxs = [], []
        for i, e in enumerate(embs):
            if isinstance(e, np.ndarray):
                face_idxs.append(i)
            else:
                hair_only_idxs.append(i)
        return face_idxs, hair_only_idxs

    def clean_conflicts(embs_list: List[Optional[np.ndarray]], labels_list: List[str], sim_thresh: float = 0.90):
        face_idxs, hair_only_idxs = split_face_indices(embs_list)
        if not face_idxs:
            return embs_list, labels_list  # только hair-only — ничего не чистим

        E = np.stack([embs_list[i] for i in face_idxs], axis=0).astype(np.float32)
        sims = np.dot(E, E.T)

        # вспомогательные соответствия
        idx_map = {k: i for i, k in enumerate(face_idxs)}  # глобальный -> локальный в E
        remove = set()

        # группировка по root
        root_to_idxs = {}
        for gi in face_idxs:
            root = str(labels_list[gi]).split('__')[0]
            root_to_idxs.setdefault(root, []).append(gi)

        # intra scores (только по face)
        intra = {gi: 1.0 for gi in face_idxs}
        for root, gidxs in root_to_idxs.items():
            for gi in gidxs:
                others = [gj for gj in gidxs if gj != gi]
                if not others: 
                    intra[gi] = 1.0
                else:
                    li = idx_map[gi]
                    lo = [idx_map[x] for x in others]
                    intra[gi] = float(np.mean(sims[li, lo]))

        # конфликты между разными root
        for a_i in range(len(face_idxs)):
            gi = face_idxs[a_i]
            if gi in remove: 
                continue
            for b_i in range(a_i+1, len(face_idxs)):
                gj = face_idxs[b_i]
                if gj in remove: 
                    continue
                if str(labels_list[gi]).split('__')[0] == str(labels_list[gj]).split('__')[0]:
                    continue
                if sims[a_i, b_i] > sim_thresh:
                    # удаляем менее "укоренённый"
                    if intra[gi] < intra[gj]:
                        remove.add(gi); break
                    else:
                        remove.add(gj)

        keep = [i for i in range(len(embs_list)) if i not in remove]
        embs_f  = [embs_list[i]  for i in keep]
        labels_f= [labels_list[i] for i in keep]
        return embs_f, labels_f

    def prune_non_unique(embs_list: List[Optional[np.ndarray]], labels_list: List[str], intra_min: float = 0.35, inter_max: float = 0.88):
        face_idxs, hair_only_idxs = split_face_indices(embs_list)
        if not face_idxs:
            return embs_list, labels_list

        E = np.stack([embs_list[i] for i in face_idxs], axis=0).astype(np.float32)
        sims = np.dot(E, E.T)
        idx_map = {k: i for i, k in enumerate(face_idxs)}

        # группировка по root (только face)
        root_to_idxs = {}
        for gi in face_idxs:
            root = str(labels_list[gi]).split('__')[0]
            root_to_idxs.setdefault(root, []).append(gi)

        keep_face = set()
        for gi in face_idxs:
            root = str(labels_list[gi]).split('__')[0]
            own = [x for x in root_to_idxs[root] if x != gi]
            li = idx_map[gi]
            intra = float(np.mean([sims[li, idx_map[o]] for o in own])) if own else 1.0
            others = [x for x in face_idxs if str(labels_list[x]).split('__')[0] != root]
            max_inter = float(np.max([sims[li, idx_map[o]] for o in others])) if others else 0.0
            if intra >= intra_min and max_inter <= inter_max:
                keep_face.add(gi)

        # hair-only всегда сохраняем (их чистка делается на этапе ранжирования за счёт низких приоритетов)
        keep_all = set(hair_only_idxs) | keep_face
        embs_f   = [embs_list[i]  for i in range(len(embs_list)) if i in keep_all]
        labels_f = [labels_list[i] for i in range(len(embs_list)) if i in keep_all]
        return embs_f, labels_f

    # чистка
    all_embeddings, all_labels = clean_conflicts(all_embeddings, all_labels, sim_thresh=0.90)
    all_embeddings, all_labels = prune_non_unique(all_embeddings, all_labels, intra_min=0.35, inter_max=0.88)

    # Подготовим hair под текущий список меток
    label_to_hair = {str(l): h for l, h in zip(filtered_existing_labels + new_labels,
                                               filtered_existing_hairs + new_hairs)}
    all_hairs = [label_to_hair.get(str(l), None) for l in all_labels]

    # --- Сохранение (dtype=object, т.к. есть None) ---
    try:
        np.save(EMBEDDINGS_FILE, np.array(all_embeddings, dtype=object), allow_pickle=True)
        np.save(LABELS_FILE,     np.array(all_labels,     dtype=object), allow_pickle=True)
        np.save(HAIR_FILE,       np.array(all_hairs,      dtype=object), allow_pickle=True)
        print(f"База обновлена. Всего эталонов: {len(all_labels)}")
    except Exception as e:
        print(f"[!] Ошибка при сохранении базы: {e}")
        return

    # --- Комбинированный скоринг (пример использования) ---
    def prototype_priority(label: str) -> float:
        suffix = label.split('__')[-1] if '__' in label else 'frontal'
        return PRIORITIES.get(suffix, 0.5)

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32); b = b.astype(np.float32)
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def combined_similarity(query_emb: Optional[np.ndarray],
                            db_embs: List[Optional[np.ndarray]],
                            query_hair: Optional[np.ndarray],
                            db_hairs: List[Optional[np.ndarray]]) -> np.ndarray:
        """
        Возвращает массив скорингов по БД:
          score_i = P(label_i) * [ face_sim_i (если оба есть) + hair_sim_i ]
        где вклад hair маленький за счёт P для '__back' и '__hair'.
        """
        N = len(db_embs)
        scores = np.zeros((N,), dtype=np.float32)

        # Нормализация query_emb (если есть)
        q_emb = None
        if isinstance(query_emb, np.ndarray):
            q_emb = normalize_vec(query_emb)

        # Нормализация hair (если есть)
        qh = None
        if isinstance(query_hair, np.ndarray):
            qh = query_hair.astype(np.float32)
            n = np.linalg.norm(qh)
            if n > 0:
                qh = qh / n
            else:
                qh = None

        for i in range(N):
            label_i = str(all_labels[i])
            pr = prototype_priority(label_i)

            face_sim = 0.0
            if q_emb is not None and isinstance(db_embs[i], np.ndarray):
                face_sim = float(np.dot(db_embs[i], q_emb))  # обе стороны уже L2-норм.

            hair_sim = 0.0
            if qh is not None and isinstance(db_hairs[i], np.ndarray):
                hh = db_hairs[i].astype(np.float32)
                nh = np.linalg.norm(hh)
                if nh > 0:
                    hh = hh / nh
                    hair_sim = float(np.dot(hh, qh))

            scores[i] = pr * (face_sim + hair_sim)

        return scores

    # Подсказка по использованию:
    # emb_db  = list(np.load(EMBEDDINGS_FILE, allow_pickle=True))
    # labs_db = list(np.load(LABELS_FILE,     allow_pickle=True))
    # hair_db = list(np.load(HAIR_FILE,       allow_pickle=True))
    # sims = combined_similarity(query_emb, emb_db, query_hair, hair_db)
    # topk = np.argsort(-sims)[:5]; print([labs_db[i] for i in topk])

if __name__ == '__main__':
    main()
