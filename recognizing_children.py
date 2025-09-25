from __future__ import annotations
import re
import builtins
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import os
from minio import Minio
# попытка импортировать faiss и insightface, graceful fallback
try:
    import faiss
except Exception:
    faiss = None  # если faiss нет — поиск по базе будет нерабоч

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None

# ------------------ GLOBAL PRINT SILENCER ------------------
# сохраняем оригинальную функцию print — будем её использовать только для логов attendance
__orig_print = builtins.print


def _silent_print(*args, **kwargs):
    # подавляем все обычные вызовы print в модуле
    return None


# переопределяем глобальную print в модуле
builtins.print = _silent_print
# ----------------------------------------------------------

# ---------------- CONFIG ----------------
ATTENDANCE_URL = "https://kindy.kz/api/profile/open-api/attendance/mark"
VIDEO_SOURCE = "rtsp://admin:Cosmo445566@cosmo2025.ddns.net:56323/Streaming/channels/601"
EMB_MODEL_NAME = "buffalo_l"

FRAME_SKIP = 4
RECOG_INTERVAL_SEC = 150.0

MIN_RECOG_ATTEMPTS_INTERVAL = 60
RERECOG_MIN_FRAMES = 900
RERECOG_DET_IMPROVEMENT = 0.02

DET_CONF_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.3
MIN_FACE_PIXELS = 10
MAX_MISSED_FRAMES = 10
IOU_THRESHOLD = 0.1

RECOGNITION_CONFIDENCE_FRONTAL = 0.40
RECOGNITION_CONFIDENCE_PROFILE = 0.35
POSE_PROFILE_PERCENT = 90.0
SINGLE_FRAME_PROFILE_PERCENT = 50.0
POSE_PROFILE_THRESHOLD = POSE_PROFILE_PERCENT / 100.0
SINGLE_FRAME_PROFILE_THRESHOLD = SINGLE_FRAME_PROFILE_PERCENT / 100.0

POSE_HISTORY_LEN = 5
POSE_SMOOTH_ALPHA = 0.6
POSE_MIN_VALID = 2

SMOOTHING_ALPHA = 0.6
VEL_ALPHA = 0.4
PADDING_PIXELS = 12

HAIR_BINS = 16
HAIR_PRIORITY = 0.10

PRIORITIES = {
    "frontal": 1.00,
    "left_profile": 0.80,
    "right_profile": 0.80,
    "back": 0.50,
    "hair": 0.30,
}

NOTIFIED_DB_PATH = Path("notified_db.json")

# ---------------- RECONNECT CONFIG ----------------
RECONNECT_MAX_RETRIES = None      # None = бесконечно
RECONNECT_BACKOFF_BASE = 1.0      # базовый интервал (сек)
RECONNECT_BACKOFF_MAX = 60.0      # макс интервал (сек)
RECONNECT_LOG_PREFIX = "[RECONNECT]"
# ----------------------------------------------------
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "10.201.75.10:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "system_operator")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "71OL9WpdfU0bVjy")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "False").lower() in ("1","true","yes")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "kindy-ai")

# locks and globals
SENT_SNAPSHOTS_SESSION = set()
SENT_SNAPSHOTS_LOCK = threading.Lock()

active_label_map = {}
active_label_map_lock = threading.Lock()
LABEL_STEAL_MARGIN = 0.02

_track_id_counter = 0
_track_id_lock = threading.Lock()


def next_track_id() -> int:
    global _track_id_counter
    with _track_id_lock:
        _track_id_counter += 1
        return _track_id_counter


# ----------------- utility functions -----------------

def _current_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _is_int_string(s: str) -> bool:
    try:
        int(str(s))
        return True
    except Exception:
        return False


def normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0 or np.isnan(norm):
        return v
    return (v / (norm + 1e-12)).astype(np.float32)


def calculate_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = max(0, (x2_1 - x1_1)) * max(0, (y2_1 - y1_1))
    box2_area = max(0, (x2_2 - x1_2)) * max(0, (y2_2 - y1_2))
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0.0


def clamp_bbox_xyxy(bbox, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w - 1, int(round(x2))))
    y2 = max(0, min(h - 1, int(round(y2))))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def extract_keypoints_from_face(face) -> Optional[np.ndarray]:
    candidates = ["kps", "keypoints", "landmark_2d_106", "landmark", "landmarks"]
    for name in candidates:
        k = getattr(face, name, None)
        if k is None:
            continue
        try:
            arr = np.array(k, dtype=np.float32)
            if arr.ndim == 1 and arr.size % 2 == 0:
                arr = arr.reshape((-1, 2))
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
        except Exception:
            continue
    return None


def compute_pose_from_kps_arr(kps: np.ndarray) -> Optional[float]:
    if kps is None:
        return None
    try:
        if not isinstance(kps, np.ndarray) or kps.size == 0 or kps.shape[0] < 3:
            return None
        left_eye_x = float(kps[0, 0]); right_eye_x = float(kps[1, 0]); nose_x = float(kps[2, 0])
        eye_mid = (left_eye_x + right_eye_x) / 2.0
        eye_dist = max(1.0, abs(right_eye_x - left_eye_x))
        pose = (nose_x - eye_mid) / eye_dist
        if np.isnan(pose):
            return None
        return float(pose)
    except Exception:
        return None


def keypoints_visible_for_bbox(kps: np.ndarray, bbox, min_in_bbox: Optional[int] = None) -> bool:
    if kps is None or not isinstance(kps, np.ndarray) or kps.size == 0:
        return False
    total = len(kps)
    if min_in_bbox is None:
        min_in_bbox = max(2, int(0.3 * total))
    x1, y1, x2, y2 = [int(v) for v in bbox]
    in_bbox = 0
    for (x, y) in kps:
        if np.isfinite(x) and np.isfinite(y) and x1 <= x <= x2 and y1 <= y <= y2:
            in_bbox += 1
    return in_bbox >= min_in_bbox or in_bbox >= 0.5 * total


def compute_hairstyle_descriptor(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
    if frame is None:
        return None
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox_xyxy(bbox, w_img, h_img)
    face_h = max(1, y2 - y1)
    top = max(0, y1 - face_h // 2)
    roi = frame[top:y1, x1:x2]
    if roi.size == 0:
        return None
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        hist = cv2.calcHist([h_channel], [0], None, [HAIR_BINS], [0, 180])
        hist = hist.flatten().astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s
        return hist
    except Exception:
        return None


def bbox_brightness(frame: np.ndarray, bbox) -> float:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox_xyxy(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = frame[y1:y2, x1:x2]
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 2]))
    except Exception:
        return float(np.mean(roi))


def is_ideal_face(person, frame) -> bool:
    try:
        x1, y1, x2, y2 = person.bbox.astype(int)
        width = x2 - x1
        brightness = bbox_brightness(frame, person.bbox)
        if person.det_score >= 0.9 and width >= 120 and 50 <= brightness <= 200:
            return True
    except Exception:
        pass
    return False


# ------------------ TrackedPerson ------------------
class TrackedPerson:
    """Хранит состояние трека: bbox, embedding history, pose и т.д."""

    def __init__(self, face_obj, frame_shape, track_id: int, frame_image=None):
        self.track_id = int(track_id)
        x1, y1, x2, y2 = np.array(face_obj.bbox, dtype=int)
        h, w = frame_shape[:2]
        x1 = max(0, x1 - PADDING_PIXELS); y1 = max(0, y1 - PADDING_PIXELS)
        x2 = min(w - 1, x2 + PADDING_PIXELS); y2 = min(h - 1, y2 + PADDING_PIXELS)
        self.bbox = np.array([x1, y1, x2, y2], dtype=float)
        self.smoothed_bbox = self.bbox.copy()
        self.vel = np.zeros(4, dtype=float)

        self.det_score = float(getattr(face_obj, 'det_score', 0.0))
        self.prev_det_score = self.det_score

        self.missed_frames = 0
        self.history = deque(maxlen=5)

        self.label = 'Unknown'
        self.fixed_label = None
        self.recognized = False
        self.similarity = 0.0

        self.last_recog_attempt_frame = -9999
        self.last_recognition_frame = -9999
        self.created_frame = -1

        self.kps = extract_keypoints_from_face(face_obj)
        self.hair = getattr(face_obj, 'hair', None)

        self.pose_history = deque(maxlen=POSE_HISTORY_LEN)
        self.pose_smoothed = 0.0
        self.pose_confident = False
        initial_pose = compute_pose_from_kps_arr(self.kps) if self.kps is not None else None
        if initial_pose is not None:
            self.pose_history.append(initial_pose)
            self.pose_smoothed = float(initial_pose)
            self.pose_confident = len(self.pose_history) >= POSE_MIN_VALID
            self.pose_was_frontal = abs(self.pose_smoothed) <= POSE_PROFILE_THRESHOLD
        else:
            self.pose_was_frontal = False

        self.embedding = None
        self.initial_embedding = None
        self.prototype_embedding = None
        self.emb_history = deque(maxlen=8)

        emb = None
        if getattr(face_obj, 'normed_embedding', None) is not None:
            try:
                emb = np.array(face_obj.normed_embedding, dtype=np.float32)
            except Exception:
                emb = None
        elif getattr(face_obj, 'embedding', None) is not None:
            try:
                emb = np.array(face_obj.embedding, dtype=np.float32)
            except Exception:
                emb = None
        if emb is not None:
            emb = normalize_vec(emb)
            self.embedding = emb
            self.emb_history.append((emb, self.det_score, time.time()))
            self.prototype_embedding = emb.copy()

        self.snapshot_sent = False
        self.unrecognized_frames = 0

    def get_search_embedding(self) -> Optional[np.ndarray]:
        return self.prototype_embedding if self.prototype_embedding is not None else self.embedding

    def update(self, face_obj, frame_image=None) -> None:
        try:
            self.prev_det_score = self.det_score
            new_bbox = np.array(face_obj.bbox, dtype=float)
            x1, y1, x2, y2 = new_bbox.astype(int)
            x1 -= PADDING_PIXELS; y1 -= PADDING_PIXELS; x2 += PADDING_PIXELS; y2 += PADDING_PIXELS
            new_bbox = np.array([x1, y1, x2, y2], dtype=float)

            new_vel = new_bbox - self.smoothed_bbox
            self.vel = self.vel * (1.0 - VEL_ALPHA) + new_vel * VEL_ALPHA

            self.smoothed_bbox = SMOOTHING_ALPHA * new_bbox + (1.0 - SMOOTHING_ALPHA) * self.smoothed_bbox
            self.bbox = self.smoothed_bbox.copy()
            self.missed_frames = 0
            self.last_seen_at = time.time()

            self.det_score = float(getattr(face_obj, 'det_score', self.det_score))
            self.history.append(self.bbox.copy())

            new_kps = extract_keypoints_from_face(face_obj)
            if new_kps is not None:
                try:
                    self.kps = new_kps
                except Exception:
                    pass
            if hasattr(face_obj, 'hair'):
                self.hair = getattr(face_obj, 'hair', None)

            try:
                new_pose = compute_pose_from_kps_arr(self.kps) if self.kps is not None else None
                if new_pose is not None:
                    self.pose_history.append(new_pose)
                    self.pose_smoothed = POSE_SMOOTH_ALPHA * float(new_pose) + (1.0 - POSE_SMOOTH_ALPHA) * float(self.pose_smoothed)
            except Exception:
                pass
            self.pose_confident = (len(self.pose_history) >= POSE_MIN_VALID)

            emb = None
            if getattr(face_obj, 'normed_embedding', None) is not None:
                try:
                    emb = np.array(face_obj.normed_embedding, dtype=np.float32)
                except Exception:
                    emb = None
            elif getattr(face_obj, 'embedding', None) is not None:
                try:
                    emb = np.array(face_obj.embedding, dtype=np.float32)
                except Exception:
                    emb = None
            if emb is not None:
                emb = normalize_vec(emb)
                if np.linalg.norm(emb) > 1e-6:
                    self.embedding = emb
                    self.emb_history.append((emb, self.det_score, time.time()))
                    try:
                        embs = np.stack([e for (e, s, t) in self.emb_history], axis=0)
                        scores = np.array([s for (e, s, t) in self.emb_history], dtype=np.float32)
                        w = scores / (scores.sum() + 1e-12)
                        agg = (embs * w[:, None]).sum(axis=0)
                        agg = normalize_vec(agg)
                        if np.linalg.norm(agg) > 1e-6:
                            self.prototype_embedding = agg
                    except Exception:
                        pass
        except Exception:
            pass

    def predict(self) -> np.ndarray:
        self.smoothed_bbox = self.smoothed_bbox + self.vel
        self.bbox = self.smoothed_bbox.copy()
        self.missed_frames += 1
        return self.bbox.astype(int)

    def has_visible_keypoints(self, frame_shape=None) -> bool:
        try:
            if self.kps is None:
                return False
            return keypoints_visible_for_bbox(self.kps, self.bbox)
        except Exception:
            return False


# ------------------ FaceDatabase ------------------
class FaceDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.index = None
        self.labels = np.array([], dtype=object)
        self.db_hairs = []
        self.db_embeddings = None
        self.full_embeddings = []
        self.face_index_to_global = []
        self.prototype_weights = []
        self.prototype_priorities = []
        self.load_database()

    def load_database(self) -> None:
        embeddings_file = self.db_path / 'db_embeddings.npy'
        labels_file = self.db_path / 'db_labels.npy'
        hair_file = self.db_path / 'db_hair.npy'
        weights_file = self.db_path / 'db_weights.npy'

        if not embeddings_file.exists() or not labels_file.exists():
            return

        try:
            raw_embeddings = np.load(embeddings_file, allow_pickle=True)
            raw_labels = np.load(labels_file, allow_pickle=True)
            self.labels = np.array(list(raw_labels), dtype=object)

            emb_list = []
            if isinstance(raw_embeddings, np.ndarray) and raw_embeddings.dtype != object:
                for i in range(raw_embeddings.shape[0]):
                    emb = np.array(raw_embeddings[i], dtype=np.float32)
                    emb_list.append(emb)
            else:
                for e in raw_embeddings:
                    if e is None or (isinstance(e, float) and np.isnan(e)):
                        emb_list.append(None)
                    else:
                        emb_list.append(np.array(e, dtype=np.float32))
            self.full_embeddings = emb_list

            if hair_file.exists():
                try:
                    raw_hairs = np.load(hair_file, allow_pickle=True)
                    hairs = []
                    for h in raw_hairs:
                        if h is None or (isinstance(h, float) and np.isnan(h)):
                            hairs.append(None)
                        else:
                            hairs.append(np.array(h, dtype=np.float32))
                    self.db_hairs = hairs
                except Exception:
                    self.db_hairs = [None] * len(self.labels)
            else:
                self.db_hairs = [None] * len(self.labels)

            if weights_file.exists():
                try:
                    raw_w = np.load(weights_file, allow_pickle=True)
                    self.prototype_weights = [float(x) if x is not None else 1.0 for x in raw_w]
                except Exception:
                    self.prototype_weights = [1.0] * len(self.labels)
            else:
                self.prototype_weights = [1.0] * len(self.labels)

            prios = []
            for lab in self.labels:
                s = str(lab)
                suffix = s.split('__')[-1] if '__' in s else 'frontal'
                pr = PRIORITIES.get(suffix, 0.5)
                prios.append(float(pr))
            self.prototype_priorities = prios

            face_embs = []
            face_global_idxs = []
            for i, emb in enumerate(self.full_embeddings):
                if isinstance(emb, np.ndarray):
                    face_embs.append(normalize_vec(emb))
                    face_global_idxs.append(i)

            if face_embs and faiss is not None:
                E = np.stack(face_embs, axis=0).astype(np.float32)
                self.db_embeddings = E.copy()
                faiss.normalize_L2(self.db_embeddings)
                dim = self.db_embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dim)
                self.index.add(self.db_embeddings)
                self.face_index_to_global = face_global_idxs
            else:
                self.index = None
                self.face_index_to_global = []

        except Exception:
            # при ошибке просто делаем пустую базу
            self.index = None
            self.labels = np.array([], dtype=object)
            self.db_hairs = []
            self.db_embeddings = None
            self.full_embeddings = []
            self.prototype_weights = []
            self.prototype_priorities = []

    def _base_label(self, label: str) -> str:
        try:
            s = str(label)
            if '__' in s:
                return s.split('__')[0]
            return s
        except Exception:
            return str(label)

    def search(self, embedding: np.ndarray, query_hair: Optional[np.ndarray] = None,
               threshold: float = None, query_pose: Optional[float] = None):
        if threshold is None:
            threshold = SIMILARITY_THRESHOLD

        best_combined = -1.0
        best_face_sim = 0.0
        best_i = -1
        best_raw_label = None

        try:
            q_emb = normalize_vec(np.array(embedding, dtype=np.float32)) if embedding is not None else None
        except Exception:
            q_emb = None

        qh = None
        try:
            if query_hair is not None:
                qh = np.array(query_hair, dtype=np.float32)
                qh = qh / (np.linalg.norm(qh) + 1e-8)
        except Exception:
            qh = None

        pose_suffix = None
        if query_pose is not None:
            try:
                if query_pose < -0.25:
                    pose_suffix = 'left_profile'
                elif query_pose > 0.25:
                    pose_suffix = 'right_profile'
                else:
                    pose_suffix = 'frontal'
            except Exception:
                pose_suffix = None

        # поиск по embedding через faiss если доступен
        if self.index is not None and q_emb is not None and self.index.ntotal > 0:
            q = np.expand_dims(q_emb.astype(np.float32), axis=0)
            faiss.normalize_L2(q)
            top_k = min(64, self.index.ntotal)
            D, I = self.index.search(q, k=top_k)
            face_sims = D[0]
            idxs = I[0]

            for i_local, idx in enumerate(idxs):
                if idx < 0:
                    continue
                global_idx = self.face_index_to_global[int(idx)]
                face_sim = float(face_sims[i_local])
                hair_sim = 0.0
                dbh = None
                if global_idx < len(self.db_hairs):
                    dbh = self.db_hairs[global_idx]
                if qh is not None and isinstance(dbh, np.ndarray):
                    nh = np.linalg.norm(dbh)
                    if nh > 0:
                        hh = dbh.astype(np.float32) / nh
                        hair_sim = float(np.dot(hh, qh))
                proto_weight = float(self.prototype_weights[global_idx]) if global_idx < len(self.prototype_weights) else 1.0
                proto_prio = float(self.prototype_priorities[global_idx]) if global_idx < len(self.prototype_priorities) else 1.0

                pose_boost = 1.0
                if pose_suffix is not None:
                    try:
                        cand_suffix = str(self.labels[global_idx]).split('__')[-1] if '__' in str(self.labels[global_idx]) else 'frontal'
                        if cand_suffix == pose_suffix:
                            pose_boost = 1.25
                        else:
                            if cand_suffix in ('left_profile', 'right_profile') and pose_suffix == 'frontal':
                                pose_boost = 0.95
                    except Exception:
                        pose_boost = 1.0

                effective_hair_prio = HAIR_PRIORITY
                if pose_suffix is not None and pose_suffix in ('left_profile', 'right_profile'):
                    effective_hair_prio = max(HAIR_PRIORITY, 0.25)

                combined = proto_weight * proto_prio * pose_boost * ((1.0 - effective_hair_prio) * face_sim + effective_hair_prio * hair_sim)

                if combined > best_combined:
                    best_combined = combined
                    best_face_sim = face_sim
                    best_i = int(global_idx)
                    best_raw_label = str(self.labels[global_idx])

        # доп. поиск по волосам (если в db есть только волосы)
        if qh is not None:
            for global_idx, emb in enumerate(self.full_embeddings):
                if isinstance(emb, np.ndarray):
                    continue
                dbh = self.db_hairs[global_idx] if global_idx < len(self.db_hairs) else None
                if dbh is None:
                    continue
                nh = np.linalg.norm(dbh)
                if nh == 0:
                    continue
                hh = dbh.astype(np.float32) / nh
                hair_sim = float(np.dot(hh, qh))
                proto_weight = float(self.prototype_weights[global_idx]) if global_idx < len(self.prototype_weights) else 1.0
                proto_prio = float(self.prototype_priorities[global_idx]) if global_idx < len(self.prototype_priorities) else 1.0

                pose_boost = 1.0
                if pose_suffix is not None:
                    cand_suffix = str(self.labels[global_idx]).split('__')[-1] if '__' in str(self.labels[global_idx]) else 'frontal'
                    if cand_suffix in ('back', 'hair') and pose_suffix in ('left_profile', 'right_profile'):
                        pose_boost = 1.1

                effective_hair_prio = HAIR_PRIORITY
                if pose_suffix is not None and pose_suffix in ('left_profile', 'right_profile'):
                    effective_hair_prio = max(HAIR_PRIORITY, 0.25)

                combined = proto_weight * proto_prio * pose_boost * (effective_hair_prio * hair_sim)

                if combined > best_combined:
                    best_combined = combined
                    best_face_sim = 0.0
                    best_i = int(global_idx)
                    best_raw_label = str(self.labels[global_idx])

        if best_i >= 0 and best_combined > threshold:
            base_label = self._base_label(best_raw_label)
            return base_label, float(best_combined), best_raw_label, int(best_i), float(best_face_sim)

        return 'Unknown', float(best_combined if best_combined >= 0 else 0.0), (str(best_raw_label) if best_raw_label is not None else None), (int(best_i) if best_i >= 0 else -1), float(best_face_sim)


# ---------------- recognition helpers ----------------

def should_recognize(person: TrackedPerson, frame_idx: int) -> bool:
    try:
        if getattr(person, 'bbox', None) is None:
            return False
        if getattr(person, 'last_recog_attempt_frame', -9999) < 0:
            return True
        if (person.label == 'Unknown' or not getattr(person, 'recognized', False)):
            if frame_idx - person.last_recog_attempt_frame >= MIN_RECOG_ATTEMPTS_INTERVAL:
                return True
        if getattr(person, 'last_recognition_frame', -9999) >= 0:
            if frame_idx - person.last_recognition_frame >= RERECOG_MIN_FRAMES:
                return True
        try:
            if (getattr(person, 'det_score', 0.0) - getattr(person, 'prev_det_score', 0.0)) >= RERECOG_DET_IMPROVEMENT:
                if frame_idx - person.last_recog_attempt_frame >= MIN_RECOG_ATTEMPTS_INTERVAL:
                    return True
        except Exception:
            pass
        try:
            cur_frontal = getattr(person, 'pose_confident', False) and abs(getattr(person, 'pose_smoothed', 1.0)) <= POSE_PROFILE_THRESHOLD
            if cur_frontal and not getattr(person, 'pose_was_frontal', False):
                if frame_idx - person.last_recog_attempt_frame >= MIN_RECOG_ATTEMPTS_INTERVAL:
                    return True
        except Exception:
            pass
    except Exception:
        return False
    return False


def claim_label_for_track(person: TrackedPerson, base_label: str, sim: float, frame_idx: int) -> bool:
    global active_label_map
    if base_label is None or base_label == 'Unknown':
        return False
    try:
        with active_label_map_lock:
            cur = active_label_map.get(base_label)
            if cur is None:
                active_label_map[base_label] = {'track': person, 'sim': float(sim), 'frame': int(frame_idx)}
                person.fixed_label = str(base_label)
                person.label = str(base_label)
                person.similarity = float(sim)
                person.recognized = True
                person.last_recognition_frame = frame_idx
                person.snapshot_sent = False
                return True
            cur_track = cur.get('track')
            if (cur_track is person) or (hasattr(cur_track, 'track_id') and getattr(cur_track, 'track_id', None) == getattr(person, 'track_id', None)):
                cur['sim'] = float(sim); cur['frame'] = int(frame_idx)
                person.similarity = float(sim); person.last_recognition_frame = frame_idx
                return True
            cur_sim = float(cur.get('sim', 0.0))
            if float(sim) > cur_sim + LABEL_STEAL_MARGIN:
                prev_track = cur.get('track')
                try:
                    if prev_track is not None:
                        prev_track.recognized = False
                        prev_track.label = 'Unknown'
                        prev_track.similarity = 0.0
                        prev_track.snapshot_sent = False
                        prev_track.fixed_label = None
                except Exception:
                    pass
                active_label_map[base_label] = {'track': person, 'sim': float(sim), 'frame': int(frame_idx)}
                person.fixed_label = str(base_label); person.label = str(base_label)
                person.similarity = float(sim); person.recognized = True; person.last_recognition_frame = frame_idx
                person.snapshot_sent = False
                return True
            return False
    except Exception:
        return False


def try_recognize_person(person: TrackedPerson, db: FaceDatabase, frame_idx: int, frame) -> None:
    # быстрые проверки
    if db.index is None and all(e is None for e in db.full_embeddings):
        return
    try:
        kps_ok = person.has_visible_keypoints(frame.shape)
    except Exception:
        kps_ok = False

    x1, y1, x2, y2 = person.bbox.astype(int)
    width = max(0, x2 - x1)
    alt_ok = (person.det_score >= 0.85 and width >= 100)
    if not (kps_ok or alt_ok):
        return

    query_pose = None
    if getattr(person, 'pose_confident', False):
        query_pose = float(person.pose_smoothed)
    per_pose_recog = RECOGNITION_CONFIDENCE_FRONTAL if (getattr(person, 'pose_confident', False) and abs(getattr(person, 'pose_smoothed', 0.0)) <= POSE_PROFILE_THRESHOLD) else RECOGNITION_CONFIDENCE_PROFILE

    emb_for_search = person.get_search_embedding()
    if emb_for_search is None:
        return

    try:
        label, combined_sim, _, _, _ = db.search(emb_for_search, query_hair=getattr(person, 'hair', None), threshold=per_pose_recog, query_pose=query_pose)
    except Exception:
        return

    if label != 'Unknown' and combined_sim >= per_pose_recog:
        claimed = claim_label_for_track(person, label, combined_sim, frame_idx)
        if claimed:
            if not getattr(person, 'recognized', False):
                if person.embedding is not None:
                    person.initial_embedding = np.array(person.embedding, dtype=np.float32).copy()
            person.label = label; person.similarity = combined_sim; person.recognized = True
            person.last_recognition_frame = frame_idx
            person.unrecognized_frames = 0
            send_snapshot_if_recognized(person, frame, frame.shape[1], frame.shape[0], frame_idx)
    else:
        if getattr(person, 'recognized', False) or getattr(person, 'fixed_label', None) is not None:
            person.unrecognized_frames = person.unrecognized_frames + 1 if getattr(person, 'unrecognized_frames', None) is not None else 1
            if getattr(person, 'initial_embedding', None) is not None and person.embedding is not None:
                sim = 0.0
                try:
                    a = np.array(person.embedding, dtype=np.float32); b = np.array(person.initial_embedding, dtype=np.float32)
                    na = np.linalg.norm(a); nb = np.linalg.norm(b)
                    if na > 1e-6 and nb > 1e-6:
                        sim = float(np.dot(a, b) / (na * nb))
                except Exception:
                    sim = 0.0
                if sim >= 0.02:
                    person.unrecognized_frames = 0
                    person.similarity = sim
                    return
            if person.unrecognized_frames >= 6000:
                person.label = 'Unknown'; person.recognized = False; person.similarity = 0.0
                person.initial_embedding = None; person.fixed_label = None


# ------------------ Attendance API sender ------------------
# добавьте эту функцию к другим вспомогательным функциям
def extract_number_from_string(s: str) -> Optional[int]:
    """Извлекает первое число из строки, используя регулярные выражения."""
    try:
        # Ищем одно или несколько чисел в начале строки, за которыми следует символ '_'
        match = re.search(r'^(\d+)_', s)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None


# Функция для отправки запроса на API
def mark_attendance_api(child_identifier, track_id: Optional[int] = None, bbox: Optional[tuple] = None,
                        frame_idx: Optional[int] = None, timeout: int = 8, retries: int = 3, backoff: float = 0.8) -> bool:
    """Отправляет POST на ATTENDANCE_URL. Логи отправки выводятся через __orig_print.

    child_identifier может быть int (childId) или str (childName).
    Возвращает True при 2xx ответе.
    """
    try:
        payload = {"present": True}
        if child_identifier is None:
            __orig_print("[ERROR] child_identifier is None — пропускаем отправку.")
            return False

        # Попытка извлечь числовой ID из строки child_identifier
        child_id = extract_number_from_string(str(child_identifier))

        if child_id is not None:
            payload["childId"] = child_id
        else:
            # Если число не найдено, отправляем как childName
            payload["childName"] = str(child_identifier)

        headers = {"Content-Type": "application/json"}

        __orig_print("------------------------------------------------------------")
        __orig_print(f"[LOG] {_current_timestamp()} -> Отправка attendance to {ATTENDANCE_URL}")
        __orig_print(f"[LOG] Отправляемый payload: {payload}")
        __orig_print("------------------------------------------------------------")

        for attempt in range(retries):
            try:
                response = requests.post(ATTENDANCE_URL, data=json.dumps(payload), headers=headers, timeout=timeout)
                response.raise_for_status()
                __orig_print(f"[SUCCESS] Attendance API: {response.status_code} - {response.reason}")
                __orig_print(f"[INFO] Payload: {payload}")
                return True
            except requests.exceptions.HTTPError as http_err:
                __orig_print(f"[ERROR] HTTP-ошибка: {http_err} (Status: {http_err.response.status_code})")
                __orig_print(f"[INFO] Тело ответа: {http_err.response.text}")
                if 400 <= http_err.response.status_code < 500:
                    break
            except requests.exceptions.RequestException as req_err:
                __orig_print(f"[ERROR] Ошибка запроса: {req_err}")
                if attempt < retries - 1:
                    time.sleep(backoff * (2 ** attempt))
            except Exception as e:
                __orig_print(f"[ERROR] Неизвестная ошибка: {e}")
                break
        return False
    except Exception as e:
        __orig_print(f"[ERROR] Непредвиденная ошибка в mark_attendance_api: {e}")
        return False


def save_snapshot_to_folder(image_np: np.ndarray, label: str, sim: float, track_id: int, frame_idx: int, out_dir: Path = None) -> bool:
    """Wrapper: раньше сохранял файл — сейчас только вызывает mark_attendance_api."""
    try:
        if label is None:
            __orig_print("[ERROR] save_snapshot_to_folder called with label=None")
            return False
        return mark_attendance_api(child_identifier=label, track_id=track_id, bbox=None, frame_idx=frame_idx)
    except Exception as e:
        __orig_print(f"[!] save_snapshot_to_folder wrapper error: {e}")
        return False


def send_snapshot_if_recognized(person: TrackedPerson, frame, w: int, h: int, frame_idx: int) -> None:
    """Если человек распознан и snapshot ещё не отправлен — вызываем API"""
    try:
        if not getattr(person, "recognized", False):
            return
        if getattr(person, "snapshot_sent", False):
            return
        label = getattr(person, "label", "")
        if not label or label == "Unknown":
            return

        try:
            x1, y1, x2, y2 = person.bbox
            if x2 <= x1 or y2 <= y1:
                return
        except Exception:
            return

        ok = mark_attendance_api(child_identifier=label, track_id=getattr(person, "track_id", None),
                                 bbox=tuple(map(int, clamp_bbox_xyxy(person.bbox, w, h))) if getattr(person, "bbox", None) is not None else None,
                                 frame_idx=frame_idx)
        if ok:
            person.snapshot_sent = True
    except Exception as e:
        __orig_print(f"[!] send_snapshot_if_recognized error for label={getattr(person, 'label', None)}: {e}")


# ---------------- main loop ----------------

def load_notified_db():
    global notified_db
    try:
        if NOTIFIED_DB_PATH.exists():
            with open(NOTIFIED_DB_PATH, 'r', encoding='utf-8') as f:
                notified_db = json.load(f)
        else:
            notified_db = {}
    except Exception:
        notified_db = {}


def save_notified_db():
    try:
        with open(NOTIFIED_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(notified_db, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ------------------ NEW: open_capture_with_retry ------------------
def open_capture_with_retry(source: str,
                            max_retries: Optional[int] = RECONNECT_MAX_RETRIES,
                            backoff_base: float = RECONNECT_BACKOFF_BASE,
                            backoff_max: float = RECONNECT_BACKOFF_MAX) -> Optional[cv2.VideoCapture]:
    """
    Пытается открыть VideoCapture в цикле. Возвращает открытый cap или None (в случае исчерпания max_retries).
    Логи — через __orig_print.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            __orig_print(f"{RECONNECT_LOG_PREFIX} Попытка открыть видео (attempt={attempt}) -> {source}")
            cap = cv2.VideoCapture(source)
            # небольшая пауза чтобы дать OpenCV шанс инициализироваться
            time.sleep(0.5)
            if cap.isOpened():
                __orig_print(f"{RECONNECT_LOG_PREFIX} Успешно подключились на попытке {attempt}")
                return cap
            else:
                __orig_print(f"{RECONNECT_LOG_PREFIX} Не удалось открыть поток на попытке {attempt}")
                try:
                    cap.release()
                except Exception:
                    pass
        except Exception as e:
            __orig_print(f"{RECONNECT_LOG_PREFIX} Исключение при открытии: {e}")

        # если установлен лимит попыток и исчерпан — вернуть None
        if (max_retries is not None) and (attempt >= int(max_retries)):
            __orig_print(f"{RECONNECT_LOG_PREFIX} Достигнут лимит попыток ({max_retries}), прекращаем попытки.")
            return None

        # экспоненциальный backoff
        sleep_for = min(backoff_max, backoff_base * (2 ** (attempt - 1)))
        __orig_print(f"{RECONNECT_LOG_PREFIX} Ждём {sleep_for:.1f}s перед новой попыткой...")
        time.sleep(sleep_for)
# ---------------------------------------------------------------


def download_files_from_minio(local_dir: str = "downloads"):
    """
    Скачивает все файлы из бакета MINIO_BUCKET в локальную папку local_dir.
    Использует настройки из переменных окружения.
    """
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )

    os.makedirs(local_dir, exist_ok=True)

    # список объектов
    objects = client.list_objects(MINIO_BUCKET, recursive=True)
    for obj in objects:
        file_path = os.path.join(local_dir, obj.object_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        client.fget_object(MINIO_BUCKET, obj.object_name, file_path)
        print(f"[OK] Скачан {obj.object_name} → {file_path}")
def main():
    load_notified_db()

    if FaceAnalysis is None:
        __orig_print('[ERROR] insightface.FaceAnalysis not available — проверьте установку insightface')
        return

    try:
        detector_app = FaceAnalysis(name=EMB_MODEL_NAME, allowed_modules=['detection'], providers=['CPUExecutionProvider'])
        detector_app.prepare(ctx_id=0, det_size=(640, 640))
        recognizer_app = FaceAnalysis(name=EMB_MODEL_NAME, allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
        recognizer_app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        __orig_print(f'[ERROR] cannot initialize FaceAnalysis: {e}')
        return

    download_dir = "downloads"
    download_files_from_minio(download_dir)

    # Инициализируем FaceDatabase, указывая путь к папке с db_*.npy
    db = FaceDatabase(Path(download_dir))

    # Открываем захват с ретраем; если не открыли — не выходим, будем пытаться в цикле
    cap = open_capture_with_retry(VIDEO_SOURCE)
    if cap is None or not getattr(cap, "isOpened", lambda: False)():
        __orig_print(f"[ERROR] cannot open video after retries: {VIDEO_SOURCE} — попробуем позже в цикле")
        cap = None

    tracked_persons = {}
    frame_idx = 0
    last_recog_time = 0.0
    force_heavy = False

    try:
        while True:
            try:
                if cap is None:
                    # если нет открытого cap -> пытаемся открыть
                    cap = open_capture_with_retry(VIDEO_SOURCE)
                    if cap is None or not cap.isOpened():
                        # ждём и повторим
                        time.sleep(RECONNECT_BACKOFF_BASE)
                        continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    __orig_print(f"{RECONNECT_LOG_PREFIX} чтение кадра вернуло ret={ret} — переподключаемся")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    # небольшой sleep чтобы не крутить цикл слишком быстро
                    time.sleep(RECONNECT_BACKOFF_BASE)
                    continue

                frame_idx += 1
                h, w = frame.shape[:2]
                now = time.time()

                for t in tracked_persons.values():
                    t.predict()

                heavy_pending = (now - last_recog_time) >= RECOG_INTERVAL_SEC

                if FRAME_SKIP > 1 and (frame_idx % FRAME_SKIP) != 0:
                    if heavy_pending:
                        force_heavy = True
                        continue
                    else:
                        continue

                try:
                    det_faces = detector_app.get(frame)
                except Exception:
                    det_faces = []

                dets = []
                for f in det_faces:
                    try:
                        if getattr(f, 'det_score', 0.0) < DET_CONF_THRESHOLD:
                            continue
                        bx = list(map(int, f.bbox))
                        if (bx[2] - bx[0]) < MIN_FACE_PIXELS or (bx[3] - bx[1]) < MIN_FACE_PIXELS:
                            continue
                        dets.append({'bbox': bx, 'conf': float(getattr(f, 'det_score', 0.0)), 'insight_obj': f, 'track_id': -1})
                    except Exception:
                        continue

                unmatched = list(range(len(dets)))
                for tid, track in list(tracked_persons.items()):
                    best_match_idx = -1; best_iou = IOU_THRESHOLD
                    for i in list(unmatched):
                        try:
                            det_bbox = dets[i]['bbox']
                            iou = calculate_iou(track.bbox, det_bbox)
                            if iou > best_iou:
                                best_iou = iou; best_match_idx = i
                        except Exception:
                            continue
                    if best_match_idx != -1:
                        dets[best_match_idx]['track_id'] = tid
                        try:
                            track.update(dets[best_match_idx]['insight_obj'], frame_image=frame)
                        except Exception:
                            pass
                        if best_match_idx in unmatched:
                            unmatched.remove(best_match_idx)

                for i in list(unmatched):
                    try:
                        f = dets[i]['insight_obj']
                        new_id = next_track_id()
                        dets[i]['track_id'] = new_id
                        p = TrackedPerson(f, frame.shape, track_id=new_id, frame_image=frame)
                        p.created_frame = frame_idx
                        tracked_persons[new_id] = p
                    except Exception:
                        continue

                do_heavy = heavy_pending or force_heavy
                if do_heavy:
                    try:
                        recog_faces = recognizer_app.get(frame)
                    except Exception:
                        recog_faces = []

                    rec_list = []
                    for rf in recog_faces:
                        try:
                            if getattr(rf, 'det_score', 0.0) < DET_CONF_THRESHOLD:
                                continue
                            rx1, ry1, rx2, ry2 = [int(x) for x in rf.bbox]
                            emb = None
                            for cand in ('normed_embedding', 'embedding', 'feat', 'face_embedding', 'face_feat'):
                                if hasattr(rf, cand):
                                    try:
                                        arr = np.array(getattr(rf, cand))
                                        if arr is None:
                                            continue
                                        if arr.ndim > 1:
                                            arr = arr.reshape(-1)
                                        arr = arr.astype(np.float32)
                                        if np.isnan(arr).any() or np.isinf(arr).any():
                                            continue
                                        nrm = float(np.linalg.norm(arr))
                                        if nrm <= 1e-8:
                                            continue
                                        arr = arr / (nrm + 1e-12)
                                        emb = arr
                                        break
                                    except Exception:
                                        continue
                            kps = extract_keypoints_from_face(rf)
                            hair = compute_hairstyle_descriptor(frame, [rx1, ry1, rx2, ry2])
                            rec_list.append({'bbox': [rx1, ry1, rx2, ry2], 'det_score': float(getattr(rf, 'det_score', 0.0)), 'embedding': emb, 'kps': kps, 'hair': hair})
                        except Exception:
                            continue

                    for rec in rec_list:
                        best_tid = None; best_iou = IOU_THRESHOLD
                        for tid, person in tracked_persons.items():
                            try:
                                iou = calculate_iou(person.bbox, rec['bbox'])
                                if iou > best_iou:
                                    best_iou = iou; best_tid = tid
                            except Exception:
                                continue
                        if best_tid is not None:
                            class SimpleFace: pass
                            sf = SimpleFace()
                            sf.bbox = np.array(rec['bbox'], dtype=float)
                            sf.det_score = rec['det_score']
                            sf.normed_embedding = rec['embedding']
                            sf.embedding = rec['embedding']
                            sf.hair = rec['hair']
                            sf.kps = rec['kps']
                            try:
                                tracked_persons[best_tid].update(sf, frame_image=frame)
                                tracked_persons[best_tid].last_recog_attempt_frame = frame_idx
                                try_recognize_person(tracked_persons[best_tid], db, frame_idx, frame)
                            except Exception:
                                pass
                        else:
                            try:
                                new_id = next_track_id()
                                class SimpleFace2: pass
                                sf2 = SimpleFace2()
                                sf2.bbox = np.array(rec['bbox'], dtype=float)
                                sf2.det_score = rec['det_score']
                                sf2.normed_embedding = rec['embedding']
                                sf2.embedding = rec['embedding']
                                sf2.hair = rec['hair']
                                sf2.kps = rec['kps']
                                person = TrackedPerson(sf2, frame.shape, track_id=new_id, frame_image=frame)
                                person.created_frame = frame_idx
                                tracked_persons[new_id] = person
                                person.last_recog_attempt_frame = frame_idx
                                try_recognize_person(person, db, frame_idx, frame)
                            except Exception:
                                continue

                    last_recog_time = time.time()
                    force_heavy = False

                # cleanup stale tracks
                for tid in list(tracked_persons.keys()):
                    p = tracked_persons[tid]
                    if p.missed_frames >= MAX_MISSED_FRAMES:
                        try:
                            if p.fixed_label:
                                with active_label_map_lock:
                                    cur = active_label_map.get(p.fixed_label)
                                    if cur and cur.get('track') is p:
                                        del active_label_map[p.fixed_label]
                        except Exception:
                            pass
                        del tracked_persons[tid]

            except KeyboardInterrupt:
                raise
            except Exception as e:
                # любая ошибка внутри цикла — логируем, освобождаем cap и переподключаемся
                __orig_print(f"{RECONNECT_LOG_PREFIX} Внутренняя ошибка цикла: {e}. Попытка переподключиться.")
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                cap = None
                # даём небольшую паузу перед повтором
                time.sleep(RECONNECT_BACKOFF_BASE)
                continue

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        save_notified_db()


if __name__ == '__main__':
    main()
