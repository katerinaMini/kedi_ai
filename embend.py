#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import logging
import socket
import tempfile
import io
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple, Callable
from urllib.parse import urlparse, urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from requests.exceptions import RequestException
from tqdm import tqdm

# numpy/opencv/insightface imports (embeddings)
import numpy as np
import cv2

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

# --- optional minio / boto3 clients for uploading embeddings ---
MINIO_CLIENT_AVAILABLE = False
MINIO_BOTO3_FALLBACK = False
try:
    from minio import Minio
    MINIO_CLIENT_AVAILABLE = True
except Exception:
    try:
        import boto3
        from botocore.client import Config
        MINIO_BOTO3_FALLBACK = True
    except Exception:
        pass

# -----------------------
# CONFIG
# -----------------------
BASE_ROOT = os.environ.get('KINDY_BASE', 'https://kindy.kz')
KINDERGARTENS_GROUPS = f"{BASE_ROOT}/api/profile/open-api/platform-ai/kindergartens/1/groups"
GROUP_MEMBERS_TPL = f"{BASE_ROOT}/api/profile/open-api/platform-ai/groups/{{group_id}}/members"
CHILD_MEDIA_TPL = f"{BASE_ROOT}/api/profile/open-api/platform-ai/children/{{child_id}}/media"
ATTENDANCE_MARK = f"{BASE_ROOT}/api/profile/open-api/attendance/mark"

DOWNLOAD_DIR = Path(os.environ.get("KINDY_DOWNLOAD_DIR", "downloads"))
HTTP_RETRIES = int(os.environ.get("KINDY_HTTP_RETRIES", 4))
BACKOFF_FACTOR = float(os.environ.get("KINDY_BACKOFF_FACTOR", 0.6))
CONNECT_TIMEOUT = float(os.environ.get("KINDY_CONNECT_TIMEOUT", 5.0))
READ_TIMEOUT = float(os.environ.get("KINDY_READ_TIMEOUT", 60.0))
LOG_FILE = Path(os.environ.get("KINDY_LOG_FILE", "kindy_attendance.log"))

AUTH_HEADERS: Dict[str, str] = {}

MAX_MEDIA_PER_CHILD = int(os.environ.get('KINDY_MAX_MEDIA_PER_CHILD', 2))

INPUT_PATHS = [Path(os.environ.get("KINDY_INPUT_PATH", DOWNLOAD_DIR))]
DB_EMBED_DIR = Path(os.environ.get("KINDY_DB_EMBED_DIR", r'project/embend'))
EMB_MODEL_NAME = os.environ.get("KINDY_EMB_MODEL", 'buffalo_l')

EMBEDDINGS_FILE = DB_EMBED_DIR / 'db_embeddings.npy'
LABELS_FILE     = DB_EMBED_DIR / 'db_labels.npy'
HAIR_FILE       = DB_EMBED_DIR / 'db_hair.npy'
CANDIDATES_FILE = DB_EMBED_DIR / 'candidates_by_person.npy'

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

DET_CONF_THRESHOLD = float(os.environ.get("KINDY_DET_CONF", 0.45))
MIN_FACE_PIXELS    = int(os.environ.get("KINDY_MIN_FACE_PIXELS", 50))
FRAME_INTERVAL     = int(os.environ.get("KINDY_FRAME_INTERVAL", 1))

USE_GPU = os.environ.get("KINDY_USE_GPU", "True") in ("1", "true", "True")
VERBOSE = True

POSE_BINS = (-0.28, 0.28)
HAIR_ENABLED  = True
HAIR_BINS     = 16
PRIORITIES = {
    "frontal": 1.00,
    "left_profile": 0.80,
    "right_profile": 0.80,
    "back": 0.30,
    "hair": 0.20,
}

# MinIO / S3 settings (env)
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "10.201.75.10:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "system_operator")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "71OL9WpdfU0bVjy")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "False").lower() in ("1","true","yes")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "kindy-ai")

# -----------------------
# Logging
# -----------------------
logger = logging.getLogger("kindy_attendance_embdb")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S"))
logger.addHandler(ch)
fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S"))
logger.addHandler(fh)

if os.environ.get('KINDY_DEBUG', '0') in ('1', 'true', 'True'):
    ch.setLevel(logging.DEBUG)

# -----------------------
# HTTP session helper
# -----------------------
def create_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=HTTP_RETRIES, backoff_factor=BACKOFF_FACTOR,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=frozenset(['GET','POST','PUT','DELETE','HEAD','OPTIONS']))
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.setdefault("User-Agent", "kindy-attendance-embdb/1.0")
    s.headers.setdefault("Accept", "*/*")
    s.headers.setdefault("Referer", BASE_ROOT + "/")
    if AUTH_HEADERS:
        s.headers.update(AUTH_HEADERS)
    return s

# -----------------------
# Utilities
# -----------------------
def sanitize_filename(name: str) -> str:
    name = re.sub(r'[^0-9A-Za-zА-Яа-яёЁ _\-\.\(\)]', '_', name)
    return name.strip().replace(' ', '_')

def make_abs_url(url: str, base: str = BASE_ROOT) -> Optional[str]:
    if not url:
        return None
    url = str(url).strip()
    if url.startswith('//'):
        return 'https:' + url
    parsed = urlparse(url)
    if parsed.scheme in ('http', 'https'):
        return url
    return urljoin(base + '/', url.lstrip('/'))

# -----------------------
# Simple face detection (опционально)
# -----------------------
try:
    import cv2 as _cv2
    _cv2_available = True
    try:
        _haar = _cv2.CascadeClassifier(_cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if _haar.empty():
            _cv2_available = False
    except Exception:
        _cv2_available = False
except Exception:
    _cv2_available = False

def recognize_face_default_image_bgr(image_bgr: np.ndarray) -> bool:
    """Простая локальная детекция через Haar (при отсутствии insightface)."""
    if not _cv2_available:
        logger.warning("OpenCV Haar cascades not available.")
        return False
    try:
        gray = _cv2.cvtColor(image_bgr, _cv2.COLOR_BGR2GRAY)
        faces = _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        logger.debug("Detected %d faces in image", len(faces))
        return len(faces) > 0
    except Exception:
        logger.exception("Exception in Haar detect")
        return False

# -----------------------
# Network helpers: groups/members/media/attendance
# -----------------------
def get_groups(session: requests.Session) -> List[Dict[str, Any]]:
    try:
        r = session.get(KINDERGARTENS_GROUPS, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        r.raise_for_status()
        groups = r.json()
        if isinstance(groups, list):
            return groups
        logger.warning("Expected list of groups, got %s", type(groups))
    except Exception:
        logger.exception("Failed to fetch groups")
    return []

def get_group_members(session: requests.Session, group_id: int) -> List[Dict[str, Any]]:
    url = GROUP_MEMBERS_TPL.format(group_id=group_id)
    try:
        r = session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        r.raise_for_status()
        members = r.json()
        if isinstance(members, list):
            return members
        logger.warning("Expected list of members for group %s, got %s", group_id, type(members))
    except Exception:
        logger.exception("Failed to fetch members for group %s", group_id)
    return []

def get_child_media(session: requests.Session, child_id: int) -> List[Dict[str, Any]]:
    url = CHILD_MEDIA_TPL.format(child_id=child_id)
    try:
        r = session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ('media', 'medias', 'files'):
                if isinstance(data.get(k), list):
                    return data.get(k)
        logger.warning("Unexpected child media response for %s: %s", child_id, type(data))
    except Exception:
        logger.exception("Failed to fetch media for child %s", child_id)
    return []

def mark_attendance(session: requests.Session, child_id: int, present: bool = True) -> bool:
    payload = {"childId": int(child_id), "present": bool(present)}
    try:
        r = session.post(ATTENDANCE_MARK, json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        logger.debug("Attendance response: %s %s", r.status_code, r.text[:400])
        r.raise_for_status()
        logger.info("Marked attendance for child %s -> present=%s (status=%s)", child_id, present, r.status_code)
        return True
    except Exception:
        logger.exception("Failed to mark attendance for child %s", child_id)
        return False

# -----------------------
# Embedding helpers & quality metrics (same as before)
# -----------------------
def normalize_vec(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    if n == 0 or np.isnan(n):
        return v
    return (v / n).astype(np.float32)

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

def pose_score_from_kps(face) -> float:
    try:
        kps = np.array(face.kps)
        left_eye_x = float(kps[0, 0]); right_eye_x = float(kps[1, 0]); nose_x = float(kps[2, 0])
        eye_mid = (left_eye_x + right_eye_x) / 2.0
        eye_dist = max(1.0, abs(right_eye_x - left_eye_x))
        return (nose_x - eye_mid) / eye_dist
    except Exception:
        return 0.0

def compute_hairstyle_descriptor(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
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
# Face model init (shared)
# -----------------------
def init_face_app() -> Any:
    if FaceAnalysis is None:
        logger.error("insightface not available. Install insightface to enable robust embedding extraction.")
        return None
    try:
        face_app = FaceAnalysis(name=EMB_MODEL_NAME, allowed_modules=['detection', 'recognition'])
        ctx_id = 0 if USE_GPU else -1
        face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info("InsightFace model loaded: %s (gpu=%s)", EMB_MODEL_NAME, USE_GPU)
        return face_app
    except Exception:
        logger.exception("Failed to initialize FaceAnalysis")
        return None

# -----------------------
# Media processing helpers (no local saving)
# -----------------------
def is_image_url_or_path(url: str) -> bool:
    ext = Path(urlparse(url).path).suffix.lower()
    if ext in IMAGE_EXTS:
        return True
    return False

def is_video_url_or_path(url: str) -> bool:
    ext = Path(urlparse(url).path).suffix.lower()
    if ext in VIDEO_EXTS:
        return True
    return False

def read_image_from_url(session: requests.Session, url: str) -> Optional[np.ndarray]:
    try:
        r = session.get(url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        r.raise_for_status()
        data = r.content
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.debug("cv2.imdecode returned None for %s", url)
        return img
    except Exception:
        logger.exception("Failed to fetch/parse image %s", url)
        return None

def extract_frames_from_video_source(source, frame_step: int = 30, max_frames: int = 0):
    """
    source: path-like or URL string. If URL - pass directly to cv2.VideoCapture (works if OpenCV compiled with ffmpeg)
    Yields rows (idx, frame_bgr)
    """
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        logger.warning("Не удалось открыть видео-источник %s", source)
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

# -----------------------
# Upload embeddings to MinIO (single def)
# -----------------------
def upload_embeddings_to_minio(embeddings_obj: List[Any],
                               labels_obj: List[str],
                               hairs_obj: List[Any],
                               candidates_obj: Dict[str, List[Dict[str, Any]]]) -> bool:
    """
    Сохранение 4-х файлов (npy / candidates) в MinIO/S3 bucket.
    Попытка через minio python client, иначе через boto3.
    """
    DB_EMBED_DIR.mkdir(parents=True, exist_ok=True)
    # prepare byte buffers
    try:
        buf_emb = io.BytesIO()
        np.save(buf_emb, np.array(embeddings_obj, dtype=object), allow_pickle=True)
        buf_emb.seek(0)

        buf_labels = io.BytesIO()
        np.save(buf_labels, np.array(labels_obj, dtype=object), allow_pickle=True)
        buf_labels.seek(0)

        buf_hair = io.BytesIO()
        np.save(buf_hair, np.array(hairs_obj, dtype=object), allow_pickle=True)
        buf_hair.seek(0)

        buf_cand = io.BytesIO()
        np.save(buf_cand, candidates_obj, allow_pickle=True)
        buf_cand.seek(0)
    except Exception:
        logger.exception("Failed to serialize arrays to buffers")
        return False

    def _upload_minio(minio_client):
        try:
            bucket = MINIO_BUCKET
            # ensure bucket exists for Minio client
            try:
                if hasattr(minio_client, 'bucket_exists'):
                    if not minio_client.bucket_exists(bucket):
                        minio_client.make_bucket(bucket)
                # boto3 has different API
            except Exception:
                pass

            # put objects
            minio_client.put_object(bucket, "db_embeddings.npy", buf_emb, length=buf_emb.getbuffer().nbytes, content_type="application/octet-stream")
            buf_emb.seek(0)
            minio_client.put_object(bucket, "db_labels.npy", buf_labels, length=buf_labels.getbuffer().nbytes, content_type="application/octet-stream")
            buf_labels.seek(0)
            minio_client.put_object(bucket, "db_hair.npy", buf_hair, length=buf_hair.getbuffer().nbytes, content_type="application/octet-stream")
            buf_hair.seek(0)
            minio_client.put_object(bucket, "candidates_by_person.npy", buf_cand, length=buf_cand.getbuffer().nbytes, content_type="application/octet-stream")
            buf_cand.seek(0)
            return True
        except Exception:
            logger.exception("Minio client upload failed")
            return False

    def _upload_boto3(s3_client):
        try:
            bucket = MINIO_BUCKET
            # ensure bucket
            try:
                s3_client.head_bucket(Bucket=bucket)
            except Exception:
                try:
                    s3_client.create_bucket(Bucket=bucket)
                except Exception:
                    pass

            s3_client.put_object(Bucket=bucket, Key="db_embeddings.npy", Body=buf_emb.getvalue())
            s3_client.put_object(Bucket=bucket, Key="db_labels.npy", Body=buf_labels.getvalue())
            s3_client.put_object(Bucket=bucket, Key="db_hair.npy", Body=buf_hair.getvalue())
            s3_client.put_object(Bucket=bucket, Key="candidates_by_person.npy", Body=buf_cand.getvalue())
            return True
        except Exception:
            logger.exception("boto3 upload failed")
            return False

    # Try minio client
    if MINIO_CLIENT_AVAILABLE:
        try:
            client = Minio(MINIO_ENDPOINT,
                           access_key=MINIO_ACCESS_KEY,
                           secret_key=MINIO_SECRET_KEY,
                           secure=MINIO_SECURE)
            # minio put_object expects file-like with length; our buffers support getbuffer()
            success = _upload_minio(client)
            if success:
                logger.info("Uploaded embeddings to MinIO (minio client) bucket=%s", MINIO_BUCKET)
                return True
        except Exception:
            logger.exception("Minio client usage failed")

    # Fallback: boto3
    if MINIO_BOTO3_FALLBACK:
        try:
            s3 = boto3.client('s3',
                              endpoint_url=("https://" if MINIO_SECURE else "http://") + MINIO_ENDPOINT,
                              aws_access_key_id=MINIO_ACCESS_KEY or None,
                              aws_secret_access_key=MINIO_SECRET_KEY or None,
                              config=Config(signature_version='s3v4'))
            success = _upload_boto3(s3)
            if success:
                logger.info("Uploaded embeddings to MinIO (boto3) bucket=%s", MINIO_BUCKET)
                return True
        except Exception:
            logger.exception("boto3 fallback failed")

    # Final fallback: save locally and warn
    try:
        np.save(EMBEDDINGS_FILE, np.array(embeddings_obj, dtype=object), allow_pickle=True)
        np.save(LABELS_FILE, np.array(labels_obj, dtype=object), allow_pickle=True)
        np.save(HAIR_FILE, np.array(hairs_obj, dtype=object), allow_pickle=True)
        np.save(CANDIDATES_FILE, candidates_obj, allow_pickle=True)
        logger.warning("MinIO upload not available — saved embeddings locally to %s", DB_EMBED_DIR)
        return True
    except Exception:
        logger.exception("Failed to save fallback local files")
        return False

# -----------------------
# Main combined pipeline (no local saving of media)
# -----------------------
def run_all():
    """
    Выполняет:
     - чтение групп / участников
     - для каждого участника: получение media URLs -> обработка в памяти/по URL -> формирование кандидатов
     - пост-обработка кандидатов -> формирование 5 прототипов на человека
     - загрузка результатов в MinIO
    """
    if FaceAnalysis is None:
        logger.error("insightface не установлен - нужно установить insightface для извлечения эмбеддингов.")
        return

    face_app = init_face_app()
    if face_app is None:
        logger.error("Не удалось инициализировать face_app")
        return

    session = create_session()
    groups = get_groups(session)
    if not groups:
        logger.error("Groups list empty — выходим")
        return

    group_ids = []
    for g in groups:
        if isinstance(g, dict) and g.get('id'):
            group_ids.append(int(g.get('id')))
        elif isinstance(g, int):
            group_ids.append(g)
        else:
            logger.debug("Unknown group entry format: %s", g)

    logger.info("Found %d groups: %s", len(group_ids), group_ids)

    # Собираем кандидатов в памяти: map person_name -> list[candidate dict]
    all_candidates: Dict[str, List[Dict[str, Any]]] = {}

    def process_frame(person_name: str, frame_bgr: np.ndarray):
        faces = face_app.get(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        h_img, w_img = frame_bgr.shape[:2]
        for face in faces:
            if face.det_score < DET_CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = face.bbox
            if (x2 - x1) * (y2 - y1) < MIN_FACE_PIXELS ** 2:
                continue

            bw, bh = int(x2 - x1), int(y2 - y1)
            mx, my = int(bw * 0.15), int(bh * 0.15)
            px1 = max(0, int(x1) - mx)
            py1 = max(0, int(y1) - my)
            px2 = min(w_img, int(x2) + mx)
            py2 = min(h_img, int(y2) + my)
            face_crop = frame_bgr[py1:py2, px1:px2].copy() if (py2 > py1 and px2 > px1) else None

            quality = calculate_quality_score(face, frame_bgr)
            if quality <= 0:
                continue
            emb = normalize_vec(face.normed_embedding.astype(np.float32))
            pose = pose_score_from_kps(face)
            hair = compute_hairstyle_descriptor(frame_bgr, face.bbox) if HAIR_ENABLED else None

            candidate = {
                "emb": emb,
                "weight": float(quality),
                "pose": float(pose),
                "hair": hair,
            }
            if face_crop is not None:
                candidate["face_crop_shape"] = face_crop.shape

            all_candidates.setdefault(person_name, []).append(candidate)

    # Для каждой группы и участника: берём media URLs и обрабатываем
    for gid in group_ids:
        logger.info("Processing group %s", gid)
        members = get_group_members(session, gid)
        if not members:
            logger.info("No members for group %s", gid)
            continue

        members_with_photo = [m for m in members if m.get('photoPath')]
        logger.info("Group %s: %d members, %d with photoPath", gid, len(members), len(members_with_photo))

        for member in members_with_photo:
            try:
                child_id = member.get('id') or member.get('childId') or member.get('child_id')
                if child_id is None:
                    logger.debug("Member without id: %s", member)
                    continue
                child_id = int(child_id)
                name = member.get('name') or member.get('fullName') or f'child_{child_id}'
                safe_name = sanitize_filename(str(name))
                person_key = f"{child_id}_{safe_name}"

                medias = get_child_media(session, child_id)
                if not medias:
                    logger.info("No media for child %s (%s)", child_id, name)
                    continue

                # extract urls
                urls = []
                for it in medias:
                    if isinstance(it, dict):
                        for k in ('url','path','src','file','photoPath'):
                            if it.get(k):
                                urls.append(it.get(k))
                                break
                        for v in it.values():
                            if isinstance(v, str) and v.startswith('http'):
                                urls.append(v)
                    elif isinstance(it, str):
                        urls.append(it)

                seen = set()
                normalized = []
                for u in urls:
                    abs_u = make_abs_url(u)
                    if not abs_u:
                        continue
                    if abs_u in seen:
                        continue
                    seen.add(abs_u)
                    normalized.append(abs_u)

                if not normalized:
                    logger.info("No normalized media URLs for child %s", child_id)
                    continue

                recognized = False
                for i, url in enumerate(normalized[:MAX_MEDIA_PER_CHILD], start=1):
                    try:
                        if is_image_url_or_path(url):
                            img = read_image_from_url(session, url)
                            if img is None:
                                logger.warning("Failed to read image %s", url)
                                continue
                            process_frame(person_key, img)
                            recognized = True
                        elif is_video_url_or_path(url):
                            # Try to open video by URL (no local saving)
                            frames_gen = extract_frames_from_video_source(url, frame_step=FRAME_INTERVAL, max_frames=10)
                            any_frame = False
                            for _, frame_bgr in frames_gen:
                                any_frame = True
                                process_frame(person_key, frame_bgr)
                            if any_frame:
                                recognized = True
                            else:
                                logger.debug("No frames read from video URL %s", url)
                        else:
                            # Try HEAD / Content-Type to decide
                            try:
                                head = session.head(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
                                ctype = (head.headers.get('content-type') or '').lower()
                                if 'image' in ctype:
                                    img = read_image_from_url(session, url)
                                    if img is None:
                                        continue
                                    process_frame(person_key, img)
                                    recognized = True
                                elif 'video' in ctype:
                                    frames_gen = extract_frames_from_video_source(url, frame_step=FRAME_INTERVAL, max_frames=10)
                                    any_frame = False
                                    for _, frame_bgr in frames_gen:
                                        any_frame = True
                                        process_frame(person_key, frame_bgr)
                                    if any_frame:
                                        recognized = True
                                else:
                                    # fallback: try as image
                                    img = read_image_from_url(session, url)
                                    if img is not None:
                                        process_frame(person_key, img)
                                        recognized = True
                            except Exception:
                                logger.debug("HEAD failed for %s, trying GET as image", url)
                                img = read_image_from_url(session, url)
                                if img is not None:
                                    process_frame(person_key, img)
                                    recognized = True
                    except Exception:
                        logger.exception("Error processing media %s for child %s", url, child_id)

                if not recognized:
                    logger.info("Child %s: no usable media processed (group %s)", child_id, gid)

            except Exception:
                logger.exception("Error processing member: %s", member)

    # Сохраним candidates в MinIO (или локально как fallback)
    try:
        np.save(CANDIDATES_FILE, all_candidates, allow_pickle=True)
        logger.info("Saved candidates locally: %s", CANDIDATES_FILE)
    except Exception:
        logger.exception("Failed to save local candidates")

    # --- Формирование 5 прототипов на человека (как раньше) ---
    new_embeddings: List[Optional[np.ndarray]] = []
    new_labels: List[str] = []
    new_hairs: List[Optional[np.ndarray]] = []

    def hair_average(items: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        hs = [c['hair'] for c in items if c.get('hair') is not None]
        if not hs:
            return None
        return np.mean(np.stack(hs, axis=0), axis=0).astype(np.float32)

    logger.info("Computing 5 prototypes per person...")
    for person_name, cands in all_candidates.items():
        if not cands:
            continue

        left_items, frontal_items, right_items = [], [], []
        for c in cands:
            p = c.get('pose', 0.0)
            if p < POSE_BINS[0]:
                left_items.append(c)
            elif p > POSE_BINS[1]:
                right_items.append(c)
            else:
                frontal_items.append(c)

        def centroid(items: List[Dict[str, Any]]) -> Optional[np.ndarray]:
            if not items:
                return None
            return weighted_avg([it['emb'] for it in items], [it['weight'] for it in items])

        frontal_emb = centroid(frontal_items)
        if frontal_emb is None:
            frontal_emb = centroid(cands)

        hair_avg_all = hair_average(cands)

        left_emb = centroid(left_items)
        if left_emb is None:
            left_emb = frontal_emb
        right_emb = centroid(right_items)
        if right_emb is None:
            right_emb = frontal_emb

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
        new_embeddings.append(frontal_emb if frontal_emb is not None else np.zeros((512,), dtype=np.float32))
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

        # 4) back (hair-only)
        new_embeddings.append(None)
        new_labels.append(f"{person_name}__back")
        new_hairs.append(hair_avg_all)

        # 5) hair (hair-only)
        new_embeddings.append(None)
        new_labels.append(f"{person_name}__hair")
        new_hairs.append(hair_avg_all)

    # Merge with existing DB if present (optional)
    try:
        existing_embeddings = list(np.load(EMBEDDINGS_FILE, allow_pickle=True)) if EMBEDDINGS_FILE.exists() else []
        existing_labels     = list(np.load(LABELS_FILE, allow_pickle=True)) if LABELS_FILE.exists() else []
        existing_hairs      = list(np.load(HAIR_FILE, allow_pickle=True)) if HAIR_FILE.exists() else []
    except Exception:
        existing_embeddings, existing_labels, existing_hairs = [], [], []

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
        logger.warning("No labels to save - aborting.")
        return

    # Clean conflicts & prune as in original code (functions re-used)
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
            return embs_list, labels_list

        E = np.stack([embs_list[i] for i in face_idxs], axis=0).astype(np.float32)
        sims = np.dot(E, E.T)

        idx_map = {k: i for i, k in enumerate(face_idxs)}
        remove = set()

        root_to_idxs = {}
        for gi in face_idxs:
            root = str(labels_list[gi]).split('__')[0]
            root_to_idxs.setdefault(root, []).append(gi)

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

        keep_all = set(hair_only_idxs) | keep_face
        embs_f   = [embs_list[i]  for i in range(len(embs_list)) if i in keep_all]
        labels_f = [labels_list[i] for i in range(len(labels_list)) if i in keep_all]
        return embs_f, labels_f

    all_embeddings, all_labels = clean_conflicts(all_embeddings, all_labels, sim_thresh=0.90)
    all_embeddings, all_labels = prune_non_unique(all_embeddings, all_labels, intra_min=0.35, inter_max=0.88)

    # prepare hair list for final labels
    label_to_hair = {str(l): h for l, h in zip(filtered_existing_labels + new_labels,
                                               filtered_existing_hairs + new_hairs)}
    all_hairs = [label_to_hair.get(str(l), None) for l in all_labels]

    # Upload to MinIO (single function)
    ok = upload_embeddings_to_minio(all_embeddings, all_labels, all_hairs, all_candidates)
    if ok:
        logger.info("Embeddings pipeline finished and uploaded to MinIO.")
    else:
        logger.error("Embeddings pipeline finished but upload failed.")

# -----------------------
# Entry point: simple run without CLI
# -----------------------
if __name__ == '__main__':
    try:
        run_all()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Unhandled exception in main")
