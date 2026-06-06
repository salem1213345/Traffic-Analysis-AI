"""
=============================================================================
Kural Tabanlı Karar Destek Sistemi ile Video Tabanlı Trafik Araç Takibi
=============================================================================
Yazılım Mühendisliği – Final Ödevi
Açıklama : MOT17 veri kümesi üzerinde GMM segmentasyonu, SORT takibi ve
           kural tabanlı karar destek sistemi uygulaması.
Gereksinimler: pip install opencv-python numpy scipy
=============================================================================
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# بديل خفيف لـ filterpy.kalman.KalmanFilter حتى لا تحتاج تثبيت filterpy
class KalmanFilter:
    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1), dtype=float)
        self.F = np.eye(dim_x, dtype=float)
        self.H = np.zeros((dim_z, dim_x), dtype=float)
        self.P = np.eye(dim_x, dtype=float)
        self.R = np.eye(dim_z, dtype=float)
        self.Q = np.eye(dim_x, dtype=float)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(self.dim_z, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x = self.x + (K @ y)
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import csv
import time
import os
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. VERİ YAPILARI
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Tespit edilen nesnenin sınırlayıcı kutusu."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2])


@dataclass
class TrackedVehicle:
    """Takip edilen araç nesnesi."""
    track_id: int
    bbox: BoundingBox
    speed_kmh: float = 0.0
    lane_id: int = 0
    frames_tracked: int = 0
    frames_missing: int = 0
    is_confirmed: bool = False
    position_history: List[Tuple[float, float]] = field(default_factory=list)

    def update_history(self):
        self.position_history.append((self.bbox.cx, self.bbox.cy))
        if len(self.position_history) > 30:
            self.position_history.pop(0)


@dataclass
class TrafficAlert:
    """Üretilen trafik uyarısı."""
    frame_id: int
    vehicle_id: int
    rule_id: str
    description: str
    severity: str   # KRİTİK | YÜKSEK | ORTA | DÜŞÜK
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# 2. VİDEO SEGMENTASYON MODÜLİ
# ─────────────────────────────────────────────────────────────────────────────

class VideoSegmenter:
    """
    GMM (MOG2) tabanlı arka plan çıkarımı ve araç segmentasyonu.

    Algoritma:
        1. Gauss bulanıklaştırma ile gürültü azaltma
        2. MOG2 ile arka plan / ön plan ayrıştırma
        3. Morfolojik işlemlerle maske iyileştirme
        4. Kontur tespiti ile bounding box çıkarımı
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (1280, 720),
        learning_rate: float = 0.005,
        min_contour_area: int = 1800,
        aspect_ratio_range: Tuple[float, float] = (1.15, 6.50),
        road_roi_polygon: Optional[List[Tuple[int, int]]] = None,
    ):
        self.target_size = target_size
        self.learning_rate = learning_rate
        self.min_contour_area = min_contour_area
        self.aspect_ratio_range = aspect_ratio_range

        # منطقة الطريق فقط بعد التحويل إلى 1280x720.
        # أي حركة خارج هذه المنطقة لن تُقرأ نهائياً، وهذا يمنع قراءة الناس والجمهور.
        self.road_roi_polygon = road_roi_polygon or [
            # ROI مخصص لفيديو الطريق السريع: الطريق فقط، بدون سماء/أشجار قدر الإمكان
            (35, 250), (1280, 165), (1280, 720), (0, 720)
        ]

        # MOG2 arka plan çıkarıcı (K=5, gölge tespiti aktif)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=16,
            detectShadows=True
        )

        # Morfolojik işlem çekirdekleri
        self.kernel_erode  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _road_roi_mask(self) -> np.ndarray:
        """ينشئ ماسك لمنطقة الطريق فقط."""
        w, h = self.target_size
        roi = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(self.road_roi_polygon, dtype=np.int32)
        cv2.fillPoly(roi, [pts], 255)
        return roi

    def _inside_road_roi(self, bbox: BoundingBox) -> bool:
        """يتأكد أن مركز الجسم داخل منطقة الطريق."""
        roi = self._road_roi_mask()
        cx = int(np.clip(bbox.cx, 0, self.target_size[0] - 1))
        cy = int(np.clip(bbox.cy, 0, self.target_size[1] - 1))
        return roi[cy, cx] == 255

    def _looks_like_vehicle(self, x: int, y: int, w: int, h: int, area: float) -> bool:
        """فلتر قوي: يقبل شكل السيارة ويرفض شكل الإنسان/الرأس/الأقدام/الحواجز."""
        if h <= 0 or w <= 0:
            return False

        aspect = w / (h + 1e-6)
        rect_area = w * h
        fill_ratio = area / max(rect_area, 1)

        # الأشخاص غالباً: طوال ورفيعون، والسيارة غالباً أعرض من طولها.
        if aspect < self.aspect_ratio_range[0] or aspect > self.aspect_ratio_range[1]:
            return False

        # تجاهل أي صندوق صغير جداً أو رفيع جداً.
        if area < self.min_contour_area or w < 45 or h < 22:
            return False

        # تجاهل شكل الإنسان: أي جسم أطول من عرضه غالباً شخص/لوحة/عمود وليس سيارة.
        if h >= w:
            return False

        # تجاهل شكل الإنسان بشكل إضافي: ارتفاع كبير مع عرض صغير.
        if h > 1.35 * w:
            return False

        # تجاهل الخطوط/الكتابات/الظلال الرفيعة.
        if fill_ratio < 0.18:
            return False

        return True

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Kareyi 1280x720'ye yeniden boyutlandır ve Gauss filtresi uygula."""
        resized = cv2.resize(frame, self.target_size)
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        return blurred

    def extract_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """GMM ile ön plan maskesi oluştur ve morfolojik işlem uygula."""
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)

        # Gölgeleri (127) sil → yalnızca hareketli ön plan (255) kalsın
        _, binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morfoloji: Erozyon → Genişletme → Closing
        mask = cv2.erode(binary,  self.kernel_erode,  iterations=1)
        mask = cv2.dilate(mask,   self.kernel_dilate, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        # الأهم: لا نسمح بأي كشف خارج منطقة الطريق.
        mask = cv2.bitwise_and(mask, self._road_roi_mask())

        return mask

    def detect_vehicles(self, mask: np.ndarray) -> List[BoundingBox]:
        """Maskeden araç adayı bounding box listesi çıkar."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[BoundingBox] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if not self._looks_like_vehicle(x, y, w, h, area):
                continue

            bbox = BoundingBox(
                x1=float(x), y1=float(y),
                x2=float(x + w), y2=float(y + h)
            )

            if not self._inside_road_roi(bbox):
                continue

            detections.append(bbox)

        return detections

    def process_frame(self, frame: np.ndarray) -> Tuple[List[BoundingBox], np.ndarray, np.ndarray]:
        """Tam işlem hattı: ham kare → tespitler."""
        preprocessed = self.preprocess(frame)
        mask = self.extract_foreground_mask(preprocessed)
        detections = self.detect_vehicles(mask)
        return detections, preprocessed, mask


# ─────────────────────────────────────────────────────────────────────────────
# 3. SORT TAKİP ALGORITMASI
# ─────────────────────────────────────────────────────────────────────────────

def iou(bb1: np.ndarray, bb2: np.ndarray) -> float:
    """İki bounding box arasındaki IoU (Intersection over Union) hesapla."""
    xi1 = max(bb1[0], bb2[0])
    yi1 = max(bb1[1], bb2[1])
    xi2 = min(bb1[2], bb2[2])
    yi2 = min(bb1[3], bb2[3])

    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter_area == 0:
        return 0.0

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    return inter_area / (bb1_area + bb2_area - inter_area + 1e-6)


class VehicleKalmanTracker:
    """
    Tek araç için Kalman filtresi tabanlı durum tahmincisi.

    Durum vektörü: [x1, y1, x2, y2, vx, vy, vs]
    Ölçüm vektörü: [x1, y1, x2, y2]
    """

    _id_counter = 0

    def __init__(self, bbox: BoundingBox):
        VehicleKalmanTracker._id_counter += 1
        self.id = VehicleKalmanTracker._id_counter
        self.frames_since_update = 0
        self.hit_streak = 0
        self.hits = 1

        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([  # Durum geçiş matrisi
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)
        kf.H = np.array([  # Ölçüm matrisi
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)

        kf.R[2:,2:] *= 10.0   # Ölçüm gürültüsü kovaryansı
        kf.P[4:,4:] *= 1000.0 # Başlangıç belirsizliği (hız)
        kf.Q[-1,-1] *= 0.01   # Süreç gürültüsü

        kf.x[:4] = np.array([
            [bbox.x1], [bbox.y1], [bbox.x2], [bbox.y2]
        ])
        self.kf = kf

    def predict(self) -> np.ndarray:
        """Bir sonraki konumu tahmin et."""
        self.kf.predict()
        self.frames_since_update += 1
        if self.frames_since_update == 1:
            self.hit_streak = 0
        return self.kf.x[:4].flatten()

    def update(self, bbox: BoundingBox):
        """Yeni tespitle Kalman filtresini güncelle."""
        self.frames_since_update = 0
        self.hit_streak += 1
        self.hits += 1
        z = np.array([[bbox.x1], [bbox.y1], [bbox.x2], [bbox.y2]])
        self.kf.update(z)

    def get_state(self) -> np.ndarray:
        return self.kf.x[:4].flatten()


class SORTTracker:
    """
    SORT: Simple Online and Realtime Tracking
    Kaynak: Bewley et al. (2016)

    Parametreler:
        max_age    : İz kaybedilmeden önceki maksimum kare sayısı
        min_hits   : İz onaylanması için gereken minimum tespit sayısı
        iou_threshold: Eşleşme için minimum IoU değeri
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[VehicleKalmanTracker] = []
        self.frame_count = 0

    def update(self, detections: List[BoundingBox]) -> List[Tuple[int, BoundingBox]]:
        """
        Tespitleri mevcut izlerle eşleştir ve güncellenmiş izleri döndür.

        Döndürür: [(track_id, BoundingBox), ...]
        """
        self.frame_count += 1

        # 1. Mevcut izleri tahmin et
        predictions: List[np.ndarray] = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            if np.any(np.isnan(pred)):
                to_del.append(i)
            else:
                predictions.append(pred)
        for i in reversed(to_del):
            self.trackers.pop(i)

        # 2. IoU matrisi oluştur ve Macar algoritmasıyla eşleştir
        matched_trk_ids = set()
        matched_det_ids = set()

        if predictions and detections:
            iou_matrix = np.zeros((len(predictions), len(detections)))
            for t, pred in enumerate(predictions):
                for d, det in enumerate(detections):
                    iou_matrix[t, d] = iou(pred, det.to_array())

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    self.trackers[r].update(detections[c])
                    matched_trk_ids.add(r)
                    matched_det_ids.add(c)

        # 3. Eşleşmeyen tespitler için yeni izler oluştur
        for d, det in enumerate(detections):
            if d not in matched_det_ids:
                self.trackers.append(VehicleKalmanTracker(det))

        # 4. Kayıp izleri temizle
        self.trackers = [
            trk for trk in self.trackers
            if trk.frames_since_update <= self.max_age
        ]

        # 5. Onaylanmış izleri döndür
        results: List[Tuple[int, BoundingBox]] = []
        for trk in self.trackers:
            if trk.frames_since_update == 0 and (
                trk.hits >= self.min_hits or self.frame_count <= self.min_hits
            ):
                state = trk.get_state()
                bbox = BoundingBox(
                    x1=state[0], y1=state[1],
                    x2=state[2], y2=state[3]
                )
                results.append((trk.id, bbox))

        return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. ÖZELLİK ÇIKARIM MODÜLİ
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    İz geçmişinden trafik özellikleri hesaplar.

    ملاحظة مهمة:
    حساب السرعة من فيديو عادي بدون معايرة كاميرا حقيقية لن يكون دقيقاً 100%.
    لذلك استخدمت هنا معايرة عملية مناسبة لفيديو الطريق السريع عندك:
    - نحسب الحركة على عدة فريمات بدل فريم واحد حتى لا تظهر 1 km/h بسبب الاهتزاز.
    - نضيف معامل منظور: السيارة البعيدة تتحرك بكسلات أقل، لذلك نرفع معاملها أكثر.
    """

    # غيّر هذا الرقم إذا أردت السرعات أعلى/أقل:
    # 1.0 = الافتراضي، 1.2 يرفع السرعة 20%، 0.8 يخفضها 20%.
    SPEED_SCALE = 0.75

    # عدد الفريمات المستخدمة لحساب السرعة. كلما زاد الرقم صارت السرعة أثبت.
    SPEED_WINDOW = 8

    def __init__(
        self,
        fps: float = 30.0,
        frame_width: int = 1280,
        frame_height: int = 720,
        lane_boundaries: Optional[List[int]] = None,
        traffic_light_rois: Optional[List[Tuple[int,int,int,int]]] = None,
    ):
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        # للتوافق مع قواعد R03 القديمة داخل الكود
        self.PIXEL_TO_KMH = 1.0

        self.lane_boundaries = lane_boundaries or [0, 320, 640, 960, 1280]
        self.traffic_light_rois = traffic_light_rois or [(1100, 50, 1200, 200)]
        self.position_buffer: Dict[int, List[Tuple[float, float, int]]] = {}
        self.speed_smooth: Dict[int, float] = {}
        self.traffic_light_state: str = "red"

    def update_position(self, track_id: int, cx: float, cy: float, frame_id: int):
        if track_id not in self.position_buffer:
            self.position_buffer[track_id] = []
        self.position_buffer[track_id].append((cx, cy, frame_id))
        if len(self.position_buffer[track_id]) > 30:
            self.position_buffer[track_id].pop(0)

    def _perspective_kmh_per_pixel_frame(self, cy: float) -> float:
        """
        معامل تحويل متغير حسب مكان السيارة في الصورة.
        أعلى الصورة = بعيد = بكسلات قليلة = معامل أكبر.
        أسفل الصورة = قريب = بكسلات أكثر = معامل أقل.
        """
        y = float(np.clip(cy, 120, self.frame_height))
        y_norm = y / max(self.frame_height, 1)

        # هذه القيم مضبوطة لتقريب سرعات طريق سريع بدل 1-4 km/h.
        # إذا ظهرت السرعات عالية جداً قلل الأرقام، وإذا منخفضة ارفعها.
        near_factor = 6.0    # عند أسفل الصورة - مناسب للسيارات القريبة
        far_factor  = 22.0   # عند أعلى/بعيد الصورة - مناسب للسيارات البعيدة
        factor = near_factor + (far_factor - near_factor) * (1.0 - y_norm)
        return factor * self.SPEED_SCALE

    def compute_speed(self, track_id: int) -> float:
        """حساب سرعة أكثر منطقية باستخدام متوسط الحركة + تصحيح المنظور."""
        buf = self.position_buffer.get(track_id, [])
        if len(buf) < 2:
            return 0.0

        # نستخدم نقطة أقدم بدل آخر فريم فقط لتقليل أثر الاهتزاز.
        n = min(self.SPEED_WINDOW, len(buf) - 1)
        cx1, cy1, f1 = buf[-1 - n]
        cx2, cy2, f2 = buf[-1]
        frame_delta = max(f2 - f1, 1)

        pixel_dist_total = float(np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2))
        pixel_per_frame = pixel_dist_total / frame_delta

        avg_y = (cy1 + cy2) / 2.0
        kmh = pixel_per_frame * self._perspective_kmh_per_pixel_frame(avg_y) * (self.fps / 30.0)

        # فلترة ناعمة حتى الرقم لا يقفز كثيراً بين الفريمات.
        old = self.speed_smooth.get(track_id, kmh)
        smooth = 0.65 * old + 0.35 * kmh

        # حدود منطقية لطريق سريع؛ تمنع 1 km/h وتمنع القفزات الخرافية.
        if smooth < 3.0:
            smooth = 0.0
        smooth = float(np.clip(smooth, 0.0, 160.0))
        self.speed_smooth[track_id] = smooth
        return smooth

    def get_lane(self, cx: float) -> int:
        """Merkez x koordinatından şerit numarasını döndür."""
        for i in range(len(self.lane_boundaries) - 1):
            if self.lane_boundaries[i] <= cx < self.lane_boundaries[i+1]:
                return i + 1
        return -1

    def is_in_traffic_light_zone(self, bbox: BoundingBox) -> bool:
        """Araç bounding box'ı trafik ışığı ROI ile kesişiyor mu?"""
        for (rx1, ry1, rx2, ry2) in self.traffic_light_rois:
            ix1 = max(bbox.x1, rx1)
            iy1 = max(bbox.y1, ry1)
            ix2 = min(bbox.x2, rx2)
            iy2 = min(bbox.y2, ry2)
            if ix2 > ix1 and iy2 > iy1:
                return True
        return False

    def compute_density(self, vehicles: List[TrackedVehicle],
                        region: Tuple[int,int,int,int]) -> float:
        """Belirli bölgedeki araç yoğunluğunu (araç/100m) hesapla."""
        rx1, ry1, rx2, ry2 = region
        count = sum(
            1 for v in vehicles
            if rx1 <= v.bbox.cx <= rx2 and ry1 <= v.bbox.cy <= ry2
        )
        region_width_m = (rx2 - rx1) * 0.1  # yaklaşık piksel → metre
        return (count / max(region_width_m, 1)) * 100


# ─────────────────────────────────────────────────────────────────────────────
# 5. KURAL TABANLI KARAR DESTEK SİSTEMİ
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedDSS:
    """
    Trafik izleme için kural tabanlı karar destek sistemi.

    Çıkarım yöntemi : İleri zincirleme (forward chaining)
    Öncelik sırası  : KRİTİK → YÜKSEK → ORTA → DÜŞÜK
    """

    # Eşik değerleri (yapılandırılabilir)
    SPEED_LIMIT_URBAN     = 120.0  # km/h – مرفوعة لأن الفيديو طريق سريع وليس شارع مدينة
    SPEED_LIMIT_HIGHWAY   = 130.0  # km/h – حد طريق سريع تقريبي
    DENSITY_CONGESTION    = 15.0   # araç/100m – tıkanıklık eşiği
    DENSITY_QUEUE         = 10.0   # araç/100m – kuyruk eşiği
    QUEUE_SPEED           = 10.0   # km/h – kuyruk hız eşiği
    STOP_DURATION_FRAMES  = 900    # kare (~30 sn @ 30FPS) – yasadışı park
    RED_LIGHT_MOVEMENT    = 5.0    # piksel/kare – kırmızı ışık hareketi eşiği

    def __init__(self, feature_extractor: FeatureExtractor):
        self.fe = feature_extractor
        self.alerts: List[TrafficAlert] = []
        self.stop_frame_counter: Dict[int, int] = {}

    def evaluate(
        self,
        frame_id: int,
        vehicles: List[TrackedVehicle],
        density: float,
        region: str = "highway",
    ) -> List[TrafficAlert]:
        """Tüm kuralları değerlendir ve uyarı listesi döndür."""
        new_alerts: List[TrafficAlert] = []

        speed_limit = (
            self.SPEED_LIMIT_HIGHWAY if region == "highway"
            else self.SPEED_LIMIT_URBAN
        )

        for v in vehicles:
            # ── R01: Hız İhlali ──────────────────────────────────────────
            if v.speed_kmh > speed_limit:
                new_alerts.append(TrafficAlert(
                    frame_id=frame_id,
                    vehicle_id=v.track_id,
                    rule_id="R01",
                    description=f"HIZ IHLALI: {v.speed_kmh:.1f} km/h > {speed_limit} km/h limiti",
                    severity="YUKSEK"
                ))

            # ── R02: Şerit İhlali ─────────────────────────────────────────
            valid_lanes = {1, 2, 3, 4}
            if v.lane_id not in valid_lanes and v.lane_id != 0:
                new_alerts.append(TrafficAlert(
                    frame_id=frame_id,
                    vehicle_id=v.track_id,
                    rule_id="R02",
                    description=f"SERIT IHLALI: Arac ID:{v.track_id} gecersiz serit",
                    severity="YUKSEK"
                ))

            # ── R03: Kırmızı Işık İhlali ──────────────────────────────────
            if (self.fe.traffic_light_state == "red"
                    and self.fe.is_in_traffic_light_zone(v.bbox)
                    and v.speed_kmh > self.RED_LIGHT_MOVEMENT * self.fe.PIXEL_TO_KMH):
                new_alerts.append(TrafficAlert(
                    frame_id=frame_id,
                    vehicle_id=v.track_id,
                    rule_id="R03",
                    description=f"KIRMIZI ISIK IHLALI: Arac ID:{v.track_id}",
                    severity="KRITIK"
                ))

            # ── R05: Yasadışı Park ────────────────────────────────────────
            if v.speed_kmh < 2.0:
                self.stop_frame_counter[v.track_id] = (
                    self.stop_frame_counter.get(v.track_id, 0) + 1
                )
            else:
                self.stop_frame_counter[v.track_id] = 0

            if self.stop_frame_counter.get(v.track_id, 0) > self.STOP_DURATION_FRAMES:
                new_alerts.append(TrafficAlert(
                    frame_id=frame_id,
                    vehicle_id=v.track_id,
                    rule_id="R05",
                    description=f"YASADISI PARK: Arac ID:{v.track_id} 30+ sn durdu",
                    severity="DUSUK"
                ))

        # ── R04: Tıkanıklık ───────────────────────────────────────────────
        if density > self.DENSITY_CONGESTION:
            new_alerts.append(TrafficAlert(
                frame_id=frame_id,
                vehicle_id=-1,
                rule_id="R04",
                description=f"TIKANIKLIK: Yogunluk {density:.1f} arac/100m",
                severity="ORTA"
            ))

        # ── R06: Kuyruk Tespiti ───────────────────────────────────────────
        slow_count = sum(1 for v in vehicles if v.speed_kmh < self.QUEUE_SPEED)
        if slow_count >= 3 and density > self.DENSITY_QUEUE:
            new_alerts.append(TrafficAlert(
                frame_id=frame_id,
                vehicle_id=-1,
                rule_id="R06",
                description=f"KUYRUK: {slow_count} arac yavaş ilerliyor",
                severity="ORTA"
            ))

        # Öncelik sırasına göre sırala
        priority = {"KRITIK": 0, "YUKSEK": 1, "ORTA": 2, "DUSUK": 3}
        new_alerts.sort(key=lambda a: priority.get(a.severity, 9))

        self.alerts.extend(new_alerts)
        return new_alerts


# ─────────────────────────────────────────────────────────────────────────────
# 6. GÖRSEL ÇIKTI MODÜLİ
# ─────────────────────────────────────────────────────────────────────────────

class Visualizer:
    """Takip sonuçlarını ve KDS uyarılarını video karesi üzerine çizer."""

    COLORS = {
        "KRITIK": (0,   0, 255),   # Kırmızı
        "YUKSEK": (0, 128, 255),   # Turuncu
        "ORTA":   (0, 255, 255),   # Sarı
        "DUSUK":  (0, 255,   0),   # Yeşil
        "DEFAULT":(255, 255, 255), # Beyaz
    }

    def draw_tracks(self, frame: np.ndarray,
                    vehicles: List[TrackedVehicle],
                    alerts: List[TrafficAlert]) -> np.ndarray:
        alert_ids = {a.vehicle_id for a in alerts}

        for v in vehicles:
            color = self.COLORS["KRITIK"] if v.track_id in alert_ids else self.COLORS["DEFAULT"]
            x1,y1,x2,y2 = int(v.bbox.x1),int(v.bbox.y1),int(v.bbox.x2),int(v.bbox.y2)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"ID:{v.track_id}  {v.speed_kmh:.0f}km/h  S{v.lane_id}"
            cv2.putText(frame, label, (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # Takip izi
            for i in range(1, len(v.position_history)):
                p1 = tuple(map(int, v.position_history[i-1]))
                p2 = tuple(map(int, v.position_history[i]))
                cv2.line(frame, p1, p2, color, 1)

        return frame

    def draw_alerts(self, frame: np.ndarray,
                    alerts: List[TrafficAlert]) -> np.ndarray:
        for i, alert in enumerate(alerts[:5]):
            color = self.COLORS.get(alert.severity, self.COLORS["DEFAULT"])
            text  = f"[{alert.rule_id}] {alert.description[:60]}"
            y_pos = 25 + i * 22
            cv2.putText(frame, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# 7. ANA UYGULAMA
# ─────────────────────────────────────────────────────────────────────────────

class TrafficAnalysisSystem:
    """
    Uçtan uca trafik analiz sistemi.

    İşlem hattı:
        Video Girişi → Ön İşleme → GMM Segm. → Kontur →
        SORT Takip → Özellik Çıkarımı → Kural Motoru → Çıktı
    """

    def __init__(self, video_path: str, output_path: str = "output.avi"):
        self.video_path  = video_path
        self.output_path = output_path

        self.segmenter = VideoSegmenter()
        self.tracker   = SORTTracker(max_age=3, min_hits=3, iou_threshold=0.3)
        self.fe        = FeatureExtractor(fps=30.0)
        self.dss       = RuleBasedDSS(self.fe)
        self.viz       = Visualizer()

        self.all_alerts: List[TrafficAlert] = []
        self.frame_times: List[float] = []

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"HATA: Video acilamadi: {self.video_path}")
            return

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fe.fps = fps

        out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            fps, (1280, 720)
        )

        frame_id = 0
        active_vehicles: Dict[int, TrackedVehicle] = {}

        print(f"Video: {self.video_path}  ({W}x{H} @ {fps:.0f} FPS)")
        print("İşlem başlıyor...\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()

            # ── Segmentasyon ──────────────────────────────────────────────
            detections, proc_frame, mask = self.segmenter.process_frame(frame)

            # ── Takip ─────────────────────────────────────────────────────
            tracks = self.tracker.update(detections)

            # ── Özellik çıkarımı ──────────────────────────────────────────
            current_vehicles: List[TrackedVehicle] = []
            for (tid, bbox) in tracks:
                self.fe.update_position(tid, bbox.cx, bbox.cy, frame_id)
                spd  = self.fe.compute_speed(tid)
                lane = self.fe.get_lane(bbox.cx)

                if tid not in active_vehicles:
                    active_vehicles[tid] = TrackedVehicle(
                        track_id=tid, bbox=bbox, speed_kmh=spd, lane_id=lane
                    )
                v = active_vehicles[tid]
                v.bbox = bbox
                v.speed_kmh = spd
                v.lane_id = lane
                v.frames_tracked += 1
                v.update_history()
                current_vehicles.append(v)

            # ── Kural değerlendirme ───────────────────────────────────────
            density = self.fe.compute_density(
                current_vehicles, (0, 0, 1280, 720)
            )
            alerts = self.dss.evaluate(frame_id, current_vehicles, density)
            self.all_alerts.extend(alerts)

            # ── Görselleştirme ────────────────────────────────────────────
            vis = proc_frame.copy()
            # إظهار حدود منطقة الطريق باللون الأحمر الخفيف حتى تعرف أين يقرأ النظام.
            cv2.polylines(vis, [np.array(self.segmenter.road_roi_polygon, dtype=np.int32)],
                          isClosed=True, color=(0, 0, 255), thickness=2)
            vis = self.viz.draw_tracks(vis, current_vehicles, alerts)
            vis = self.viz.draw_alerts(vis, alerts)
            out.write(vis)

            elapsed = time.time() - t0
            self.frame_times.append(elapsed * 1000)

            if frame_id % 100 == 0:
                avg_ms = np.mean(self.frame_times[-100:])
                print(f"  Kare {frame_id:5d} | "
                      f"Tespit: {len(detections):3d} | "
                      f"İz: {len(tracks):3d} | "
                      f"Uyarı: {len(alerts):3d} | "
                      f"Ort. {avg_ms:.1f} ms/kare")

            frame_id += 1

        cap.release()
        out.release()
        self._save_alerts_csv()
        self._print_summary(frame_id)

    def _save_alerts_csv(self):
        """Tüm uyarıları CSV dosyasına kaydet."""
        csv_path = self.output_path.replace(".avi", "_alerts.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame_id", "vehicle_id", "rule_id", "description", "severity"
            ])
            writer.writeheader()
            for a in self.all_alerts:
                writer.writerow({
                    "frame_id":    a.frame_id,
                    "vehicle_id":  a.vehicle_id,
                    "rule_id":     a.rule_id,
                    "description": a.description,
                    "severity":    a.severity,
                })
        print(f"\nUyarılar kaydedildi: {csv_path}")

    def _print_summary(self, total_frames: int):
        print("\n" + "="*60)
        print("PERFORMANS ÖZETİ")
        print("="*60)
        print(f"Toplam kare     : {total_frames}")
        print(f"Toplam uyarı    : {len(self.all_alerts)}")
        if self.frame_times:
            print(f"Ort. gecikme    : {np.mean(self.frame_times):.1f} ms/kare")
            print(f"Maks. gecikme   : {np.max(self.frame_times):.1f} ms/kare")
            print(f"İşlem hızı      : {1000/np.mean(self.frame_times):.1f} FPS")
        rule_counts: Dict[str, int] = {}
        for a in self.all_alerts:
            rule_counts[a.rule_id] = rule_counts.get(a.rule_id, 0) + 1
        print("\nKural Bazlı Uyarı Sayıları:")
        for rule, count in sorted(rule_counts.items()):
            print(f"  {rule}: {count}")
        print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# 8. GİRİŞ NOKTASI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Kural Tabanlı KDS ile Video Trafik Araç Takibi - Cars Only"
    )
    # يقبل الطريقتين:
    # python traffic_analysis_cars_only.py --video video.mp4
    # python traffic_analysis_cars_only.py video.mp4
    parser.add_argument("video_pos", nargs="?", default=None,
                        help="اختياري: مسار الفيديو بدون --video")
    parser.add_argument("--video", type=str, default=None,
                        help="Giriş video dosyası yolu")
    parser.add_argument("--output", type=str, default="output_speed_ready_v3.avi",
                        help="Çıkış video dosyası yolu")
    args = parser.parse_args()

    video_arg = args.video or args.video_pos or "video.mp4"
    video_path = Path(video_arg).expanduser()

    # لو كتبت اسم الملف فقط، ابحث عنه في المجلد الحالي ومجلد الكود.
    candidates = [
        video_path,
        Path.cwd() / video_path,
        Path(__file__).resolve().parent / video_path,
    ]

    found_path = None
    for c in candidates:
        if c.exists() and c.is_file():
            found_path = str(c.resolve())
            break

    if found_path is None:
        print("خطأ: الفيديو غير موجود، ولن أفتح webcam حتى لا يعطيك نتائج خاطئة.")
        print(f"الاسم الذي طلبته: {video_arg}")
        print("جرّب أحد هذه الأوامر:")
        print(r'  python "Kullanılan kodlar\traffic_analysis_cars_only.py" --video "C:\Users\ل\Desktop\video.mp4"')
        print(r'  python "Kullanılan kodlar\traffic_analysis_cars_only.py" --video "Kullanılan kodlar\video.mp4"')
        sys.exit(1)

    print(f"تم العثور على الفيديو: {found_path}")
    system = TrafficAnalysisSystem(
        video_path=found_path,
        output_path=args.output
    )
    system.run()
