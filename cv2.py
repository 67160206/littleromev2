"""
cv2.py — stub สำหรับ Streamlit Cloud (Python 3.14)
ทำให้ ultralytics import ได้โดยไม่ crash
"""
import numpy as np
import io
from PIL import Image as _Image

# ── Constants ──────────────────────────────
INTER_NEAREST       = 0
INTER_LINEAR        = 1
INTER_CUBIC         = 2
INTER_AREA          = 3
INTER_LANCZOS4      = 4
BORDER_CONSTANT     = 0
BORDER_REFLECT      = 2
BORDER_REFLECT_101  = 4
BORDER_REPLICATE    = 1
IMREAD_COLOR        = 1
IMREAD_GRAYSCALE    = 0
IMREAD_UNCHANGED    = -1
COLOR_BGR2RGB       = 4
COLOR_RGB2BGR       = 3
COLOR_BGR2GRAY      = 6
COLOR_GRAY2BGR      = 8
COLOR_BGR2HSV       = 40
COLOR_HSV2BGR       = 54
COLOR_BGR2LAB       = 44
COLOR_LAB2BGR       = 56
COLOR_BGRA2BGR      = 1
COLOR_BGR2BGRA      = 0
FONT_HERSHEY_SIMPLEX  = 0
FONT_HERSHEY_DUPLEX   = 1
FONT_HERSHEY_COMPLEX  = 2
LINE_AA             = 16
LINE_8              = 8
FILLED              = -1
CAP_PROP_FPS        = 5
CAP_PROP_FRAME_COUNT= 7
CAP_PROP_FRAME_WIDTH= 3
CAP_PROP_FRAME_HEIGHT=4
CAP_PROP_POS_FRAMES = 1
CAP_PROP_FOURCC     = 6
CHAIN_APPROX_SIMPLE = 2
RETR_EXTERNAL       = 0
RETR_LIST           = 1
NORM_L2             = 4
TM_CCOEFF_NORMED    = 5
ROTATE_90_CLOCKWISE = 0

# ── Core functions ─────────────────────────
def resize(img, dsize, fx=0, fy=0, interpolation=INTER_LINEAR):
    if img is None: return img
    pil = _Image.fromarray(img.astype(np.uint8))
    w, h = (int(dsize[0]), int(dsize[1])) if (dsize[0] and dsize[1]) else \
           (int(img.shape[1]*fx), int(img.shape[0]*fy))
    return np.array(pil.resize((w, h), _Image.BILINEAR))

def cvtColor(img, code):
    if img is None: return img
    if code in (COLOR_BGR2RGB, COLOR_RGB2BGR):
        return img[:, :, ::-1].copy()
    if code == COLOR_BGR2GRAY:
        return np.dot(img[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
    if code == COLOR_GRAY2BGR:
        return np.stack([img]*3, axis=-1)
    if code == COLOR_BGRA2BGR:
        return img[:, :, :3]
    if code == COLOR_BGR2BGRA:
        a = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
        return np.concatenate([img, a], axis=-1)
    return img

def imencode(ext, img, params=None):
    pil = _Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    fmt = 'JPEG' if any(x in ext for x in ['jpg','jpeg']) else 'PNG'
    pil.save(buf, format=fmt)
    return True, np.frombuffer(buf.getvalue(), dtype=np.uint8)

def imdecode(buf, flags=IMREAD_COLOR):
    try:
        arr = buf if isinstance(buf, (bytes, bytearray)) else bytes(buf)
        img = _Image.open(io.BytesIO(arr)).convert('RGB')
        out = np.array(img)
        return out[:, :, ::-1] if flags == IMREAD_COLOR else out
    except Exception:
        return None

def imread(path, flags=IMREAD_COLOR):
    try:
        img = _Image.open(path).convert('RGB')
        out = np.array(img)
        return out[:, :, ::-1] if flags == IMREAD_COLOR else out
    except Exception:
        return None

def imwrite(path, img):
    try:
        _Image.fromarray(img[:,:,::-1]).save(path)
        return True
    except Exception:
        return False

def flip(img, flipCode):
    if flipCode == 0:  return img[::-1]
    if flipCode == 1:  return img[:, ::-1]
    return img[::-1, ::-1]

def rotate(img, rotateCode):
    k = rotateCode + 1
    return np.rot90(img, k=-k)

def copyMakeBorder(img, top, bottom, left, right, borderType=BORDER_CONSTANT, value=0):
    return np.pad(img, ((top,bottom),(left,right),(0,0)), mode='constant', constant_values=value)

def split(img):
    return [img[:,:,i] for i in range(img.shape[2])]

def merge(channels):
    return np.stack(channels, axis=-1)

def normalize(src, dst=None, alpha=0, beta=1, norm_type=NORM_L2, dtype=-1, mask=None):
    mn, mx = src.min(), src.max()
    if mx == mn: return src.astype(np.float32)
    return ((src - mn) / (mx - mn) * (beta - alpha) + alpha).astype(np.float32)

def GaussianBlur(img, ksize, sigmaX, sigmaY=0, borderType=BORDER_REFLECT_101):
    return img  # passthrough stub

def medianBlur(img, ksize):
    return img

def Canny(img, threshold1, threshold2, edges=None, apertureSize=3, L2gradient=False):
    return np.zeros(img.shape[:2], dtype=np.uint8)

def dilate(img, kernel, dst=None, anchor=(-1,-1), iterations=1, borderType=BORDER_CONSTANT, borderValue=0):
    return img

def erode(img, kernel, dst=None, anchor=(-1,-1), iterations=1, borderType=BORDER_CONSTANT, borderValue=0):
    return img

def threshold(img, thresh, maxval, type):
    out = (img > thresh).astype(np.uint8) * int(maxval)
    return int(thresh), out

def getStructuringElement(shape, ksize, anchor=(-1,-1)):
    return np.ones(ksize, dtype=np.uint8)

def findContours(img, mode, method):
    return [], None

def contourArea(contour): return 0.0
def arcLength(contour, closed): return 0.0
def approxPolyDP(contour, epsilon, closed): return contour
def boundingRect(contour): return (0,0,0,0)
def convexHull(points, hull=None, clockwise=False, returnPoints=True): return points
def drawContours(img, contours, contourIdx, color, thickness=1, lineType=LINE_8): return img

def rectangle(img, pt1, pt2, color, thickness=1, lineType=LINE_8, shift=0):
    return img

def circle(img, center, radius, color, thickness=1, lineType=LINE_8, shift=0):
    return img

def line(img, pt1, pt2, color, thickness=1, lineType=LINE_8, shift=0):
    return img

def polylines(img, pts, isClosed, color, thickness=1, lineType=LINE_8, shift=0):
    return img

def putText(img, text, org, fontFace, fontScale, color, thickness=1, lineType=LINE_8, bottomLeftOrigin=False):
    return img

def getTextSize(text, fontFace, fontScale, thickness):
    w = int(len(text) * fontScale * 12)
    h = int(fontScale * 20)
    return (w, h), int(fontScale * 4)

def warpAffine(img, M, dsize, flags=INTER_LINEAR, borderMode=BORDER_CONSTANT, borderValue=0):
    return resize(img, dsize)

def warpPerspective(img, M, dsize, flags=INTER_LINEAR, borderMode=BORDER_CONSTANT, borderValue=0):
    return resize(img, dsize)

def getRotationMatrix2D(center, angle, scale):
    return np.eye(2, 3, dtype=np.float64)

def getPerspectiveTransform(src, dst, solveMethod=0):
    return np.eye(3, dtype=np.float64)

def minMaxLoc(img, mask=None):
    return float(img.min()), float(img.max()), (0,0), (0,0)

def matchTemplate(img, templ, method):
    return np.zeros((1,1), dtype=np.float32)

def dnn_NMSBoxes(bboxes, scores, score_threshold, nms_threshold, eta=1.0, top_k=0):
    return []

# ── DNN module stub ────────────────────────
class _DNN:
    def readNet(self, *a, **kw): return self
    def readNetFromONNX(self, *a, **kw): return self
    def setInput(self, *a, **kw): pass
    def forward(self, *a, **kw): return np.zeros((1,1,1,1))
    def setPreferableBackend(self, *a): pass
    def setPreferableTarget(self, *a): pass
    def NMSBoxes(self, *a, **kw): return []

dnn = _DNN()

# ── VideoCapture stub ──────────────────────
class VideoCapture:
    def __init__(self, source=0): self._open = False
    def isOpened(self): return self._open
    def read(self): return False, None
    def get(self, prop): return 0.0
    def set(self, prop, val): return False
    def release(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): self.release()

class VideoWriter:
    def __init__(self, *a, **kw): pass
    def write(self, frame): pass
    def release(self): pass
    def isOpened(self): return False

def VideoWriter_fourcc(*args): return 0

# ── Mat / misc ─────────────────────────────
def getOptimalDFTSize(n): return n
def dft(src, dst=None, flags=0): return src
def idft(src, dst=None, flags=0, nonzeroRows=0): return src
def magnitude(x, y): return np.sqrt(x**2 + y**2)
def log(src, dst=None): return np.log(src + 1e-10)
def cartToPolar(x, y, magnitude=None, angle=None, angleInDegrees=False): return np.hypot(x,y), np.arctan2(y,x)
def add(src1, src2, dst=None, mask=None, dtype=-1): return np.add(src1, src2)
def subtract(src1, src2, dst=None, mask=None, dtype=-1): return np.subtract(src1, src2)
def multiply(src1, src2, dst=None, scale=1, dtype=-1): return np.multiply(src1, src2)
def divide(src1, src2, dst=None, scale=1, dtype=-1): return np.divide(src1, src2)
def absdiff(src1, src2, dst=None): return np.abs(src1.astype(float) - src2.astype(float)).astype(src1.dtype)
def bitwise_and(src1, src2, dst=None, mask=None): return np.bitwise_and(src1, src2)
def bitwise_or(src1, src2, dst=None, mask=None): return np.bitwise_or(src1, src2)
def bitwise_not(src, dst=None, mask=None): return np.bitwise_not(src)

__version__ = "4.9.0"
