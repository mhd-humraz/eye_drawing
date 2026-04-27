 

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import time
import math
from collections import deque

 #  Page config — MUST be first
 st.set_page_config(
    page_title="Eye Canvas Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

 #   CSS — Cyberpunk / Neural aesthetic
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0f172a;
    color: #e5e7eb;
}

.block-container {
    padding: 1.2rem 1.5rem !important;
    max-width: 1100px !important;
}

/* Hide Streamlit UI */
#MainMenu, footer, header { visibility: hidden; }

/* Title */
.eye-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #f9fafb;
    margin-bottom: 4px;
}

.eye-sub {
    font-size: 0.8rem;
    color: #9ca3af;
    margin-bottom: 12px;
}

/* Cards */
.stat-row {
    display: flex;
    gap: 10px;
    margin-bottom: 1rem;
}

.stat-card {
    flex: 1;
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 10px;
    text-align: center;
}

.stat-val {
    font-size: 1.2rem;
    font-weight: 600;
    color: #ffffff;
}

.stat-lbl {
    font-size: 0.7rem;
    color: #9ca3af;
}

/* Status */
.status-badge {
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.75rem;
    margin-bottom: 10px;
    display: inline-block;
    border: 1px solid #1f2937;
}

.status-drawing { color: #22c55e; }
.status-paused  { color: #f59e0b; }
.status-cal     { color: #3b82f6; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1f2937;
}

/* Buttons */
.stButton > button {
    background: #111827;
    color: #e5e7eb;
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 6px;
    font-size: 0.8rem;
}

.stButton > button:hover {
    background: #1f2937;
}

/* Sliders */
.stSlider label {
    font-size: 0.75rem;
    color: #9ca3af;
}

/* Info box */
.info-box {
    background: #020617;
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 10px;
    font-size: 0.75rem;
    color: #9ca3af;
    line-height: 1.6;
}

/* Image frame */
[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid #1f2937;
}
</style> 
""", unsafe_allow_html=True)

 #  Constants
 LEFT_IRIS   = [474, 475, 476, 477]
RIGHT_IRIS  = [469, 470, 471, 472]

# Refined eye contour landmarks for better EAR
L_EYE_TOP  = [159, 160, 161]
L_EYE_BOT  = [145, 144, 153]
L_EYE_L    = 33
L_EYE_R    = 133
R_EYE_TOP  = [386, 387, 388]
R_EYE_BOT  = [374, 373, 380]
R_EYE_L    = 362
R_EYE_R    = 263

# Cheek landmarks — used to compute head-pose-aware iris normalisation
NOSE_TIP   = 1
LEFT_CHEEK = 234
RIGHT_CHEEK= 454

BLINK_FRAMES   = 3
BLINK_COOLDOWN = 0.5
CAL_COLLECT    = 55

CAL_TARGETS = [
    (0.10,0.10),(0.50,0.10),(0.90,0.10),
    (0.10,0.50),(0.50,0.50),(0.90,0.50),
    (0.10,0.90),(0.50,0.90),(0.90,0.90),
]
CAL_LABELS = [
    "TOP LEFT","TOP CENTRE","TOP RIGHT",
    "MID LEFT","CENTRE","MID RIGHT",
    "BOT LEFT","BOT CENTRE","BOT RIGHT",
]

# Palette: (BGR, hex_for_ui, label)
PALETTE_DEF = [
    ((0,255,255),  "#ffff00", "YELLOW", "1"),
    ((255,80,80),  "#5050ff", "BLUE",   "2"),
    ((80,255,80),  "#50ff50", "GREEN",  "3"),
    ((80,80,255),  "#ff5050", "RED",    "4"),
    ((255,80,255), "#ff50ff", "PINK",   "5"),
    ((80,200,255), "#ffc850", "ORANGE", "6"),
    ((255,255,255),"#ffffff", "WHITE",  "7"),
]

 #  Session state
 def init():
    D = dict(
        canvas=None, prev_x=0, prev_y=0,
        drawing=False,
        color=(0,255,255), color_name="YELLOW",
        mode="DRAW",
        brush=5, eraser=45,
        calibrated=False, cal_step=0,
        cal_iris=[], cal_screen=[], cal_buf=[],
        cal_cx=None, cal_cy=None,
        smooth_x=deque(maxlen=10),
        smooth_y=deque(maxlen=10),
        kalman=None,
        blink_counter=0, last_blink=0.0,
        blink_triggered=False,
        blink_count=0,
        ear_val=0.0,
        ear_thresh=0.21,
        fps_buf=deque(maxlen=30),
        last_frame_t=time.time(),
        cursor_x=0, cursor_y=0,
        eye_detected=False,
    )
    for k,v in D.items():
        if k not in st.session_state:
            st.session_state[k]=v

init()
ss = st.session_state

 #  Kalman
 def make_kalman():
    kf = cv2.KalmanFilter(4,2)
    kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],
                                        [0,0,1,0],[0,0,0,1]],np.float32)
    kf.processNoiseCov     = np.eye(4,dtype=np.float32)*0.008
    kf.measurementNoiseCov = np.eye(2,dtype=np.float32)*0.08
    kf.errorCovPost        = np.eye(4,dtype=np.float32)
    return kf

if ss.kalman is None:
    ss.kalman = make_kalman()

 #  MediaPipe
 @st.cache_resource
def load_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.65,
    )
 
#  EAR  (multi-point, more robust)
 def ear(lms, top_ids, bot_ids, l_id, r_id, W, H):
    def pt(i): return np.array([lms[i].x*W, lms[i].y*H])
    top  = np.mean([pt(i) for i in top_ids], axis=0)
    bot  = np.mean([pt(i) for i in bot_ids], axis=0)
    vert = np.linalg.norm(top-bot)
    horz = np.linalg.norm(pt(l_id)-pt(r_id))
    return vert/(horz+1e-6)
 
#  Head-pose-normalised iris position
#  Divides iris displacement by face width → robust to head movement
 def get_iris_norm(lms, W, H):
    def lm(i): return np.array([lms[i].x*W, lms[i].y*H])

    # Face reference points
    nose      = lm(NOSE_TIP)
    lc        = lm(LEFT_CHEEK)
    rc        = lm(RIGHT_CHEEK)
    face_w    = np.linalg.norm(rc-lc) + 1e-6
    face_cx   = (lc[0]+rc[0])/2
    face_cy   = (lc[1]+rc[1])/2

    # Iris centres
    def iris_c(ids):
        xs = [lms[i].x*W for i in ids]
        ys = [lms[i].y*H for i in ids]
        return np.array([np.mean(xs), np.mean(ys)])

    left_c  = iris_c(LEFT_IRIS)
    right_c = iris_c(RIGHT_IRIS)
    avg_c   = (left_c+right_c)/2

    # Normalise by face width → head-pose invariant
    nx = (avg_c[0]-face_cx)/face_w + 0.5
    ny = (avg_c[1]-face_cy)/face_w + 0.5
    return float(nx), float(ny), left_c, right_c
 
#  Polynomial calibration
                                                                   
def poly_row(ix,iy):
    return [1, ix, iy, ix*ix, iy*iy, ix*iy]

def fit_cal(iris_pts, screen_pts):
    A  = np.array([poly_row(x,y) for x,y in iris_pts], np.float64)
    Sx = np.array([p[0] for p in screen_pts], np.float64)
    Sy = np.array([p[1] for p in screen_pts], np.float64)
    cx,*_ = np.linalg.lstsq(A, Sx, rcond=None)
    cy,*_ = np.linalg.lstsq(A, Sy, rcond=None)
    return cx, cy

def apply_cal(ix,iy,cx,cy,W,H):
    row = np.array(poly_row(ix,iy), np.float64)
    sx  = int(np.clip(row@cx, 0, W-1))
    sy  = int(np.clip(row@cy, 0, H-1))
    return sx, sy

def kf_update(kf,x,y):
    kf.predict()
    est = kf.correct(np.array([[np.float32(x)],[np.float32(y)]]))
    return int(est[0]), int(est[1])

  
#  OpenCV frame overlays  (pro dark-theme HUD) 
FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN

def draw_scan_lines(frame):
    """Subtle CRT scanline effect on dark areas"""
    H,W = frame.shape[:2]
    for y in range(0,H,4):
        cv2.line(frame,(0,y),(W,y),(0,0,0),1)
    # blend lightly
    return frame

def draw_corner_brackets(frame, x1,y1,x2,y2, color, size=18, t=2):
    """Draw corner brackets around a region"""
    pts = [
        [(x1,y1),(x1+size,y1)],[(x1,y1),(x1,y1+size)],
        [(x2,y1),(x2-size,y1)],[(x2,y1),(x2,y1+size)],
        [(x1,y2),(x1+size,y2)],[(x1,y2),(x1,y2-size)],
        [(x2,y2),(x2-size,y2)],[(x2,y2),(x2,y2-size)],
    ]
    for p1,p2 in pts:
        cv2.line(frame,p1,p2,color,t)

def draw_cal_overlay(frame, step, n_col):
    H,W = frame.shape[:2]
    # dark overlay
    ov = np.zeros_like(frame)
    frame = cv2.addWeighted(frame,0.18,ov,0.82,0)

    tx = int(CAL_TARGETS[step][0]*W)
    ty = int(CAL_TARGETS[step][1]*H)
    t  = n_col/CAL_COLLECT

    # animated concentric rings
    for r,alpha in [(55,0.3),(42,0.5),(30,0.8)]:
        ov2 = frame.copy()
        cv2.circle(ov2,(tx,ty),r,(0,255,160),-1)
        frame = cv2.addWeighted(frame,1,ov2,alpha*0.08,0)

    # shrinking fill
    r_fill = max(4, int(28*(1-t*0.85)))
    cv2.circle(frame,(tx,ty),r_fill,(0,255,160),-1)
    cv2.circle(frame,(tx,ty),38,(0,255,160),1)
    cv2.circle(frame,(tx,ty),55,(0,180,100),1)
    # crosshair
    cv2.line(frame,(tx-65,ty),(tx-42,ty),(0,255,160),1)
    cv2.line(frame,(tx+42,ty),(tx+65,ty),(0,255,160),1)
    cv2.line(frame,(tx,ty-65),(tx,ty-42),(0,255,160),1)
    cv2.line(frame,(tx,ty+42),(tx,ty+65),(0,255,160),1)
    cv2.circle(frame,(tx,ty),3,(255,255,255),-1)

    # completed dots
    for i in range(step):
        px=int(CAL_TARGETS[i][0]*W)
        py=int(CAL_TARGETS[i][1]*H)
        cv2.circle(frame,(px,py),7,(0,200,80),-1)
        cv2.circle(frame,(px,py),10,(0,200,80),1)

    # panel
    pw,ph = 380,100
    px2 = (W-pw)//2
    py2 = H-ph-10
    cv2.rectangle(frame,(px2,py2),(px2+pw,py2+ph),(8,15,28),-1)
    cv2.rectangle(frame,(px2,py2),(px2+pw,py2+ph),(0,60,40),1)
    draw_corner_brackets(frame,px2,py2,px2+pw,py2+ph,(0,255,160),10,1)

    cv2.putText(frame,"CALIBRATION",(px2+10,py2+22),FONT,0.55,(0,255,160),1)
    cv2.putText(frame,f"POINT {step+1}/{len(CAL_TARGETS)} — {CAL_LABELS[step]}",
                (px2+10,py2+44),FONT,0.52,(180,220,200),1)
    cv2.putText(frame,"Fix your gaze on the dot and hold still",
                (px2+10,py2+64),FONT_MONO,1,(80,120,100),1)

    # progress bar
    bx,by = px2+10, py2+80
    bw2   = pw-20
    cv2.rectangle(frame,(bx,by),(bx+bw2,by+8),(20,35,30),-1)
    cv2.rectangle(frame,(bx,by),(bx+int(bw2*t),by+8),(0,255,160),-1)
    cv2.rectangle(frame,(bx,by),(bx+bw2,by+8),(0,60,40),1)

    return frame

def draw_hud(frame, ss, cursor=None, fps=0):
    H,W = frame.shape[:2]

    # ── Top bar ─────────────────────────────────────────
    cv2.rectangle(frame,(0,0),(W,50),(5,8,18),-1)
    cv2.line(frame,(0,50),(W,50),(0,50,35),1)

    # Title
    cv2.putText(frame,"EYE CANVAS PRO",(12,32),FONT,0.7,(0,200,130),1)

    # FPS
    fps_txt = f"FPS {fps:4.1f}"
    cv2.putText(frame,fps_txt,(W-110,32),FONT,0.5,(40,80,60),1)

    # Mode badge
    mode_col = (0,255,100) if ss.drawing else (0,160,255)
    mode_txt = "● DRAW" if ss.drawing else "○ GAZE"
    cv2.putText(frame,mode_txt,(W//2-40,32),FONT,0.55,mode_col,1)

    # ── Color swatch + mode ──────────────────────────────
    sw_x, sw_y = W-60, 58
    cv2.rectangle(frame,(sw_x,sw_y),(sw_x+44,sw_y+30),ss.color,-1)
    cv2.rectangle(frame,(sw_x,sw_y),(sw_x+44,sw_y+30),(0,100,70),1)
    draw_corner_brackets(frame,sw_x,sw_y,sw_x+44,sw_y+30,(0,200,120),5,1)
    cv2.putText(frame,ss.color_name[:3],(sw_x+4,sw_y+20),FONT,0.4,
                (0,0,0) if ss.color!=(255,255,255) else (100,100,100),1)

    # ── Eye detected indicator ───────────────────────────
    det_col = (0,220,100) if ss.eye_detected else (0,40,220)
    cv2.circle(frame,(W-80,H-20),5,det_col,-1)
    cv2.putText(frame,"EYE" if ss.eye_detected else "NO EYE",
                (W-70,H-15),FONT,0.35,det_col,1)

    # ── Bottom info strip ────────────────────────────────
    cv2.rectangle(frame,(0,H-36),(W,H),(5,8,18),-1)
    cv2.line(frame,(0,H-36),(W,H-36),(0,40,30),1)
    info = (f"BLINKS:{ss.blink_count}   "
            f"MODE:{ss.mode}   "
            f"BRUSH:{ss.brush}px   "
            f"EAR:{ss.ear_val:.3f}   "
            f"KEYS: 1-7=COLOR  E=ERASE  C=CLEAR  SPACE=RECAL")
    cv2.putText(frame,info,(10,H-12),FONT_MONO,1,(40,80,60),1)

    # ── Cursor ───────────────────────────────────────────
    if cursor:
        cx2,cy2 = cursor
        c_col   = ss.color if ss.mode=="DRAW" else (60,60,60)

        # outer ring
        cv2.circle(frame,(cx2,cy2),ss.brush+14,c_col,1)
        # inner ring
        cv2.circle(frame,(cx2,cy2),ss.brush+7,c_col,2)
        # centre dot
        cv2.circle(frame,(cx2,cy2),3,(255,255,255),-1)

        # active drawing pulse
        if ss.drawing:
            cv2.circle(frame,(cx2,cy2),ss.brush+22,(0,255,100),1)
            # small cross
            cv2.line(frame,(cx2-5,cy2),(cx2+5,cy2),(0,255,100),1)
            cv2.line(frame,(cx2,cy2-5),(cx2,cy2+5),(0,255,100),1)

        # blink flash
        if ss.blink_triggered:
            cv2.circle(frame,(cx2,cy2),ss.brush+32,(0,255,100),2)

    # ── EAR bar (eye openness) ───────────────────────────
    ear_w  = 120
    ear_x  = 10
    ear_y  = H-36-20
    ear_pct= min(1.0, ss.ear_val/0.4)
    cv2.rectangle(frame,(ear_x,ear_y),(ear_x+ear_w,ear_y+10),(15,25,20),-1)
    col_ear= (0,200,80) if ss.ear_val>ss.ear_thresh else (0,80,220)
    cv2.rectangle(frame,(ear_x,ear_y),(ear_x+int(ear_w*ear_pct),ear_y+10),col_ear,-1)
    cv2.rectangle(frame,(ear_x,ear_y),(ear_x+ear_w,ear_y+10),(0,60,40),1)
    th_x = ear_x+int(ear_w*min(1,ss.ear_thresh/0.4))
    cv2.line(frame,(th_x,ear_y-2),(th_x,ear_y+12),(0,255,160),1)
    cv2.putText(frame,"EAR",(ear_x,ear_y-4),FONT,0.3,(0,150,80),1)

    return frame
 
#  Sidebar UI
 

with st.sidebar:
    st.markdown('<p class="eye-title" style="font-size:1.1rem">EYE CANVAS</p>', unsafe_allow_html=True)
    st.markdown('<p class="eye-sub">NEURAL GAZE INTERFACE v2.0</p>', unsafe_allow_html=True)

    run = st.checkbox("▶  ACTIVATE CAMERA", value=False)

    st.markdown('<p class="sidebar-section">BRUSH</p>', unsafe_allow_html=True)
    ss.brush  = st.slider("Brush Size",  1, 25, ss.brush,  label_visibility="visible")
    ss.eraser = st.slider("Eraser Size", 10, 80, ss.eraser, label_visibility="visible")

    st.markdown('<p class="sidebar-section">BLINK DETECTION</p>', unsafe_allow_html=True)
    ss.ear_thresh = st.slider("EAR Threshold", 0.10, 0.30,
                              float(ss.ear_thresh), 0.005,
                              help="Lower = more sensitive to blinks")
    smooth_sz = st.slider("Gaze Smoothing", 3, 16, 10,
                          help="Higher = smoother, more lag")

    st.markdown('<p class="sidebar-section">ACTIONS</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 CLEAR"):
            ss.canvas = None
    with col2:
        if st.button("🔄 RECAL"):
            ss.calibrated=False; ss.cal_step=0
            ss.cal_iris=[]; ss.cal_screen=[]; ss.cal_buf=[]
            ss.cal_cx=None; ss.cal_cy=None
            ss.kalman=make_kalman()
            ss.smooth_x.clear(); ss.smooth_y.clear()

    st.markdown('<p class="sidebar-section">KEYBOARD</p>', unsafe_allow_html=True)
    st.markdown("""
<div class="key-grid">
  <div class="key-pill"><span class="key-code">1-7</span> colors</div>
  <div class="key-pill"><span class="key-code">E</span> eraser</div>
  <div class="key-pill"><span class="key-code">C</span> clear</div>
  <div class="key-pill"><span class="key-code">SPC</span> recal</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section">HOW TO USE</p>', unsafe_allow_html=True)
    st.markdown("""
<div class="info-box">
① Sit 60-70cm from screen<br>
② Good front lighting<br>
③ Complete 9-point calibration<br>
④ Gaze moves the cursor<br>
⑤ <b style="color:#00ffaa">Blink</b> once → start drawing<br>
⑥ <b style="color:#00ffaa">Blink</b> again → stop drawing<br>
⑦ Keys 1-7 change color
</div>
""", unsafe_allow_html=True)
 
#  Main layout


st.markdown('<p class="eye-title">👁 EYE CANVAS PRO</p>', unsafe_allow_html=True)
st.markdown('<p class="eye-sub">NEURAL IRIS TRACKING · BLINK TO DRAW · 9-POINT CALIBRATION</p>',
            unsafe_allow_html=True)

# Stat cards (updated each frame via placeholders)
stat_ph = st.empty()
status_ph = st.empty()
FRAME_WINDOW = st.image([])

def render_stats(fps, blinks, ear_v, detected):
    ear_pct  = min(100, int(ear_v/0.4*100))
    det_icon = "✅" if detected else "❌"
    draw_icon= "🟢" if ss.drawing else "🟡"
    stat_ph.markdown(f"""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-val">{fps:.0f}</div>
    <div class="stat-lbl">FPS</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{blinks}</div>
    <div class="stat-lbl">BLINKS</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{ear_pct}%</div>
    <div class="stat-lbl">EYE OPEN</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{det_icon}</div>
    <div class="stat-lbl">EYE DETECT</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{draw_icon}</div>
    <div class="stat-lbl">DRAWING</div>
  </div>
</div>
""", unsafe_allow_html=True)

def render_status():
    if not ss.calibrated:
        status_ph.markdown(
            '<div class="status-badge status-cal">'
            '<div class="pulse-dot dot-blue"></div>'
            'CALIBRATING — STARE AT EACH DOT</div>', unsafe_allow_html=True)
    elif ss.drawing:
        status_ph.markdown(
            '<div class="status-badge status-drawing">'
            '<div class="pulse-dot dot-green"></div>'
            f'DRAWING · {ss.color_name} · {ss.brush}px</div>', unsafe_allow_html=True)
    else:
        status_ph.markdown(
            '<div class="status-badge status-paused">'
            '<div class="pulse-dot dot-orange"></div>'
            'GAZE MODE — BLINK TO START DRAWING</div>', unsafe_allow_html=True)

if not run:
    render_stats(0, 0, 0.25, False)
    render_status()
    st.info("✅ Enable **▶ ACTIVATE CAMERA** in the sidebar to begin.")
    st.stop()


#  Camera + main loop
face_mesh = load_face_mesh()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

if not cap.isOpened():
    st.error("❌ Camera not found.")
    st.stop()

# Update smooth buf size from slider
ss.smooth_x = deque(ss.smooth_x, maxlen=smooth_sz)
ss.smooth_y = deque(ss.smooth_y, maxlen=smooth_sz)

frame_count = 0

while run:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    H,W   = frame.shape[:2]

    # FPS
    now = time.time()
    ss.fps_buf.append(1.0/(now - ss.last_frame_t + 1e-6))
    ss.last_frame_t = now
    fps = float(np.mean(ss.fps_buf))

    # Canvas
    if ss.canvas is None or ss.canvas.shape[:2]!=(H,W):
        ss.canvas = np.zeros((H,W,3),dtype=np.uint8)
    canvas = ss.canvas

    # ── Keyboard ──────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        k = chr(key).lower()
        for bgr,_,name,num in PALETTE_DEF:
            if k == num:
                ss.color=bgr; ss.color_name=name; ss.mode="DRAW"
        if k=='e': ss.mode="ERASE"
        elif k=='c':
            ss.canvas=np.zeros((H,W,3),dtype=np.uint8); canvas=ss.canvas
        elif key==32:
            ss.calibrated=False; ss.cal_step=0
            ss.cal_iris=[]; ss.cal_screen=[]; ss.cal_buf=[]
            ss.cal_cx=None; ss.cal_cy=None
            ss.kalman=make_kalman()
            ss.smooth_x.clear(); ss.smooth_y.clear()

    # ── MediaPipe ─────────────────────────────────────────
    rgb     = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    detected = False
    iris_nx = iris_ny = 0.0
    l_iris_c = r_iris_c = None
    avg_ear_val = 0.25

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark
        detected = True

        # Head-pose normalised iris
        iris_nx, iris_ny, l_iris_c, r_iris_c = get_iris_norm(lms, W, H)

        # EAR
        le = ear(lms,L_EYE_TOP,L_EYE_BOT,L_EYE_L,L_EYE_R,W,H)
        re = ear(lms,R_EYE_TOP,R_EYE_BOT,R_EYE_L,R_EYE_R,W,H)
        avg_ear_val = (le+re)/2.0
        ss.ear_val  = avg_ear_val

        # Draw iris rings
        for ids,col in [(LEFT_IRIS,(0,255,180)),(RIGHT_IRIS,(0,200,255))]:
            pts = np.array([[int(lms[i].x*W),int(lms[i].y*H)] for i in ids])
            cv2.polylines(frame,[pts],True,col,1)
            cx_i=int(np.mean(pts[:,0])); cy_i=int(np.mean(pts[:,1]))
            cv2.circle(frame,(cx_i,cy_i),2,(255,255,255),-1)

        # Blink detection
        ss.blink_triggered = False
        if avg_ear_val < ss.ear_thresh:
            ss.blink_counter += 1
        else:
            if ss.blink_counter >= BLINK_FRAMES:
                t_now = time.time()
                if t_now - ss.last_blink > BLINK_COOLDOWN:
                    ss.drawing   = not ss.drawing
                    ss.last_blink= t_now
                    ss.blink_count += 1
                    ss.blink_triggered = True
                    if not ss.drawing:
                        ss.prev_x=ss.prev_y=0
            ss.blink_counter = 0

    ss.eye_detected = detected

    # ── Calibration ───────────────────────────────────────
    if not ss.calibrated:
        step = ss.cal_step
        if step < len(CAL_TARGETS):
            frame = draw_cal_overlay(frame, step, len(ss.cal_buf))
            if detected:
                ss.cal_buf.append((iris_nx, iris_ny))
            if len(ss.cal_buf) >= CAL_COLLECT:
                avg_ix = float(np.mean([p[0] for p in ss.cal_buf]))
                avg_iy = float(np.mean([p[1] for p in ss.cal_buf]))
                ss.cal_iris.append((avg_ix,avg_iy))
                ss.cal_screen.append((
                    int(CAL_TARGETS[step][0]*W),
                    int(CAL_TARGETS[step][1]*H),
                ))
                ss.cal_buf=[]; ss.cal_step+=1
        else:
            ss.cal_cx,ss.cal_cy = fit_cal(ss.cal_iris,ss.cal_screen)
            ss.calibrated=True
            ss.prev_x=ss.prev_y=0
            ss.smooth_x.clear(); ss.smooth_y.clear()

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        frame_count += 1
        if frame_count % 5 == 0:
            render_stats(fps,ss.blink_count,avg_ear_val,detected)
            render_status()
        continue

    # ── Gaze → canvas ─────────────────────────────────────
    cursor = None
    if detected and ss.cal_cx is not None:
        raw_x,raw_y = apply_cal(iris_nx,iris_ny,ss.cal_cx,ss.cal_cy,W,H)

        # Kalman
        kx,ky = kf_update(ss.kalman,raw_x,raw_y)

        # Median smoother
        ss.smooth_x.append(kx); ss.smooth_y.append(ky)
        sx = int(np.clip(np.median(ss.smooth_x),0,W-1))
        sy = int(np.clip(np.median(ss.smooth_y),0,H-1))
        cursor = (sx,sy)
        ss.cursor_x,ss.cursor_y = sx,sy

        if ss.drawing:
            px,py = ss.prev_x,ss.prev_y
            if px==0 and py==0: px,py=sx,sy
            dist = math.hypot(sx-px,sy-py)
            if dist < 140:   # reject glitch jumps
                if ss.mode=="DRAW":
                    cv2.line(canvas,(px,py),(sx,sy),ss.color,ss.brush)
                else:
                    cv2.line(canvas,(px,py),(sx,sy),(0,0,0),ss.eraser)
            ss.prev_x,ss.prev_y=sx,sy
        else:
            ss.prev_x=ss.prev_y=0
    else:
        ss.prev_x=ss.prev_y=0

    # ── Blend canvas ──────────────────────────────────────
    gray    = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    _,mask  = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    mask_i  = cv2.bitwise_not(mask)
    bg  = cv2.bitwise_and(frame,frame,mask=mask_i)
    fg  = cv2.bitwise_and(canvas,canvas,mask=mask)
    frame = cv2.add(bg,fg)

    # ── HUD ───────────────────────────────────────────────
    frame = draw_hud(frame,ss,cursor,fps)

    # ── Streamlit display ─────────────────────────────────
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    frame_count += 1
    if frame_count % 5 == 0:
        render_stats(fps,ss.blink_count,ss.ear_val,detected)
        render_status()

cap.release()
cv2.destroyAllWindows()
