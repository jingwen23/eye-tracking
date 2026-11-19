"""
macOS: allow camera access for the app that launches this script (PsychoPy or
Terminal) in System Settings > Privacy & Security > Camera, then restart that app.
Needs a Python 3.9-3.12 interpreter to build the runtime (PsychoPy's own, or a
python.org/Homebrew install is searched automatically).

Startup: 0-5 choose camera, N next, R retry, L layout, ENTER confirms preview.
C: still-head calibration, H: head-movement calibration, R: recenter,
V: 20 independent trials,
L: switch layout (4 columns <-> 2x2 grid; the mapping is cleared, so recalibrate),
O: one-point calibration (reuse the last saved full calibration of this layout,
   then align it with a single look at the centre dot),
G: switch method (regression <-> geometric 3-D eye model; the 3-D model needs no calibration,
   and O/R then only align it to the centre dot; E = optional eyeball-centre calibration:
   look at the camera lens for a moment so the model learns this person's eye positions),
D: gaze-dwell demo on/off, B: back, I: eye diagnostics on/off, Q/ESC: exit.

NEW in this revision (head pose, virtual chin rest, error in degrees):
K: virtual chin rest on/off (starts OFF; set START_CHINREST below),
T: cycle chin-rest tolerance (loose / medium / strict),
P: set the head reference position now (uses the last second of head pose).
Head pose (yaw, pitch, roll in degrees and x/y/z in mm) is logged for every frame in
BOTH methods. A reference position is stored automatically: the centre point of a C or H
calibration, or the moment of R/O alignment. With the chin rest ON, the trial clock and data
collection pause and a banner asks the participant to return to the reference position
whenever the head drifts beyond the tolerance (calibration with H is exempt, since it is
meant to include movement). In normal tracking the estimate is withheld while out of range.
The tolerances in CHINREST_LEVELS are untuned starting values.
The trial CSV now also reports, per trial, accuracy (median gaze point vs target, degrees),
signed bias, precision (RMS of sample-to-sample displacement, and SD, degrees), data loss,
head-hold time and the distance/screen size used. The 4-column layout reports horizontal
error only. Degrees need the eye-to-screen distance: set VIEW_DISTANCE_MM from a ruler
measurement; otherwise it is estimated from head depth, which is only approximate.

NEW in v6 (optional MediaPipe Face Landmarker; extra logging only):
Set LANDMARKER = 'tasks' below to use MediaPipe's newer Face Landmarker (Tasks API) instead of
the legacy FaceMesh. Its landmarks (including the irises) feed the same pipeline, so calibration,
regression and the 3-D model work as before. The Landmarker also provides 52 blendshape scores and
a facial transformation matrix; these are only LOGGED, not used for gaze yet. The frame CSV gets
the 14 eye blendshapes (eyeBlink, eyeLook Up/Down/In/Out, eyeSquint, eyeWide; Left/Right = the
subject's own eyes) and the 16 matrix entries (row-major, raw, not converted to angles), and a
*_blendshapes.csv holds all 52 scores per frame. The model file face_landmarker.task (a few MB) is
downloaded once beside this script; if that fails, download it manually from the MediaPipe Face
Landmarker page and save it next to the script. LANDMARKER = 'facemesh' (default) keeps the
legacy behaviour. Both modes log which landmarker was used, so you can compare the same
participant both ways.

NEW in v7 (fixation detection, ported from Labvanced's open-source detector used in the paper):
A streaming dispersion-based FixationDetector (a step-for-step Python port of the JavaScript
original, checked against it sample by sample) runs on the raw gaze estimates in 'track' and
'test' states. Gaze is converted to millimetres on the screen (horizontal only in the 4-column
layout). A fixation is a window of samples whose mean distance from their centroid stays under
calibration error x 3.25; the allowed growth in dispersion shrinks with fixation duration, and gaps
over 150 ms end a fixation. Calibration error: for the regression method, the in-sample residual of
the calibration fit (never below FIX_MIN_CALIB_ERROR_DEG, because in-sample is optimistic); for the
3-D model and saved one-point models, FIX_DEFAULT_CALIB_ERROR_DEG. The console prints the resulting
limit whenever the detector is reset. Outputs: *_fixations.csv (one row per concluded fixation:
start/end/duration, centroid in mm and screen fractions, dispersion, why it ended, context), a
fix_status column in the frame CSV, and per trial the number of fixations in the voting window,
the longest one's duration and its centroid error in degrees (fix_* columns). Dwell selection still
uses the old strict rule; USE_FIXATIONS = False switches all of this off.

NEW in v8 (3-D model horizontal gain, for when the far left/right boxes are hard to reach):
The 3-D model's horizontal estimate is stretched about the screen centre, separately for the left
and right half: x' = 0.5 + gain * (x - 0.5). The centre stays where O/R aligned it, so only the reach
changes. A gain above 1 makes that side more sensitive.
X: edge stretch (3-D model only). Look at the dot at the far left, then the far right (about 8 s).
   The two gains are set so those dots land exactly on the dots. Do O (or R) first so the centre is
   aligned. Repeat it after changing seat or screen.
]  [   right-side gain up / down by 0.05;   '  ;   left-side gain up / down by 0.05.
The gains are shown on the status line, logged per frame (gain_left, gain_right), and printed when
X sets them; copy them into GEO_GAIN_LEFT / GEO_GAIN_RIGHT to keep them between runs.
Also changed: in 3-D mode the previous frame's raw estimate is cleared each frame, so a dropped
frame can no longer feed a stale value into the O/R alignment or X.

NEW in v9 (clean screen, 3-D model only, word labels, colours):
The main screen now shows only the camera preview and the four boxes: background #004BFF, boxes
#FFB400, labelled Pain / Care / Communication / Basic needs (BOX_LABELS). All on-screen text and
status lines are commented out (the information still goes to the console and the logs).
The script starts in the 3-D geometric model (START_METHOD). The regression calibration is still in
the file but commented out where it is switched on: the C and H keys, the G method switch and the
regression version of the O key are disabled (look for the "# v9" markers to restore them).
Still available by key in 3-D mode: O or R (align to centre; do this first), E, X (edge stretch),
] [ ' ; (gains), V (test), D (dwell), K T P (chin rest), L (layout), I (eye view), M, B, Q.
Procedure prompts stay because you trigger them: the target dots, the E prompt and, with the chin
rest on, the head-moved banner. The gaze cursor ring is hidden (SHOW_GAZE_CURSOR) and the camera
selection screen is unchanged (SHOW_CAMERA_SELECTOR = False skips it when the camera opens).

NEW in v10: the box the gaze is on gets a thick dark-green border (ACTIVE_COLOR #00BF63,
ACTIVE_THICKNESS), drawn inside the box so it never spills into the gaps.

Layouts (both fill the whole screen; the canvas takes the screen's aspect ratio):
  4 columns: horizontal gaze only, nine calibration dots share one row.
             Vertical iris ratios are excluded. Looking above/below a column
             does not deselect it.
  2x2 grid:  horizontal AND vertical gaze, nine calibration dots on a 3x3 grid.
             Vertical gaze is the weaker axis of webcam estimation, so this mode
             adds two eyelid features (mean eyelid opening, iris offset from the
             eyelid centre). C (still head) fits iris + eyelid features only;
             H (head movement) also includes the head descriptors.
             Check accuracy with V before relying on it.

This is experimental webcam gaze estimation, NOT validated eye tracking.
Head geometry descriptors enter a regularised empirical screen mapping. This
is not a calibrated 3-D eye model and cannot guarantee head-motion tolerance.
Calibration warnings do not block use. Unknown frames reset the dwell timer.
No EEG/trigger support. Letter entry remains manual (click or keys 1-7).
Video is not saved. Numeric frame diagnostics, trial results and startup logs
are saved locally. Recalibrate after changing camera geometry/FOV or posture.
After calibration a cyan ring shows filtered gaze (horizontal at a fixed height
in 4-column mode, horizontal + vertical in 2x2 mode).
Display only: median of three frames, then a time-based 120 ms exponential filter.
Validation, selection, and numeric gaze logs continue to use raw estimates.
It is hidden during calibration/testing and when a usable estimate is absent.
"""
from pathlib import Path
import os
import sys
import subprocess
import shutil
import time
import random
import math
import csv
import hashlib
import urllib.request
import traceback
from datetime import datetime
from collections import Counter

ROOT = Path(__file__).resolve().parent
IS_WIN = sys.platform == 'win32'
IS_MAC = sys.platform == 'darwin'
CAMERA_INDEX = 0  # A live selector is shown; this is not assumed to be the Brio.
REQUEST_WIDTH, REQUEST_HEIGHT = 1920, 1080
DWELL_SECONDS = 1.0
W, H = 1200, 800  # H is replaced at start-up so the canvas matches the screen aspect ratio.
START_LAYOUT = 'row'  # 'row' = 4 columns, 'grid' = 2x2

# --- v9 look (OpenCV colours are BGR) ---
BG_COLOR = (255, 75, 0)            # #004BFF
BOX_COLOR = (0, 180, 255)          # #FFB400
BOX_TEXT_COLOR = (20, 20, 20)      # dark text; white on #FFB400 is hard to read
BOX_LABELS = ['Pain', 'Care', 'Communication', 'Basic needs']
ACTIVE_COLOR = (99, 191, 0)        # #00BF63, border of the box the gaze is on
ACTIVE_THICKNESS = 8             # border thickness in canvas pixels (the canvas is 1200 wide)
SHOW_GAZE_CURSOR = True           # True shows the cyan gaze ring again
SHOW_CAMERA_SELECTOR = True        # False: open CAMERA_INDEX directly (the selector still appears if it fails)

GROUPS = [list('ABCDEFG'),list('HIJKLMN'),list('OPQRSTU'),
          ['V','W','X','Y','Z','SPACE','DELETE']]
LAYOUT_NAMES = {'row':'4 columns','grid':'2x2 grid'}
LAYOUT = START_LAYOUT
BOXES, CENTRES, CAL_POINTS, LETTERS, BACK = [], [], [], [], (0,0,0,0)


def configure(layout, height=None):
    """(Re)build every screen-dependent geometry for the chosen layout.
    Boxes span the whole canvas width (and height for the grid); a thin gap
    between boxes stays neutral. Group numbering: row = 1-4 left to right,
    grid = 1 top-left, 2 top-right, 3 bottom-left, 4 bottom-right."""
    global H, LAYOUT, BOXES, CENTRES, CAL_POINTS, LETTERS, BACK
    if height is not None: H = height
    LAYOUT = layout
    m, g = 4, 16  # outer margin and gap between boxes, canvas units
    if layout == 'row':
        bw = (W-2*m-3*g)/4
        top, bottom = round(.34*H), round(.72*H)
        BOXES = [(round(m+i*(bw+g)),top,round(m+i*(bw+g)+bw),bottom) for i in range(4)]
    else:
        bw, bh = (W-2*m-g)/2, (H-2*m-g)/2
        BOXES = [(round(m+c*(bw+g)),round(m+r*(bh+g)),
                  round(m+c*(bw+g)+bw),round(m+r*(bh+g)+bh))
                 for r in range(2) for c in range(2)]
    CENTRES = [((a+c)/2/W,(b+d)/2/H) for a,b,c,d in BOXES]
    if layout == 'row':
        CAL_POINTS = [(.08+i*.105, CENTRES[0][1]) for i in range(9)]
    else:
        # 3x3 grid, index = row*3+col, so index 4 is still the centre.
        CAL_POINTS = [(x,y) for y in (.12,.5,.88) for x in (.1,.5,.9)]
    # Letter screen (manual entry): 4 + 3 boxes spread over the screen.
    top, bottom = round(.2*H), round(.82*H)
    rh = (bottom-top-g)//2
    lw = (W-2*m-3*g)/4
    LETTERS = [(round(m+i*(lw+g)),top,round(m+i*(lw+g)+lw),top+rh) for i in range(4)]
    start = (W-(3*lw+2*g))/2
    LETTERS += [(round(start+i*(lw+g)),top+rh+g,round(start+i*(lw+g)+lw),top+2*rh+g)
                for i in range(3)]
    BACK = (30,round(.86*H),180,round(.97*H))


configure(START_LAYOUT)


def venv_python(runtime):
    return runtime/'Scripts'/'python.exe' if IS_WIN else runtime/'bin'/'python'


def clean_env():
    env = os.environ.copy()
    for name in ('PYTHONHOME','PYTHONPATH'):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
               PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2')
    return env


def find_base_python(env):
    """Return an interpreter (3.9-3.12) that can build the isolated runtime."""
    options = []
    if IS_WIN:
        exe = Path(sys.executable).with_name('python.exe')
        options.append(exe if exe.exists() else Path(sys.executable))
    else:
        options.append(Path(sys.executable))
        for minor in (12,11,10,9):
            name = 'python3.%d' % minor
            found = shutil.which(name)
            if found: options.append(Path(found))
            options += [Path('/opt/homebrew/bin')/name, Path('/usr/local/bin')/name,
                        Path('/Library/Frameworks/Python.framework/Versions/3.%d/bin/%s' % (minor,name))]
    seen = set()
    for option in options:
        if str(option) in seen or not option.exists(): continue
        seen.add(str(option))
        try:
            out = subprocess.run([str(option),'-I','-c',
                'import sys;print(sys.version_info[0],sys.version_info[1])'],env=env,
                capture_output=True,text=True,timeout=30)
            major,minor = (int(v) for v in out.stdout.split()[:2])
            if major == 3 and 9 <= minor <= 12: return option
        except (OSError,ValueError,subprocess.TimeoutExpired):
            continue
    raise RuntimeError('No Python 3.9-3.12 interpreter found to build the gaze runtime. '
                       'Install one from python.org (or "brew install python@3.12") and run again.')


def launch():
    if not (IS_WIN or IS_MAC):
        raise RuntimeError('This version targets Windows and macOS.')
    env = clean_env()
    logpath = ROOT / 'mx_brio_startup.log'
    probe = ('import cv2,numpy,mediapipe as mp; '
             'assert hasattr(mp,"solutions") and hasattr(mp.solutions,"face_mesh"); '
             'assert "GUI:                           NONE" not in cv2.getBuildInformation(); '
             'print("Versions:",cv2.__version__,numpy.__version__,mp.__version__)')
    with logpath.open('a',encoding='utf-8') as log:
        log.write('\nSESSION ' + datetime.now().isoformat() + '\n')
        def run(args):
            child = subprocess.Popen(args,cwd=str(ROOT),env=env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                encoding='utf-8',errors='replace')
            try:
                for line in child.stdout:
                    print(line.rstrip(),flush=True); log.write(line); log.flush()
                if child.wait() != 0:
                    raise RuntimeError('Step failed. Send the last lines of ' + str(logpath))
            finally:
                if child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
        runtime_name = 'gaze_runtime_mx' if IS_WIN else 'gaze_runtime_mx_mac'
        candidates = [venv_python(ROOT/runtime_name)]
        if IS_WIN:
            candidates += [venv_python(ROOT/'gaze_runtime'),
                           venv_python(ROOT.parent/'gaze_runtime')]
        else:
            candidates += [venv_python(ROOT/'gaze_runtime_mac'),
                           venv_python(ROOT.parent/'gaze_runtime_mac')]
        python = None
        for candidate in candidates:
            if not candidate.exists(): continue
            try:
                result = subprocess.run([str(candidate),'-I','-c',probe],env=env,
                    capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
                log.write(str(candidate)+'\n'+result.stdout+result.stderr); log.flush()
                if result.returncode == 0:
                    python = candidate; print(result.stdout,flush=True); break
            except (OSError,subprocess.TimeoutExpired) as exc:
                log.write(str(exc)+'\n')
        if python is None:
            runtime = ROOT/runtime_name
            python = venv_python(runtime)
            print('Preparing an isolated gaze runtime. First setup needs internet; allow several minutes.',flush=True)
            base = find_base_python(env)
            log.write('Base interpreter: '+str(base)+'\n'); log.flush()
            if not python.exists(): run([str(base),'-I','-m','venv',str(runtime)])
            check = ('import sys,pathlib; assert sys.prefix!=sys.base_prefix; '
                     'assert pathlib.Path(sys.prefix).resolve()==pathlib.Path(sys.argv[1]).resolve()')
            run([str(python),'-I','-c',check,str(runtime)])
            run([str(python),'-I','-m','pip','install','--disable-pip-version-check',
                 '--only-binary=:all:','numpy==1.26.4','opencv-contrib-python==4.11.0.86',
                 'mediapipe==0.10.21','jax==0.4.30','jaxlib==0.4.30'])
            run([str(python),'-I','-c',probe])
        print('Using '+str(python),flush=True)
        run([str(python),'-I',str(Path(__file__).resolve()),'--worker'])


FEATURES = 12  # 4 iris ratios, 6 head descriptors, mean eyelid opening, mean iris-to-eyelid-centre offset


def feature_mask(vertical, use_head, np):
    """1 = column used by the mapping, 0 = column zeroed."""
    mask = np.ones(FEATURES)
    if not vertical: mask[[1,3,10,11]] = 0   # 4 columns: horizontal iris + head only
    elif not use_head: mask[4:10] = 0        # 2x2 still-head: iris + eyelid only
    return mask


def fit_mapping(blocks, np, vertical=False, use_head=True):
    """Block-balanced ridge regression using nine known screen targets.
    Features: four eye-local iris ratios + six head geometry descriptors
    + two eyelid features. Cross terms let the eye-to-screen relation vary
    with the other features.
    vertical=False (4 columns): vertical iris/eyelid features are excluded.
    vertical=True (2x2 grid): they are used and the targets are 2-D;
    use_head=False additionally drops the head descriptors.
    """
    if set(b['point'] for b in blocks) != set(range(9)):
        raise ValueError('Calibration requires all nine points.')
    xs,ys = [],[]
    for b in blocks:
        x = np.asarray(b['x'])
        if len(x) < 15: raise ValueError('Too few calibration samples.')
        take = np.linspace(0,len(x)-1,min(50,len(x))).astype(int)
        xs.append(x[take]); ys.append(np.tile(CAL_POINTS[b['point']],(len(take),1)))
    x,y = np.concatenate(xs),np.concatenate(ys)
    mask = feature_mask(vertical,use_head,np)
    x = x*mask
    mean = x.mean(axis=0)
    floors = np.array([.015]*4+[.025,.025,.025,.04,.04,.04,.01,.01])
    scale = np.maximum(x.std(axis=0),floors)
    z = (x-mean)/scale
    design = expand_features(z,np)
    penalty = np.eye(design.shape[1])*4.0
    penalty[0,0] = 0
    weights = np.linalg.solve(design.T@design+penalty,design.T@y)
    centre = np.median(np.concatenate([b['x'] for b in blocks if b['point']==4]),axis=0)*mask
    return {'mean':mean,'scale':scale,'weights':weights,'vertical':vertical,'mask':mask,
            'centre_reference':centre,'recenter_shift':np.zeros(FEATURES),
            'low':np.quantile(x,.01,axis=0)-1.5*scale,
            'high':np.quantile(x,.99,axis=0)+1.5*scale}


def expand_features(z,np):
    z = np.atleast_2d(z)
    interactions = (z[:,:4,None]*z[:,None,4:]).reshape(len(z),-1)
    return np.column_stack([np.ones(len(z)),z,interactions])


def predict(x,model,np):
    if x is None: return None,'eyes_unavailable',None
    if model is None: return None,'not_calibrated',None
    vertical = model.get('vertical',False)
    mask = model.get('mask',np.ones(FEATURES))
    x=(np.asarray(x)-model.get('recenter_shift',np.zeros(FEATURES)))*mask
    if not np.isfinite(x).all(): return None,'invalid_features',None
    if np.any(x < model['low']) or np.any(x > model['high']):
        return None,'outside_sampled_range',None
    xy = (expand_features((x-model['mean'])/model['scale'],np)@model['weights'])[0]
    if not np.isfinite(xy).all(): return None,'invalid_prediction',None
    group = hit_test(xy,vertical)
    return (group,'accepted',xy) if group is not None else (None,'between_or_outside_sections',xy)


def hit_test(xy,vertical):
    """4 columns: horizontal only. 2x2 grid: both axes. Gaps between boxes stay neutral."""
    for g,(a,b,c,d) in enumerate(BOXES):
        if a/W <= xy[0] <= c/W and (not vertical or b/H <= xy[1] <= d/H):
            return g
    return None


def recenter_mapping(samples,model,np):
    """Translate features to the calibrated centre posture; this is not a new fit."""
    values=np.asarray(samples)
    if len(values)<20 or not np.isfinite(values).all():
        raise ValueError('Too few usable samples. Look at centre and press R again.')
    if np.max(np.std(values[:,[0,2]],axis=0))>.025:
        raise ValueError('Eyes moved too much. Hold gaze on centre and press R again.')
    centre=np.median(values,axis=0)*model.get('mask',np.ones(FEATURES))
    model['recenter_shift']=centre-model['centre_reference']


MODEL_DIR = ROOT / 'mx_brio_models'


def save_model(model,layout,shape,np):
    """Keep the latest full calibration of each layout for one-point use."""
    MODEL_DIR.mkdir(exist_ok=True)
    np.savez(MODEL_DIR/('last_full_%s.npz' % layout),
             mean=model['mean'],scale=model['scale'],weights=model['weights'],
             low=model['low'],high=model['high'],centre_reference=model['centre_reference'],
             mask=model['mask'],vertical=np.array(model['vertical']),frame_hw=np.array(shape))


def load_model(layout,shape,np):
    path = MODEL_DIR/('last_full_%s.npz' % layout)
    if not path.exists():
        raise ValueError('No saved %s calibration yet. Run C or H once first.' % LAYOUT_NAMES[layout])
    with np.load(path) as d:
        saved = tuple(int(v) for v in d['frame_hw'])
        if saved != tuple(shape):
            raise ValueError('Saved calibration used camera frames %dx%d, now %dx%d. Run C or H.' %
                             (saved[1],saved[0],shape[1],shape[0]))
        model = {k:d[k].copy() for k in ('mean','scale','weights','low','high','centre_reference','mask')}
        model['vertical'] = bool(d['vertical'])
    model['recenter_shift'] = np.zeros(FEATURES)
    return model


# ---------------------------------------------------------------------------
# Geometric 3-D eye model (calibration-free). Physical constants: edit to match
# your setup. None of these are measured by the script.
# ---------------------------------------------------------------------------
GEO_FOV_DIAG_DEG = 78.0          # MX Brio diagonal FOV setting (65 / 78 / 90 in Logi Options+)
GEO_SCREEN_MM = None             # (width, height) of the visible screen in mm; None = auto-detect
GEO_CAMERA_ABOVE_SCREEN_MM = 15. # camera centre above the top edge of the visible screen
GEO_CAMERA_PITCH_DEG = 0.        # camera tilt, positive = pointing down relative to the screen normal
GEO_EYE_BACK_MM = 10.            # eyeball centre behind the eyelid-margin plane
GEO_EYE_RADIUS_MM = 12.          # eyeball radius (iris centre assumed on this sphere)
GEO_IRIS_DIAMETER_MM = 11.7      # population-average iris diameter, used to refine face scale
# START_METHOD = 'reg'             # v9: regression start disabled
START_METHOD = 'geo'             # v9: 'geo' = geometric 3-D model ('reg' = regression + calibration)
GEO_GAIN_LEFT = 1.0              # 3-D model: horizontal sensitivity of the left half (>1 = reaches further left)
GEO_GAIN_RIGHT = 1.0             # 3-D model: horizontal sensitivity of the right half (>1 = reaches further right)

# Rigid-ish landmarks of MediaPipe's canonical metric face (mm, OpenCV camera axes:
# x right, y down, z away from the camera, face turned toward the camera).
PNP_POINTS = [
    (10,0.0,-82.6,-44.8), (67,-35.2,-80.1,-37.3), (109,-18.9,-82.4,-42.7), (338,18.9,-82.4,-42.7),
    (297,35.2,-80.1,-37.3), (70,-57.2,-42.5,-28.3), (63,-49.9,-48.0,-37.5), (105,-39.9,-51.1,-44.7),
    (66,-27.6,-51.0,-50.2), (107,-14.0,-50.1,-53.2), (336,14.0,-50.1,-53.2), (296,27.6,-51.0,-50.2),
    (334,39.9,-51.1,-44.7), (293,49.9,-48.0,-37.5), (300,57.2,-42.5,-28.3), (168,0.0,-32.7,-52.4),
    (6,0.0,-24.7,-57.9), (197,0.0,-17.3,-63.2), (195,0.0,-10.6,-67.7), (5,0.0,-3.7,-72.4),
    (4,0.0,4.6,-75.9), (1,0.0,11.3,-74.8), (33,-44.5,-26.6,-31.7), (133,-18.6,-25.9,-37.6),
    (362,18.6,-25.9,-37.6), (263,44.5,-26.6,-31.7), (234,-76.6,-6.7,24.4), (127,-77.4,-23.6,20.1),
    (454,76.6,-6.7,24.4), (356,77.4,-23.6,20.1), (93,-75.4,10.5,24.3), (323,75.4,10.5,24.3),
    (98,-14.1,17.1,-52.4), (327,14.1,17.1,-52.4), (129,-17.9,9.8,-48.5), (358,17.9,9.8,-48.5),
]
PNP_IDX = [p[0] for p in PNP_POINTS]
PNP_XYZ = [p[1:] for p in PNP_POINTS]
# Eyelid-margin ring centres (landmark ring of each eye) in the same head frame, and iris landmarks.
EYES = (((-31.6,-26.4,-36.3),468,(469,470,471,472)),   # eye with corners 33/133
        ((31.6,-26.4,-36.3),473,(474,475,476,477)))    # eye with corners 362/263


def physical_screen_mm():
    """Best-effort physical size of the main display in mm (EDID). None if unavailable."""
    try:
        if IS_MAC:
            out = subprocess.run(['osascript','-l','JavaScript','-e',
                'ObjC.import("CoreGraphics"); var s=$.CGDisplayScreenSize($.CGMainDisplayID()); '
                'Math.round(s.width)+" "+Math.round(s.height)'],capture_output=True,text=True,timeout=10)
            w,h = (float(v) for v in out.stdout.split()[:2])
        elif IS_WIN:
            ps = ('Get-CimInstance -Namespace root\\wmi -ClassName WmiMonitorBasicDisplayParams | '
                  'Select-Object -First 1 | ForEach-Object { "$($_.MaxHorizontalImageSize) $($_.MaxVerticalImageSize)" }')
            out = subprocess.run(['powershell','-NoProfile','-Command',ps],
                                 capture_output=True,text=True,timeout=15)
            w,h = (float(v)*10 for v in out.stdout.split()[:2])  # EDID reports whole cm
        else:
            return None
        return (w,h) if w > 100 and h > 60 else None
    except (OSError,ValueError,subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------------------
# Head pose logging, virtual chin rest and error in degrees (added 7 Nov 2026).
# ---------------------------------------------------------------------------
START_CHINREST = False        # virtual chin rest at start-up (K toggles in the program)
START_CHINREST_LEVEL = 1      # index into CHINREST_LEVELS (T cycles)
CHINREST_LEVELS = [           # (name, lateral mm, depth mm, angle deg): untuned starting values
    ('loose', 50., 70., 15.),
    ('medium', 25., 40., 8.),
    ('strict', 12., 20., 4.)]
VIEW_DISTANCE_MM = None       # eye-to-screen distance for degrees. Measure with a ruler and set it;
                              # None = estimate from head depth (approximate: depends on FOV and face size)
EYE_DEPTH_OFFSET_MM = -36.    # eyes sit this far in front of the head-model origin (see EYES above)

LANDMARKER = 'facemesh'       # 'facemesh' = legacy mp.solutions.face_mesh (default); 'tasks' = Face Landmarker
FACE_LANDMARKER_MODEL = ROOT / 'face_landmarker.task'
FACE_LANDMARKER_URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/'
                       'face_landmarker/float16/1/face_landmarker.task')
FACE_LANDMARKER_SHA256 = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
EYE_BLENDSHAPES = ['eyeBlinkLeft','eyeBlinkRight','eyeLookDownLeft','eyeLookDownRight',
                   'eyeLookInLeft','eyeLookInRight','eyeLookOutLeft','eyeLookOutRight',
                   'eyeLookUpLeft','eyeLookUpRight','eyeSquintLeft','eyeSquintRight',
                   'eyeWideLeft','eyeWideRight']


def ensure_landmarker_model():
    """Return the path of face_landmarker.task, downloading it once if it is missing."""
    path = FACE_LANDMARKER_MODEL
    if path.exists() and path.stat().st_size > 500000: return path
    print('Downloading the Face Landmarker model (one time) ...',flush=True)
    tmp = path.with_suffix('.part')
    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_URL,str(tmp))
        if tmp.stat().st_size < 500000: raise OSError('the downloaded file is too small')
        tmp.replace(path)
    except Exception as exc:
        try: tmp.unlink()
        except OSError: pass
        raise RuntimeError('Could not download the Face Landmarker model (%s). Download face_landmarker.task '
                           'from %s and save it as %s, or set LANDMARKER = "facemesh".' %
                           (exc,FACE_LANDMARKER_URL,path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FACE_LANDMARKER_SHA256:
        print('NOTE: the model checksum (%s...) differs from the one this script expects; continuing.' % digest[:12],flush=True)
    return path


def unpack_tasks_result(res, np):
    """FaceLandmarkerResult -> (landmarks or None, {blendshape name: score} or None, 4x4 matrix or None)."""
    lm = res.face_landmarks[0] if res.face_landmarks else None
    blend = ({c.category_name: float(c.score) for c in res.face_blendshapes[0]}
             if res.face_blendshapes else None)
    matrix = (np.asarray(res.facial_transformation_matrixes[0],dtype=float)
              if res.facial_transformation_matrixes else None)
    return lm, blend, matrix


class FaceMeshDetector:
    """Legacy mp.solutions.face_mesh. process(frame_bgr) -> (landmarks, blendshapes, matrix)."""
    def __init__(self, mp, cv2, np):
        self.cv2 = cv2
        self.mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,
                        min_detection_confidence=.6,min_tracking_confidence=.6)
    def process(self, frame):
        result = self.mesh.process(self.cv2.cvtColor(frame,self.cv2.COLOR_BGR2RGB))
        lm = result.multi_face_landmarks[0].landmark if result.multi_face_landmarks else None
        return lm, None, None
    def close(self): self.mesh.close()


class TasksDetector:
    """MediaPipe Face Landmarker (Tasks API) in VIDEO mode, with blendshapes and the transformation matrix."""
    def __init__(self, mp, cv2, np, model_path):
        self.mp, self.cv2, self.np = mp, cv2, np
        vision = mp.tasks.vision
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=.6, min_face_presence_confidence=.6,
            min_tracking_confidence=.6,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.last_ts = 0
    def process(self, frame):
        rgb = self.cv2.cvtColor(frame,self.cv2.COLOR_BGR2RGB)
        ts = int(time.perf_counter()*1000)
        if ts <= self.last_ts: ts = self.last_ts+1     # VIDEO mode needs strictly increasing ms
        self.last_ts = ts
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb)
        return unpack_tasks_result(self.landmarker.detect_for_video(image,ts),self.np)
    def close(self): self.landmarker.close()


def pose_vector(rvec, tvec, np, cv2):
    """[yaw, pitch, roll in degrees, tx, ty, tz in mm] in CAMERA axes (x right in the
    unmirrored image, y down, z away). yaw > 0: nose points toward image right (the user's
    left); pitch > 0: nose points up; roll > 0: head rolled clockwise in the image.
    The metric face looks toward -z, so the nose direction is R @ (0, 0, -1)."""
    R = cv2.Rodrigues(rvec)[0]
    f = R @ np.array([0., 0., -1.])
    r = R @ np.array([1., 0., 0.])
    yaw = math.degrees(math.atan2(f[0], -f[2]))
    pitch = math.degrees(math.atan2(-f[1], math.hypot(f[0], f[2])))
    roll = math.degrees(math.atan2(r[1], r[0]))
    t = np.asarray(tvec, dtype=float).ravel()
    return np.array([yaw, pitch, roll, t[0], t[1], t[2]])


def angle_diff(a, b):
    return (a-b+180.) % 360. - 180.


def chinrest_status(pose, ref, level):
    """Compare a head pose with the reference. Returns (out_of_range, hint, (lateral mm,
    depth mm, max angle deg)). Translation hints are in the USER's own left/right (the
    camera sees a mirror image); rotation hints are deliberately non-directional."""
    d = [angle_diff(pose[i], ref[i]) for i in range(3)]
    dx, dy, dz = pose[3]-ref[3], pose[4]-ref[4], pose[5]-ref[5]
    lateral = math.hypot(dx, dy)
    angle = max(abs(v) for v in d)
    _, lat_tol, dz_tol, ang_tol = level
    ratios = {'lat': lateral/lat_tol, 'dz': abs(dz)/dz_tol, 'ang': angle/ang_tol}
    dev = (lateral, dz, angle)
    if max(ratios.values()) <= 1.:
        return False, '', dev
    worst = max(ratios, key=ratios.get)
    if worst == 'lat':
        if abs(dx) >= abs(dy): hint = 'move your head a little to your %s' % ('right' if dx > 0 else 'left')
        else: hint = 'move your head a little %s' % ('up' if dy > 0 else 'down')
    elif worst == 'dz':
        hint = 'move a little %s' % ('closer to the screen' if dz > 0 else 'back from the screen')
    else:
        hint = 'face the screen squarely and keep your head level'
    return True, hint, dev


def view_distance(poses, np):
    """(mm, source). A configured distance wins; otherwise the median head depth corrected to
    the eye plane; otherwise 600 mm. Only the first is trustworthy for degrees."""
    if VIEW_DISTANCE_MM: return float(VIEW_DISTANCE_MM), 'configured'
    z = [p[5] for p in poses if p is not None]
    if len(z) >= 5: return float(np.median(z))+EYE_DEPTH_OFFSET_MM, 'head_depth_estimate'
    return 600., 'assumed'


def trial_accuracy(points, target, screen_mm, distance_mm, vertical, np):
    """Per-trial accuracy and precision in degrees of visual angle.
    points: one entry per frame in the voting window, (u, v) screen fractions or None;
    target: (u, v) fractions of the target dot; vertical False (4 columns) = horizontal only.
    Errors are estimate minus target (bias_x > 0: estimate right of the target; bias_y > 0: below).
    acc = distance of the MEDIAN gaze point from the target; s2s = RMS of displacement between
    consecutive valid frames (RMS-S2S); sd = spread around the trial mean. loss = share of frames
    with no screen estimate."""
    total = len(points)
    valid = [p for p in points if p is not None]
    out = {'n': len(valid), 'loss': (1.-len(valid)/total) if total else 1.,
           'acc': None, 'bias_x': None, 'bias_y': None, 's2s': None, 'sd': None}
    if len(valid) < 5: return out
    w, h = screen_mm
    def deg(p):
        ex = math.degrees(math.atan2((p[0]-target[0])*w, distance_mm))
        ey = math.degrees(math.atan2((p[1]-target[1])*h, distance_mm)) if vertical else 0.
        return ex, ey
    err = np.array([deg(p) for p in valid])
    cx, cy = float(np.median(err[:,0])), float(np.median(err[:,1]))
    out['acc'] = math.hypot(cx, cy)
    out['bias_x'] = cx
    out['bias_y'] = cy if vertical else None
    out['sd'] = math.sqrt(float(np.var(err[:,0]))+float(np.var(err[:,1])))
    steps = []; prev = None
    for p in points:
        cur = None if p is None else deg(p)
        if prev is not None and cur is not None:
            steps.append((cur[0]-prev[0])**2+(cur[1]-prev[1])**2)
        prev = cur
    if steps: out['s2s'] = math.sqrt(sum(steps)/len(steps))
    return out


def accuracy_fields(acc):
    """CSV fields: acc_deg, bias_x_deg, bias_y_deg, rms_s2s_deg, sd_deg, data_loss_pct."""
    f = lambda v, n=2: '' if v is None else round(v, n)
    return [f(acc['acc']), f(acc['bias_x']), f(acc['bias_y']), f(acc['s2s']), f(acc['sd']),
            round(100.*acc['loss'], 1)]


USE_FIXATIONS = True
FIX_DISP_FACTOR = 3.25              # absolute dispersion limit = calibration error x this (Labvanced default)
FIX_VALUE_AT_MIN = 1.6              # allowed relative dispersion growth at 100 ms (x100 %), Labvanced default
FIX_TIME_AT_ZERO = 280.             # ms at which the allowed growth reaches 0, Labvanced default
FIX_SAMPLE_GAP_MS = 150.            # a gap longer than this ends a fixation, Labvanced default
FIX_DEFAULT_CALIB_ERROR_DEG = 1.5   # used when the calibration fit gives no residual (the paper's webcam mean was ~1.4 deg)
FIX_MIN_CALIB_ERROR_DEG = 1.0       # floor for the in-sample residual, which understates real error


class FixationDetector:
    """Streaming dispersion-based fixation detector, ported from Labvanced's JavaScript
    FixationDetector (github.com/Labvanced/eye-tracking-fixation-detection), step for step.
    Feed one sample at a time with execute(t_ms, x, y); x/y may be in any consistent 2-D unit,
    and calibration_error must be in that same unit. Dispersion = mean Euclidean distance of
    the samples in the candidate window from their centroid. The absolute dispersion limit is
    calibration_error * disp_factor; the allowed relative growth of dispersion shrinks with the
    fixation's duration (see threshold())."""
    def __init__(self, calibration_error, value_at_min=1.6, time_at_zero=280., disp_factor=3.25,
                 sample_thresh=3, sample_gap_ms=150., dup_thresh=0.0001):
        self.value_at_min = value_at_min
        self.time_at_zero = time_at_zero
        self.sample_thresh = sample_thresh
        self.sample_gap_ms = sample_gap_ms
        self.dup_thresh = dup_thresh
        self.disp_threshold = calibration_error * disp_factor
        self.window = []            # samples (t, x, y)
        self.status = None
        self.dropped = 0

    @staticmethod
    def _centroid(w):
        return (sum(p[1] for p in w)/len(w), sum(p[2] for p in w)/len(w))

    def _dispersion(self, w):
        cx, cy = self._centroid(w)
        return sum(((p[1]-cx)**2+(p[2]-cy)**2)**.5 for p in w)/len(w)

    def threshold(self, ms):
        """Allowed relative dispersion growth in percent. Linear from value_at_min*100 % at 100 ms
        to 0 at time_at_zero, then on to -50 % at 5 s. NOTE: the original's second branch measures
        from 100 ms rather than from time_at_zero, so it is already about -2 % at 280 ms; kept as is."""
        min_t, max_t, value_at_max = 100., 5000., -0.5
        ms = min(max(ms, min_t), max_t)
        per1 = self.value_at_min/(self.time_at_zero-min_t)
        per2 = value_at_max/(max_t-self.time_at_zero)
        if ms < self.time_at_zero: return (self.value_at_min-(ms-min_t)*per1)*100
        return ((ms-min_t)*per2)*100

    def _conclude(self, reason):
        w = list(self.window)
        self.status = 'concluded'
        if len(w) < 3: return {'status': None}
        cx, cy = self._centroid(w)
        return {'status': 'concluded', 'start': w[0][0], 'end': w[-1][0], 'duration': w[-1][0]-w[0][0],
                'cx': cx, 'cy': cy, 'dispersion': self._dispersion(w), 'reason': reason, 'n': len(w)}

    def peek(self):
        """The ongoing fixation so far (without concluding it), or None."""
        if self.status != 'ongoing' or len(self.window) < 3: return None
        w = self.window; cx, cy = self._centroid(w)
        return {'start': w[0][0], 'end': w[-1][0], 'duration': w[-1][0]-w[0][0],
                'cx': cx, 'cy': cy, 'dispersion': self._dispersion(w), 'n': len(w)}

    def execute(self, t, x, y):
        data = (t, x, y)
        if self.window:
            last = self.window[-1]
            if t-last[0] > self.sample_gap_ms:
                if self.status == 'ongoing':
                    out = self._conclude('time_difference')
                    self.window = [data]
                    return out
                self.dropped += len(self.window)
                self.window = [data]
                if self.status == 'concluded': self.status = 'none-detected'
                return {'status': self.status or 'none-detected'}
            if abs(last[1]-x) < self.dup_thresh and abs(last[2]-y) < self.dup_thresh:
                if self.status == 'concluded': self.status = 'none-detected'
                self.dropped += 1
                return {'status': self.status or 'none-detected'}
        if len(self.window) <= self.sample_thresh:
            self.window.append(data)
            self.status = 'none-detected'
            return {'status': self.status}
        current = self._dispersion(self.window)
        pushshift = self.window[1:]+[data]
        disp_ps = self._dispersion(pushshift)
        start = self.window[0][0]
        if disp_ps < current:
            self.dropped += 1
            self.window = pushshift
            if disp_ps < self.disp_threshold:
                self.status = 'ongoing'
                return {'status': 'ongoing', 'start': start}
            self.status = 'none-detected'
            return {'status': 'none-detected'}
        if current >= self.disp_threshold:
            if self.status == 'ongoing':
                return {'status': self.status or 'none-detected'}
            self.dropped += 1
            self.window.pop(0)
            self.status = 'none-detected'
            return {'status': 'none-detected'}
        self.status = 'ongoing'
        pushed = self.window+[data]
        disp_push = self._dispersion(pushed)
        duration = t-self.window[0][0]
        allowed = self.threshold(duration)
        if current: grow = (disp_push-current)/current*100
        else: grow = float('inf') if disp_push > current else float('nan')
        if grow > allowed:
            out = self._conclude('rel_threshold'); self.window = [data]; return out
        if disp_push > self.disp_threshold:
            out = self._conclude('abs_threshold'); self.window = [data]; return out
        self.window = pushed
        return {'status': 'ongoing', 'start': start}


def calibration_residual_mm(model, blocks, screen_mm, vertical, np):
    """In-sample mean distance (mm on the screen) between the fitted mapping and the calibration
    targets. Optimistic, because the same samples were used for the fit (horizontal only if not vertical)."""
    w, h = screen_mm
    errs = []
    for b in blocks:
        z = (np.asarray(b['x'])*model['mask']-model['mean'])/model['scale']
        xy = expand_features(z,np)@model['weights']
        tgt = CAL_POINTS[b['point']]
        dx = (xy[:,0]-tgt[0])*w
        dy = (xy[:,1]-tgt[1])*h if vertical else np.zeros(len(xy))
        errs.append(float(np.mean(np.hypot(dx,dy))))
    return float(np.mean(errs))


def fixation_calibration_error_mm(residual_mm, distance_mm):
    """Calibration error (mm) that sets the detector's dispersion limit."""
    default = distance_mm*math.tan(math.radians(FIX_DEFAULT_CALIB_ERROR_DEG))
    if residual_mm is None: return default
    return max(residual_mm, distance_mm*math.tan(math.radians(FIX_MIN_CALIB_ERROR_DEG)))


def fixation_trial_summary(events, win_start_ms, target_mm, distance_mm, vertical):
    """Fixations that end at or after the voting window starts; the longest gives the centroid error (deg)."""
    cand = [e for e in events if win_start_ms is not None and e['end'] >= win_start_ms]
    out = {'n': len(cand), 'dur': None, 'acc': None, 'disp': None}
    if not cand: return out
    e = max(cand, key=lambda ev: ev['duration'])
    ex = math.degrees(math.atan2(e['cx']-target_mm[0], distance_mm))
    ey = math.degrees(math.atan2(e['cy']-target_mm[1], distance_mm)) if vertical else 0.
    out.update(dur=e['duration'], acc=math.hypot(ex, ey), disp=e['dispersion'])
    return out


class GeoGaze:
    """Head pose (PnP on a metric face) + eyeball sphere -> gaze rays -> screen plane.
    Camera axes: x right in the image, y down, z away from the camera. The screen is
    assumed to lie in the plane through the camera centre, facing the user."""
    def __init__(self, np, cv2):
        self.np = np; self.cv2 = cv2
        self.model = np.array(PNP_XYZ,dtype=np.float64)
        self.shift = np.zeros(2)
        self.eye_head = None      # per-person eyeball centres in the head frame (set by eye calibration)
        self.capture = False; self.eye_buffer = []
        self.gain_left = GEO_GAIN_LEFT; self.gain_right = GEO_GAIN_RIGHT
        self.reset()

    def set_camera(self, width, height, screen_mm):
        np = self.np
        self.f = (math.hypot(width,height)/2)/math.tan(math.radians(GEO_FOV_DIAG_DEG)/2)
        self.cx, self.cy = width/2, height/2
        self.K = np.array([[self.f,0,self.cx],[0,self.f,self.cy],[0,0,1.]])
        self.screen_mm = screen_mm
        a = math.radians(GEO_CAMERA_PITCH_DEG)
        self.pitch = np.array([[1,0,0],[0,math.cos(a),math.sin(a)],[0,-math.sin(a),math.cos(a)]])
        self.reset()

    def reset(self):
        self.rvec = self.tvec = None
        self.p_rvec = self.p_tvec = None   # separate warm start for head_pose()
        self.scale = 1.0
        self.raw = None; self.pre = None; self.depth = None; self.buffer = []

    def head_pose(self, xy):
        """Head pose from a PnP fit of the metric face at a FIXED generic face size (scale 1.0),
        so readings stay comparable through a session and work in both methods. Angles are
        reasonable; depth/lateral mm inherit the generic face size and the FOV constant, so use
        them as relative changes, not as exact distances. Returns pose_vector() or None."""
        np, cv2 = self.np, self.cv2
        try:
            pts = np.ascontiguousarray(xy[PNP_IDX],dtype=np.float64)
            guess = self.p_rvec is not None
            ok,rvec,tvec = cv2.solvePnP(self.model,pts,self.K,None,
                                        None if not guess else self.p_rvec.copy(),
                                        None if not guess else self.p_tvec.copy(),
                                        guess,cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            self.p_rvec = self.p_tvec = None
            return None
        if not ok or not np.isfinite(tvec).all() or not 200 < float(tvec.ravel()[2]) < 3000:
            self.p_rvec = self.p_tvec = None
            return None
        self.p_rvec,self.p_tvec = rvec,tvec
        return pose_vector(rvec,tvec,np,cv2)

    def estimate(self, xy):
        """Return (screen_uv_with_shift or None, reason). self.raw keeps the unshifted value."""
        np, cv2 = self.np, self.cv2
        self.raw = None; self.pre = None
        pts = np.ascontiguousarray(xy[PNP_IDX],dtype=np.float64)
        guess = self.rvec is not None
        ok,rvec,tvec = cv2.solvePnP(self.model*self.scale,pts,self.K,None,
                                    None if not guess else self.rvec.copy(),
                                    None if not guess else self.tvec.copy(),
                                    guess,cv2.SOLVEPNP_ITERATIVE)
        if not ok or not np.isfinite(tvec).all() or not 200 < float(tvec.ravel()[2]) < 3000:
            self.rvec = self.tvec = None
            return None,'geo_pose_failed'
        self.rvec,self.tvec = rvec,tvec
        R = cv2.Rodrigues(rvec)[0]; t = tvec.ravel()
        hits=[]; depths=[]; iris_depth=[]; diameters=[]; samples=[]
        for k,(ring,iris,rim) in enumerate(EYES):
            guess = R@(np.array(ring)*self.scale+np.array([0,0,GEO_EYE_BACK_MM]))+t
            eye = guess if self.eye_head is None else R@(self.eye_head[k]*self.scale)+t
            pix = xy[iris]
            ray = np.array([(pix[0]-self.cx)/self.f,(pix[1]-self.cy)/self.f,1.])
            ray /= np.linalg.norm(ray)
            if self.capture:
                # Looking at the camera lens: the eyeball centre lies on the camera ray through
                # the iris. Keep the generic depth, correct the sideways position, store in head frame.
                samples.append(R.T@(ray*float(ray@guess)-t)/self.scale)
            b = float(ray@eye); disc = b*b-float(eye@eye)+GEO_EYE_RADIUS_MM**2
            lam = b-math.sqrt(disc) if disc >= 0 else b   # near side of the eyeball sphere
            point = lam*ray
            gaze = point-eye; gaze /= np.linalg.norm(gaze)  # eyeball centre -> iris centre
            es,gs = self.pitch@eye,self.pitch@gaze
            if gs[2] > -.05: return None,'gaze_away_from_screen'
            hits.append((es+(-es[2]/gs[2])*gs)[:2])
            depths.append(eye[2]); iris_depth.append(point[2])
            r = xy[list(rim)]
            diameters.append(max(float(np.ptp(r[:,0])),float(np.ptp(r[:,1]))))
        if self.capture and len(samples) == 2: self.eye_buffer.append(np.array(samples))
        # Refine face scale from the (near-constant) iris diameter, slowly.
        if min(diameters) > 4:
            z_iris = self.f*GEO_IRIS_DIAMETER_MM/float(np.mean(diameters))
            wanted = self.scale*z_iris/float(np.mean(iris_depth))
            self.scale += .02*(min(1.15,max(.85,wanted))-self.scale)
        px,py = np.mean(hits,axis=0)
        w_mm,h_mm = self.screen_mm
        self.raw = np.array([.5-px/w_mm,(py-GEO_CAMERA_ABOVE_SCREEN_MM)/h_mm])
        self.depth = float(np.mean(depths))
        self.pre = self.raw+self.shift            # aligned, before the horizontal gain
        out = self.pre.copy()
        d = out[0]-.5
        out[0] = .5+d*(self.gain_right if d > 0 else self.gain_left)
        return out,'ok'

    def edge_gain(self, samples, target_u):
        """Gain that moves the median pre-gain x estimate (looking at an edge dot) onto that dot."""
        np = self.np
        v = np.asarray(samples)
        if len(v) < 20: raise ValueError('Too few usable samples. Look at the dot and press X again.')
        if float(np.std(v)) > .08: raise ValueError('Gaze not steady. Look at the dot and press X again.')
        d = float(np.median(v))-.5; want = target_u-.5
        if abs(d) < .03 or d*want < 0:
            raise ValueError('The estimate did not move toward that dot. Look at it and press X again.')
        return min(5.,max(.5,want/d))

    def finish_eye_calibration(self):
        """Median of the buffered look-at-camera samples -> per-person eyeball centres."""
        np = self.np
        values = np.asarray(self.eye_buffer)
        if len(values) < 20: raise ValueError('Too few usable samples. Look at the camera lens and press E again.')
        if float(np.max(np.std(values,axis=0))) > 6.:
            raise ValueError('Head or gaze not steady. Look at the camera lens, hold still, press E again.')
        self.eye_head = np.median(values,axis=0)
        self.shift = np.zeros(2)   # any earlier centre-dot alignment no longer applies

    def align(self, target, vertical):
        """One-point alignment: shift the screen estimate so the buffered raw
        estimates (user looking at the centre dot) land on the dot."""
        np = self.np
        values = np.asarray(self.buffer)
        if len(values) < 20: raise ValueError('Too few usable samples. Look at centre and press R/O again.')
        if np.max(np.std(values,axis=0)) > .08:
            raise ValueError('Gaze not steady. Hold gaze on the centre dot and try again.')
        shift = np.asarray(target)-np.median(values,axis=0)
        if not vertical: shift[1] = 0.
        self.shift = shift


def edge_dots():
    """(left, right) calibration-dot indices used by the edge stretch (X)."""
    return (0,8) if LAYOUT=='row' else (3,5)


def geo_predict(x, xy, geo, vertical):
    if x is None: return None,'eyes_unavailable',None
    uv,why = geo.estimate(xy)
    if uv is None: return None,why,None
    if not geo.np.isfinite(uv).all(): return None,'invalid_prediction',None
    group = hit_test(uv,vertical)
    return (group,'accepted',uv) if group is not None else (None,'between_or_outside_sections',uv)


class GazeDisplayFilter:
    """Suppress isolated spikes and smooth display jitter without changing decisions."""
    def __init__(self, tau=.12):
        self.tau=tau
        self.reset()

    def reset(self):
        self.history=[]; self.value=None; self.last=None

    def update(self, value, now):
        if value is None or not math.isfinite(value) or not 0<=value<=1:
            self.reset()
            return None
        if self.last is not None and (now-self.last>.30 or now<self.last):
            self.reset()
        dt=0. if self.last is None else now-self.last
        self.last=now
        self.history=(self.history+[float(value)])[-3:]
        if len(self.history)<3: return None
        median=sorted(self.history)[1]
        if self.value is None: self.value=median
        else:
            alpha=1.-math.exp(-dt/self.tau)
            self.value+=alpha*(median-self.value)
        return self.value


class Dwell:
    def __init__(self): self.reset()
    def reset(self): self.group=None; self.start=None; self.last=None
    def update(self,group,now):
        gap = self.last is not None and now-self.last > .30
        self.last = now
        if group is None:
            self.group=None; self.start=None; return None,0.0
        if gap or group != self.group or self.start is None:
            self.group=group; self.start=now
        fraction = min(1.0,max(0.0,(now-self.start)/DWELL_SECONDS))
        if fraction >= 1:
            chosen=self.group; self.reset(); return chosen,1.0
        return None,fraction


def summarise_votes(votes):
    valid=[v for v in votes if v is not None]
    if len(votes)<10 or len(valid)<10 or len(valid)/len(votes)<.6: return None
    candidate,count=Counter(valid).most_common(1)[0]
    return candidate if count/len(valid)>=.6 else None


def extract(lm,width,height,np):
    """lm: sequence of landmarks with normalised .x/.y, or None when no face was found."""
    if lm is None: return None,'no_face',None
    if len(lm)<478: return None,'no_iris',None
    xy=np.array([(p.x*width,p.y*height) for p in lm])
    values=[]; openings=[]; lid_offsets=[]
    for a,b,top,bottom,iris in [(33,133,159,145,468),(362,263,386,374,473)]:
        if np.any(xy[[a,b,top,bottom,iris]]<0) or np.any(xy[[a,b,top,bottom,iris]]>np.array([width,height])):
            return None,'eye_out_of_frame',xy
        axis=xy[b]-xy[a]
        if axis[0]<0: axis=-axis
        distance=np.linalg.norm(axis)
        if distance<16: return None,'eye_too_small',xy
        axis/=distance; vertical=np.array([-axis[1],axis[0]])
        opening=abs(np.dot(xy[bottom]-xy[top],vertical))/distance
        if opening<.10: return None,'blink_or_occlusion',xy
        offset=xy[iris]-(xy[a]+xy[b])/2
        values += [np.dot(offset,axis)/distance,np.dot(offset,vertical)/distance]
        # Eyelid cues for vertical gaze: aperture, and iris position relative to the eyelid centre.
        openings.append(opening)
        lid_offsets.append(np.dot(xy[iris]-(xy[top]+xy[bottom])/2,vertical)/distance)
    left=(xy[33]+xy[133])/2; right=(xy[362]+xy[263])/2
    axis=right-left; span=np.linalg.norm(axis)
    if span<25: return None,'face_too_small',xy
    if axis[0]<0: axis=-axis
    midpoint=(left+right)/2
    # Nose offsets/roll are pose proxies, not calibrated yaw/pitch measurements.
    head=[midpoint[0]/width,midpoint[1]/height,span/width,
          (xy[1,0]-midpoint[0])/span,(xy[1,1]-midpoint[1])/span,
          np.arctan2(axis[1],axis[0])]
    eyelid=[float(np.mean(openings)),float(np.mean(lid_offsets))]
    return np.array(values+head+eyelid),'eyes_available',xy


def open_camera(index,cv2):
    if IS_WIN: backends=(cv2.CAP_DSHOW,cv2.CAP_MSMF)
    elif IS_MAC: backends=(cv2.CAP_AVFOUNDATION,cv2.CAP_ANY)
    else: backends=(cv2.CAP_ANY,)
    for backend in backends:
        cap=cv2.VideoCapture(index,backend)
        if not cap.isOpened(): cap.release(); continue
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,REQUEST_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,REQUEST_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,30)
        # macOS may need longer while the camera permission prompt/warm-up finishes.
        for attempt in range(60 if IS_MAC else 20):
            ok,frame=cap.read()
            if ok and frame is not None and frame.size:
                print('Camera %d, backend %s, actual frame %dx%d' %
                      (index,cap.getBackendName(),frame.shape[1],frame.shape[0]),flush=True)
                return cap,frame
            if IS_MAC: time.sleep(.05)
        cap.release()
    return None,None


def screen_size():
    """Primary screen size in points/pixels as used by the OpenCV window."""
    if IS_WIN:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            try: ctypes.windll.user32.SetProcessDPIAware()
            except (AttributeError, OSError): pass
        return (ctypes.windll.user32.GetSystemMetrics(0),
                ctypes.windll.user32.GetSystemMetrics(1))
    if IS_MAC:
        try:
            out=subprocess.run(['osascript','-l','JavaScript','-e',
                'ObjC.import("AppKit"); var f=$.NSScreen.mainScreen.frame; '
                'Math.round(f.size.width)+" "+Math.round(f.size.height)'],
                capture_output=True,text=True,timeout=10)
            w,h=(int(float(v)) for v in out.stdout.split()[:2])
            if w>0 and h>0: return w,h
        except (OSError,ValueError,subprocess.TimeoutExpired): pass
        try:
            out=subprocess.run([sys.executable,'-I','-c',
                'import tkinter as t;r=t.Tk();print(r.winfo_screenwidth(),r.winfo_screenheight());r.destroy()'],
                capture_output=True,text=True,timeout=10)
            w,h=(int(v) for v in out.stdout.split()[:2])
            if w>0 and h>0: return w,h
        except (OSError,ValueError,subprocess.TimeoutExpired): pass
    print('Could not read screen size; assuming 1440x900.',flush=True)
    return 1440,900


def worker():
    import cv2
    import numpy as np
    import mediapipe as mp
    if LANDMARKER not in ('facemesh','tasks'):
        raise RuntimeError('LANDMARKER must be "facemesh" or "tasks".')
    model_path=ensure_landmarker_model() if LANDMARKER=='tasks' else None
    name='MX Brio - gaze prototype'
    diagnostic_name='Iris diagnostics - model estimates, not measured gaze'
    sw,sh=screen_size()
    # The canvas takes the screen's aspect ratio, so boxes reach every screen edge
    # without letterbox margins or stretched text.
    configure(START_LAYOUT,round(W*sh/sw))
    draw_w,draw_h=sw,sh
    offset_x,offset_y=0,0
    canvas=None; click=None
    def put(text,x,y,size=.6,color=(235,235,235),thick=1):
        cv2.putText(canvas,str(text),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,size,color,thick,cv2.LINE_AA)
    def centred(text,cx,y,size=.6,color=(235,235,235),thick=1):
        width=cv2.getTextSize(str(text),cv2.FONT_HERSHEY_SIMPLEX,size,thick)[0][0]
        put(text,cx-width/2,y,size,color,thick)
    def draw_dot(x,y):
        # v9: white dot with a dark ring, visible on both the blue background and the orange boxes.
        cv2.circle(canvas,(x,y),16,(20,20,20),-1,cv2.LINE_AA)
        cv2.circle(canvas,(x,y),11,(255,255,255),-1,cv2.LINE_AA)
    def status(items):
        """Draw (text,y,size,colour) lines at x=25. Grid boxes are filled, so those
        lines get a dark backing plate to stay readable."""
        if LAYOUT=='grid':
            widest=max(cv2.getTextSize(str(t),cv2.FONT_HERSHEY_SIMPLEX,s,1)[0][0] for t,y,s,c in items)
            cv2.rectangle(canvas,(12,min(y for t,y,s,c in items)-28),
                          (38+widest,max(y for t,y,s,c in items)+12),(20,20,20),-1)
        for t,y,s,c in items: put(t,25,y,s,c)
    def fit_image(image,x,y,w,h):
        scale=min(w/image.shape[1],h/image.shape[0])
        im=cv2.resize(image,(max(1,round(image.shape[1]*scale)),max(1,round(image.shape[0]*scale))))
        ih,iw=im.shape[:2]; xx=x+(w-iw)//2; yy=y+(h-ih)//2
        canvas[yy:yy+ih,xx:xx+iw]=im
    def show():
        out=np.full((sh,sw,3),25,np.uint8)
        out[offset_y:offset_y+draw_h,offset_x:offset_x+draw_w]=cv2.resize(canvas,(draw_w,draw_h))
        cv2.imshow(name,out)
        key=cv2.waitKey(1)&255
        # WND_PROP_VISIBLE is unreliable with the macOS (Cocoa) backend; use Q/Esc there.
        if IS_WIN and cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1: return 27
        return key
    def mouse(event,x,y,flags,param):
        nonlocal click
        if event==cv2.EVENT_LBUTTONDOWN:
            click=((x-offset_x)*W/draw_w,(y-offset_y)*H/draw_h)
    def inside(point,box):
        return point is not None and box[0]<=point[0]<=box[2] and box[1]<=point[1]<=box[3]
    def iris_overlay(frame,xy,usable):
        # Draw only on a display copy, after landmark extraction.
        overlay=frame.copy()
        if xy is None: return overlay
        color=(80,255,80) if usable else (0,180,255)
        thickness=max(2,round(frame.shape[1]/170))
        for centre_id,rim in ((468,[469,470,471,472]),(473,[474,475,476,477])):
            centre=xy[centre_id]
            points=xy[rim]
            if not np.isfinite(points).all() or not np.isfinite(centre).all(): continue
            radius=max(2,round(float(np.mean(np.linalg.norm(points-centre,axis=1)))))
            position=tuple(np.rint(centre).astype(int))
            cv2.circle(overlay,position,radius,color,thickness,cv2.LINE_AA)
            cv2.circle(overlay,position,max(1,thickness//2),color,-1,cv2.LINE_AA)
        return overlay
    def eyes_preview(frame,xy):
        board=np.full((270,820,3),25,np.uint8)
        if xy is None: return board
        for j,ids in enumerate(([33,133,159,145,468,469,470,471,472],
                                [362,263,386,374,473,474,475,476,477])):
            pts=xy[ids]; lo=pts.min(axis=0); hi=pts.max(axis=0)
            pad=max(12,(hi[0]-lo[0])*.4)
            a=max(0,int(lo[0]-pad)); b=max(0,int(lo[1]-pad))
            c=min(frame.shape[1],int(hi[0]+pad)); d=min(frame.shape[0],int(hi[1]+pad))
            if c<=a or d<=b: continue
            crop=frame[b:d,a:c].copy()
            centre=xy[ids[4]]-np.array([a,b])
            outline=xy[ids[5:]]-np.array([a,b])
            radius=max(2,round(float(np.mean(np.linalg.norm(outline-centre,axis=1)))))
            cv2.circle(crop,tuple(np.rint(centre).astype(int)),radius,(0,255,255),1,cv2.LINE_AA)
            cv2.circle(crop,tuple(np.rint(centre).astype(int)),2,(0,255,0),-1)
            scale=min(390/crop.shape[1],210/crop.shape[0])
            tile=cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)))
            th,tw=tile.shape[:2]; bx=j*410+10; by=45
            board[by:by+th,bx:bx+tw]=tile
            cv2.putText(board,'Eye %d: raw frame pixels enlarged' % (j+1),(bx,25),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,230,230),1)
        return board

    folder=ROOT/'mx_brio_results'; folder.mkdir(exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    framefile=folder/(stamp+'_frames.csv'); trialfile=folder/(stamp+'_trials.csv'); fixfile=folder/(stamp+'_fixations.csv')
    cap=None; mesh=None; model=None; cam_index=CAMERA_INDEX; frame=None
    state='track'; dwell_enabled=False; dwell=Dwell(); cooldown=0.
    cal_type='none'; one_point=False; reg_cal_type='none'
    method=START_METHOD
    cal_id=0; test_id=0; blocks=[]; order=[]; step=0; phase=0.; collected=[]; collected_total=0; retries=0
    votes=[]; trial_rows=[]; typed=''; active_group=0; recent=[]; diagnostic=False
    cursor_filter=GazeDisplayFilter()
    cursor_filter_y=GazeDisplayFilter()
    head_calibration=False
    message='C: calibrate | O: one-point (saved model) | L: layout | I: eye views.'
    fps=0.; last=time.perf_counter(); actual_shape=None
    movement_test=False
    # Head pose / virtual chin rest / degree error state.
    chinrest_on=START_CHINREST; chin_level=START_CHINREST_LEVEL
    pose=None; pose_ref=None; pose_hist=[]; ref_samples=[]
    head_dev=None; hold=False; blocked=False; hold_hint=''; hold_time=0.
    trial_points=[]; trial_poses=[]; trial_acc=[]
    bs_writer=None; bs_handle=None; bs_names=[]
    fixdet=None; fix_reset=True; calib_resid_mm=None; fix_status=''
    trial_fix=[]; trial_win_start=None; trial_fixsum=[]; hold_total=0.
    edge_step=0; edge_samples=[]; edge_gains=[None,None]
    with framefile.open('w',newline='',encoding='utf-8') as ff, trialfile.open('w',newline='',encoding='utf-8') as tf, \
            fixfile.open('w',newline='',encoding='utf-8') as xf:
        fw=csv.writer(ff); tw=csv.writer(tf); xw=csv.writer(xf)
        xw.writerow(['end_time_s','start_ms','end_ms','duration_ms','centroid_x_mm','centroid_y_mm','centroid_u','centroid_v',
                     'dispersion_mm','n_samples','conclusion','state','test_run','trial','target_group',
                     'layout','method','calibration_type','limit_mm'])
        fix_last=[None]
        def new_fixdet():
            dist=view_distance(pose_hist,np)[0]
            err=fixation_calibration_error_mm(calib_resid_mm,dist)
            det=FixationDetector(err,FIX_VALUE_AT_MIN,FIX_TIME_AT_ZERO,FIX_DISP_FACTOR,sample_gap_ms=FIX_SAMPLE_GAP_MS)
            if fix_last[0]!=round(det.disp_threshold,1):     # print only when the limit changes
                fix_last[0]=round(det.disp_threshold,1)
                print('Fixation detector: dispersion limit %.1f mm (about %.1f deg); calibration error %.1f mm (%s).' %
                      (det.disp_threshold,math.degrees(math.atan2(det.disp_threshold,dist)),err,
                       'from the calibration fit' if calib_resid_mm is not None else 'default'),flush=True)
            return det
        def log_fix(ev):
            in_test=state=='test'
            u=ev['cx']/screen_mm[0]; v='' if LAYOUT!='grid' else round(ev['cy']/screen_mm[1],4)
            xw.writerow([round(ev['end']/1000.,3),round(ev['start']),round(ev['end']),round(ev['duration']),
                         round(ev['cx'],2),round(ev['cy'],2),round(u,4),v,round(ev['dispersion'],2),ev['n'],ev['reason'],
                         state,test_id if in_test else '',step+1 if in_test else '',order[step]+1 if in_test else '',
                         LAYOUT,method,cal_type,round(fixdet.disp_threshold,1)])
            xf.flush()
        fw.writerow(['time_s','camera_index','width','height','processing_fps','calibration','test',
                     'state','target','prediction','reason','screen_x','screen_y',
                     'eye1_x','eye1_y','eye2_x','eye2_y','head_x','head_y','head_scale','nose_x','nose_y','head_roll',
                     'lid_open','lid_iris_v','layout','calibration_type','method','geo_depth_mm','geo_face_scale',
                     'yaw_deg','pitch_deg','roll_deg','head_tx_mm','head_ty_mm','head_tz_mm',
                     'dev_lateral_mm','dev_depth_mm','dev_angle_deg','chinrest_hold','landmarker',
                     *['bs_'+n for n in EYE_BLENDSHAPES],*['tm_%d%d' % (i,j) for i in range(4) for j in range(4)],'fix_status','gain_left','gain_right'])
        tw.writerow(['test_run','calibration_run','trial','target_group','predicted_group','correct',
                     'valid_frames','total_frames','movement_instruction','layout','calibration_type',
                     'acc_deg','bias_x_deg','bias_y_deg','rms_s2s_deg','sd_deg','data_loss_pct',
                     'view_distance_mm','distance_source','screen_mm_source','head_hold_s','chinrest','landmarker',
                     'fix_n','fix_longest_ms','fix_acc_deg','fix_dispersion_mm','fix_limit_mm'])
        try:
            cv2.namedWindow(name,cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
            cv2.moveWindow(name,0,0)
            cv2.setWindowProperty(name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(name,mouse)
            cap,frame=open_camera(cam_index,cv2)
            while SHOW_CAMERA_SELECTOR or frame is None:
                if cap is not None:
                    ok,new=cap.read()
                    if ok and new is not None: frame=new
                    else: cap.release(); cap=None; frame=None
                canvas=np.full((H,W,3),25,np.uint8)
                put('SELECT CAMERA - verify the camera by covering its lens',25,40,.85)
                put('0-5: choose index | N: next index | R: retry | ENTER: use this camera | Esc: exit',25,85,.58)
                put('L: switch layout (now: %s)' % LAYOUT_NAMES[LAYOUT],25,114,.55,(0,230,255))
                if frame is not None:
                    fit_image(cv2.flip(frame,1),100,135,1000,H-265)
                    put('Camera index %d | Received %d x %d | Requested 1920 x 1080' %
                        (cam_index,frame.shape[1],frame.shape[0]),25,H-75,.7)
                else:
                    put('No usable frames. Choose another index or close other camera apps and press R.',25,round(.44*H),.65)
                    if IS_MAC:
                        put('macOS: allow camera access for PsychoPy/Terminal in System Settings > Privacy & Security.',25,round(.44*H)+35,.55)
                key=show()
                if key in (27,ord('q')): return
                if key in (10,13) and frame is not None: break
                if key in (ord('l'),ord('L')):
                    configure('grid' if LAYOUT=='row' else 'row')
                selected=None
                if ord('0')<=key<=ord('5'): selected=key-ord('0')
                elif key in (ord('n'),ord('N')): selected=(cam_index+1)%6
                elif key in (ord('r'),ord('R')): selected=cam_index
                if selected is not None:
                    if cap is not None: cap.release()
                    cam_index=selected; cap,frame=open_camera(cam_index,cv2)
            actual_shape=frame.shape[:2]
            geo=GeoGaze(np,cv2)
            screen_mm=GEO_SCREEN_MM; screen_src='configured'
            if screen_mm is None:
                screen_mm=physical_screen_mm(); screen_src='detected'
            if screen_mm is None:
                diag_mm=609.6  # assume a 24-inch screen; set GEO_SCREEN_MM for the geometric mode
                screen_mm=(diag_mm*sw/math.hypot(sw,sh),diag_mm*sh/math.hypot(sw,sh))
                screen_src='assumed'
                print('WARNING: physical screen size unknown; assuming a 24-inch screen. Set GEO_SCREEN_MM.',flush=True)
            print('Geometric mode: screen %.0f x %.0f mm, camera FOV %.0f deg diagonal, camera %.0f mm above the screen.' %
                  (screen_mm[0],screen_mm[1],GEO_FOV_DIAG_DEG,GEO_CAMERA_ABOVE_SCREEN_MM),flush=True)
            print('Degree errors use screen %.0f x %.0f mm (%s) and %s.' %
                  (screen_mm[0],screen_mm[1],screen_src,
                   ('viewing distance %.0f mm (configured)' % VIEW_DISTANCE_MM) if VIEW_DISTANCE_MM
                   else 'a viewing distance estimated from head depth (set VIEW_DISTANCE_MM for reliable degrees)'),flush=True)
            geo.set_camera(actual_shape[1],actual_shape[0],screen_mm)
            if method=='geo': cal_type='geo_uncalibrated'
            mesh=(TasksDetector(mp,cv2,np,model_path) if LANDMARKER=='tasks'
                  else FaceMeshDetector(mp,cv2,np))
            print('Landmarker: '+('Face Landmarker (Tasks API); eye blendshapes and the transformation matrix are logged'
                                  if LANDMARKER=='tasks' else 'legacy FaceMesh'),flush=True)
            print('Camera confirmed. C: calibrate, L: layout, I: iris diagnostics, V: validation, D: dwell demo, K: chin rest.',flush=True)
            while True:
                ok,frame=cap.read()
                if not ok or frame is None: raise RuntimeError('Camera stopped delivering frames. See startup log.')
                if frame.shape[:2]!=actual_shape:
                    raise RuntimeError('Camera resolution changed. Restart and recalibrate at the new setting.')
                lms,blend,tmatrix=mesh.process(frame)
                x,extraction_reason,xy=extract(lms,frame.shape[1],frame.shape[0],np)
                geo.raw=None; geo.pre=None            # no stale value from a previous frame
                if method=='geo': pred,reason,screen=geo_predict(x,xy,geo,LAYOUT=='grid')
                else: pred,reason,screen=predict(x,model,np)
                if x is None: reason=extraction_reason
                now=time.perf_counter(); dt=now-last; last=now
                fps=.9*fps+.1/max(dt,.001) if fps else 1/max(dt,.001)
                # Head pose (both methods), distance from the reference position, chin-rest decision.
                pose=geo.head_pose(xy) if xy is not None else None
                if pose is not None: pose_hist=(pose_hist+[pose])[-30:]
                out_of_range=False; head_dev=None; hold_hint=''
                if pose is not None and pose_ref is not None:
                    out_of_range,hold_hint,head_dev=chinrest_status(pose,pose_ref,CHINREST_LEVELS[chin_level])
                chin_active=(chinrest_on and pose_ref is not None and pose is not None
                             and state in ('settle','collect','test','track','letters')
                             and not (state in ('settle','collect') and head_calibration))
                out_of_range=out_of_range and chin_active
                hold=out_of_range and state in ('settle','collect','test')
                blocked=out_of_range and state in ('track','letters')
                if hold: phase+=dt; hold_time+=dt     # freeze the state timer while paused
                if blocked: pred=None; reason='head_out_of_range'
                target=order[step] if state in ('settle','collect') else order[step] if state=='test' else None
                if USE_FIXATIONS and (fix_reset or fixdet is None):
                    fixdet=new_fixdet(); fix_reset=False
                fix_status=''
                if USE_FIXATIONS and screen is not None and not (hold or blocked) and state in ('track','test'):
                    fx=fixdet.execute(now*1000.,float(screen[0])*screen_mm[0],
                                      float(screen[1])*screen_mm[1] if LAYOUT=='grid' else 0.)
                    fix_status=fx.get('status') or ''
                    if fx.get('status')=='concluded':
                        if state=='test': trial_fix.append(fx)
                        log_fix(fx)
                fw.writerow([now,cam_index,frame.shape[1],frame.shape[0],round(fps,2),cal_id,test_id,state,
                    '' if target is None else target+1,'' if pred is None else pred+1,reason,
                    *(['',''] if screen is None else list(screen)),*(['']*12 if x is None else list(x)),
                    LAYOUT,cal_type,method,
                    *(['',''] if method!='geo' or geo.depth is None else [round(geo.depth,1),round(geo.scale,4)]),
                    *(['']*6 if pose is None else [round(float(v),2) for v in pose]),
                    *(['']*3 if head_dev is None else [round(head_dev[0],1),round(head_dev[1],1),round(head_dev[2],2)]),
                    int(hold or blocked),LANDMARKER,
                    *[('' if blend is None or n not in blend else round(blend[n],4)) for n in EYE_BLENDSHAPES],
                    *(['']*16 if tmatrix is None else [round(float(v),4) for v in np.asarray(tmatrix).ravel()]),fix_status,
                    *(['',''] if method!='geo' else [round(geo.gain_left,3),round(geo.gain_right,3)])])
                if blend is not None:
                    # All 52 blendshape scores per frame, in a separate file (header taken from the first result).
                    if bs_writer is None:
                        bs_names=list(blend.keys())
                        bs_handle=(folder/(stamp+'_blendshapes.csv')).open('w',newline='',encoding='utf-8')
                        bs_writer=csv.writer(bs_handle); bs_writer.writerow(['time_s']+bs_names)
                    bs_writer.writerow([now]+[round(blend.get(n,0.),4) for n in bs_names])
                if state=='settle' and now-phase>=1.2:
                    state='collect'; phase=now
                if state=='collect' and not hold:
                    collected_total+=1
                    if x is not None: collected.append(x.copy())
                    if order[step]==4 and pose is not None: ref_samples.append(pose)
                    if now-phase>=(5.0 if head_calibration else 3.0):
                        if len(collected)<20 or len(collected)/max(1,collected_total)<.6:
                            retries+=1; collected=[]; collected_total=0
                            if retries>=3:
                                state='track'
                                message='Calibration stopped: eyes not clear enough. I: inspect eyes; C: restart.'
                                print(message,flush=True)
                            else:
                                state='settle'; phase=now
                                print('Retrying calibration point',order[step]+1,flush=True)
                        else:
                            blocks.append({'point':order[step],'x':np.array(collected)})
                            if order[step]==4 and len(ref_samples)>=10:
                                pose_ref=np.median(np.asarray(ref_samples),axis=0)
                                print('Head reference set from the centre point.',flush=True)
                            step+=1; collected=[]; collected_total=0; retries=0
                            if step<len(order): state='settle'; phase=now
                            else:
                                model=fit_mapping(blocks,np,LAYOUT=='grid',use_head=head_calibration)
                                calib_resid_mm=calibration_residual_mm(model,blocks,screen_mm,LAYOUT=='grid',np); fix_reset=True
                                state='track'; dwell.reset(); recent=[]
                                cal_type='full_head' if head_calibration else 'full_still'
                                try: save_model(model,LAYOUT,actual_shape,np)
                                except OSError as exc: print('Could not save calibration for one-point use:',exc,flush=True)
                                message=('Head-movement' if head_calibration else 'Still-head')+(' (iris + eyelid only)' if LAYOUT=='grid' and not head_calibration else '')+' calibration complete. V: test | R: recenter.'
                                print(message,flush=True)
                elif state=='recenter_settle':
                    if now-phase>=1.5:
                        state='recenter_collect'; phase=now; geo.buffer=[]; ref_samples=[]
                elif state=='recenter_collect':
                    collected_total+=1
                    if x is not None: collected.append(x.copy())
                    if method=='geo' and geo.raw is not None: geo.buffer.append(geo.raw.copy())
                    if pose is not None: ref_samples.append(pose)
                    if now-phase>=2.0:
                        try:
                            if method=='geo':
                                if len(geo.buffer)/max(1,collected_total)<.6:
                                    raise ValueError('Eyes not clear enough. Press R/O to retry.')
                                geo.align(CAL_POINTS[4],LAYOUT=='grid')
                                cal_id+=1; cal_type='geo_eye_centres+one_point' if geo.eye_head is not None else 'geo_one_point'
                                message='3-D model aligned to the centre dot. V: verify accuracy | D: enable dwell.'
                                print('Geometric alignment shift (screen fractions):',geo.shift,flush=True)
                            else:
                                if len(collected)/max(1,collected_total)<.6:
                                    raise ValueError('Eyes not clear enough. Press R to retry.')
                                recenter_mapping(collected,model,np)
                                cal_id+=1
                            if len(ref_samples)>=10:
                                pose_ref=np.median(np.asarray(ref_samples),axis=0)
                            if method=='geo': pass
                            elif one_point:
                                cal_type='one_point'; one_point=False
                                message='One-point calibration ready (%s). V: verify accuracy | D: enable dwell.' % LAYOUT_NAMES[LAYOUT]
                            else:
                                message='Recentered. V: verify accuracy | D: enable dwell.'
                            if method!='geo': print('Recenter feature offset:',model['recenter_shift'],flush=True)
                        except ValueError as exc:
                            message=str(exc)
                            if one_point and method!='geo':
                                model=None; one_point=False; cal_type='none'
                                message+=' One-point calibration cancelled; press O to retry.'
                        print(message,flush=True)
                        state='track'; cursor_filter.reset(); cursor_filter_y.reset(); dwell.reset(); recent=[]
                elif state=='eye_settle':
                    if now-phase>=1.5:
                        state='eye_collect'; phase=now; geo.eye_buffer=[]; geo.capture=True; collected_total=0
                elif state=='eye_collect':
                    collected_total+=1
                    if now-phase>=2.5:
                        geo.capture=False
                        try:
                            if len(geo.eye_buffer)/max(1,collected_total)<.6:
                                raise ValueError('Eyes not clear enough. Look at the camera lens and press E again.')
                            geo.finish_eye_calibration()
                            cal_id+=1; cal_type='geo_eye_centres'
                            message='Eye centres calibrated. O: align to centre dot (optional) | V: verify accuracy.'
                            print('Eye centres in head frame (mm):',geo.eye_head.round(1).tolist(),flush=True)
                        except ValueError as exc: message=str(exc)
                        print(message,flush=True)
                        state='track'; cursor_filter.reset(); cursor_filter_y.reset(); dwell.reset(); recent=[]
                elif state=='edge_settle':
                    if now-phase>=1.5:
                        state='edge_collect'; phase=now; edge_samples=[]; collected_total=0
                elif state=='edge_collect':
                    collected_total+=1
                    if geo.pre is not None: edge_samples.append(float(geo.pre[0]))
                    if now-phase>=2.0:
                        idx=edge_dots()[edge_step]
                        try:
                            if len(edge_samples)/max(1,collected_total)<.6:
                                raise ValueError('Eyes not clear enough. Press X to retry.')
                            edge_gains[edge_step]=geo.edge_gain(edge_samples,CAL_POINTS[idx][0])
                            if edge_step==0:
                                edge_step=1; state='edge_settle'; phase=now
                            else:
                                geo.gain_left,geo.gain_right=edge_gains
                                cal_id+=1
                                message='Edge stretch set: left x%.2f, right x%.2f. V: verify | D: dwell.' % (geo.gain_left,geo.gain_right)
                                print(message+' (To keep: GEO_GAIN_LEFT=%.2f, GEO_GAIN_RIGHT=%.2f)' % (geo.gain_left,geo.gain_right),flush=True)
                        except ValueError as exc:
                            message=str(exc); print(message,flush=True)
                            edge_gains=[None,None]; state='track'
                        if state!='edge_settle':      # finished, or failed: back to tracking
                            state='track'; cursor_filter.reset(); cursor_filter_y.reset(); dwell.reset(); recent=[]
                elif state=='test' and not hold:
                    if now-phase>=1.0:
                        votes.append(pred)
                        if trial_win_start is None: trial_win_start=now*1000.
                        trial_points.append(None if screen is None else (float(screen[0]),float(screen[1])))
                        trial_poses.append(pose)
                    if now-phase>=2.5:
                        winner=summarise_votes(votes); target=order[step]
                        dist,dist_src=view_distance(trial_poses,np)
                        acc=trial_accuracy(trial_points,CENTRES[target],screen_mm,dist,LAYOUT=='grid',np)
                        trial_acc.append(acc)
                        events=list(trial_fix); pk=fixdet.peek() if fixdet is not None else None
                        if pk is not None:
                            pk['reason']='trial_end'; log_fix(pk); events.append(pk)
                        tgt_mm=(CENTRES[target][0]*screen_mm[0],CENTRES[target][1]*screen_mm[1] if LAYOUT=='grid' else 0.)
                        fs=fixation_trial_summary(events,trial_win_start,tgt_mm,dist,LAYOUT=='grid')
                        trial_fixsum.append(fs)
                        fix_fields=[fs['n'],'' if fs['dur'] is None else round(fs['dur']),
                                    '' if fs['acc'] is None else round(fs['acc'],2),
                                    '' if fs['disp'] is None else round(fs['disp'],1),
                                    round(fixdet.disp_threshold,1) if fixdet is not None else '']
                        row=[test_id,cal_id,step+1,target+1,'' if winner is None else winner+1,
                             int(winner==target),sum(p is not None for p in votes),len(votes),
                             'small_head_movement' if movement_test else 'head_still',LAYOUT,cal_type,
                             *accuracy_fields(acc),round(dist),dist_src,screen_src,round(hold_time,2),
                             CHINREST_LEVELS[chin_level][0] if chinrest_on else 'off',LANDMARKER,*fix_fields]
                        tw.writerow(row); tf.flush(); trial_rows.append(row)
                        votes=[]; trial_points=[]; trial_poses=[]
                        hold_total+=hold_time; trial_fix=[]; trial_win_start=None; fix_reset=True
                        step+=1; phase=now
                        if step==len(order):
                            correct=sum(r[5] for r in trial_rows); unknown=sum(r[4]=='' for r in trial_rows)
                            accs=[a['acc'] for a in trial_acc if a['acc'] is not None]
                            s2s=[a['s2s'] for a in trial_acc if a['s2s'] is not None]
                            loss=100.*float(np.mean([a['loss'] for a in trial_acc]))
                            held=hold_total
                            err_txt=('%.1f' % float(np.median(accs))) if accs else 'n/a'
                            message='Test (%s, %s): %d/20 correct; %d unknown; error %s deg; loss %.0f%%. V: repeat | M: movement' % (
                                LAYOUT_NAMES[LAYOUT],cal_type,correct,unknown,err_txt,loss)
                            print(message,flush=True)
                            print('  Accuracy (median-gaze-point error per trial): median %s deg, mean %s deg | '
                                  'precision RMS-S2S mean %s deg | data loss %.1f%% | chin-rest hold %.1f s | '
                                  'chin rest %s | distance %.0f mm (%s), screen size %s%s' % (
                                  err_txt,('%.1f' % float(np.mean(accs))) if accs else 'n/a',
                                  ('%.2f' % float(np.mean(s2s))) if s2s else 'n/a',loss,held,
                                  CHINREST_LEVELS[chin_level][0] if chinrest_on else 'off',dist,dist_src,screen_src,
                                  '' if (dist_src=='configured' and screen_src!='assumed') else
                                  ' -> degrees are approximate; set VIEW_DISTANCE_MM / GEO_SCREEN_MM'),flush=True)
                            fix_accs=[f['acc'] for f in trial_fixsum if f['acc'] is not None]
                            if USE_FIXATIONS:
                                print('  Fixation detector (centroid of the longest fixation in the voting window): median error %s deg, '
                                      '%d/%d trials with a fixation.' % (('%.1f' % float(np.median(fix_accs))) if fix_accs else 'n/a',
                                      len(fix_accs),len(trial_fixsum)),flush=True)
                            state='track'; dwell.reset(); recent=[]
                        hold_time=0.
                progress=0.; stable=None
                if state=='track':
                    recent=[] if pred is None else (recent+[pred])[-5:]
                    if len(recent)==5:
                        p,n=Counter(recent).most_common(1)[0]
                        if n>=4: stable=p
                    if dwell_enabled and now>=cooldown:
                        # Strict consecutive raw estimates: blinks, disagreement or stalls reset dwell.
                        chosen,progress=dwell.update(pred,now)
                        if chosen is not None:
                            active_group=chosen; state='letters'; click=None
                            print('Gaze opened group',chosen+1,flush=True)
                    else: dwell.reset()
                else: dwell.reset()
                click_now=click; click=None
                if state=='letters':
                    if inside(click_now,BACK):
                        state='track'; cooldown=now+1.; recent=[]
                    else:
                        chosen=next((i for i,b in enumerate(LETTERS) if inside(click_now,b)),None)
                        if chosen is not None:
                            symbol=GROUPS[active_group][chosen]
                            typed=typed[:-1] if symbol=='DELETE' else typed+(' ' if symbol=='SPACE' else symbol)
                            state='track'; cooldown=now+1.; recent=[]
                            print('Manual entry:',symbol,flush=True)
                grid=LAYOUT=='grid'
                canvas=np.full((H,W,3),BG_COLOR,np.uint8)     # v9 background
                # v9: on-screen instructions removed (original lines kept below)
# v9:                 canvas=np.full((H,W,3),25,np.uint8)
# v9:                 put('MX BRIO - GAZE PROTOTYPE',25,32,.8)
# v9:                 put('Camera %d | %dx%d | Processing %.1f fps | Layout: %s (L)' %
# v9:                     (cam_index,frame.shape[1],frame.shape[0],fps,LAYOUT_NAMES[LAYOUT]),25,65,.58)
# v9:                 put('C: calibrate | H: head movement | R: recenter | V: test | D: dwell | I: eyes | K/T/P: chin rest | X: edge gain (3-D) | Q: exit',25,96,.49)
# v9:                 put('Text: '+(typed[-45:] if typed else '(empty)'),25,135,.65)
                # Hide self-preview during calibration/test to avoid drawing gaze off target.
                if state in ('track','letters') and not (grid and state=='track'):
                    fit_image(cv2.flip(iris_overlay(frame,xy,x is not None),1),1015,10,170,128)
                if state=='letters':
                    for i,b in enumerate(LETTERS):
                        cv2.rectangle(canvas,b[:2],b[2:],(85,85,85),-1)
                        centred('%s [%d]' % (GROUPS[active_group][i],i+1),(b[0]+b[2])/2,(b[1]+b[3])//2+10,1.1,thick=2)
                    cv2.rectangle(canvas,BACK[:2],BACK[2:],(110,80,60),-1)
                    label_y=(BACK[1]+BACK[3])//2+8
                    put('BACK',55,label_y,.7)
                    # v9: put('Click a letter or press 1-7. Manual entry; no EEG.',210,label_y,.6)
                else:
                    # v9: one uniform label size so the longest word fits its box
                    label_size=min(1.6,*(.85*(BOXES[i][2]-BOXES[i][0])/cv2.getTextSize(BOX_LABELS[i],cv2.FONT_HERSHEY_SIMPLEX,1.,2)[0][0] for i in range(4)))
                    for g,b in enumerate(BOXES):
                        active=state=='track' and stable==g
                        # v9: boxes filled with BOX_COLOR, green outline when selected, one word label.
                        cv2.rectangle(canvas,b[:2],b[2:],BOX_COLOR,-1)
                        # v10: thick dark-green inset border on the selected box
                        if active:
                            ih=ACTIVE_THICKNESS//2
                            cv2.rectangle(canvas,(b[0]+ih,b[1]+ih),(b[2]-ih,b[3]-ih),ACTIVE_COLOR,ACTIVE_THICKNESS)
                        # v9: cv2.rectangle(canvas,b[:2],b[2:],(80,220,80) if active else BOX_COLOR,4)
                        cx=(b[0]+b[2])/2; cy=(b[1]+b[3])//2
                        label_h=cv2.getTextSize(BOX_LABELS[g],cv2.FONT_HERSHEY_SIMPLEX,label_size,2)[0][1]
                        centred(BOX_LABELS[g],cx,cy+label_h//2,label_size,BOX_TEXT_COLOR,2)
                        # v9: original box drawing, disabled:
# v9:                     for g,b in enumerate(BOXES):
# v9:                         active=state=='track' and stable==g
# v9:                         if grid:
# v9:                             # Filled boxes cover the header text underneath.
# v9:                             cv2.rectangle(canvas,b[:2],b[2:],(45,70,45) if active else (48,48,48),-1)
# v9:                         color=(80,220,80) if active else (85,85,85)
# v9:                         cv2.rectangle(canvas,b[:2],b[2:],color,4)
# v9:                         bw,bh=b[2]-b[0],b[3]-b[1]
# v9:                         size=min(.75*bw/200,bh/200)
# v9:                         thick=max(1,round(size*1.5))
# v9:                         ls=round(50*size); lab=max(.6,size*.55)
# v9:                         cx=(b[0]+b[2])/2; cy=(b[1]+b[3])//2
# v9:                         centred('GROUP '+str(g+1),cx,b[1]+round(55*lab),lab,thick=max(1,round(lab*1.5)))
# v9:                         symbols=['SPC' if t=='SPACE' else 'DEL' if t=='DELETE' else t for t in GROUPS[g]]
# v9:                         y1=cy-ls//4
# v9:                         centred(' '.join(symbols[:4]),cx,y1,size,thick=thick)
# v9:                         centred(' '.join(symbols[4:]),cx,y1+ls,size,thick=thick)
                        if state=='track' and dwell.group==g and progress>0:
                            bar_y=b[1]+10 if grid else b[3]-20
                            cv2.rectangle(canvas,(b[0]+10,bar_y),
                                (b[0]+10+int((b[2]-b[0]-20)*progress),bar_y+10),ACTIVE_COLOR,-1)
                    if state in ('eye_settle','eye_collect'):
                        # The camera sits above the screen: point at it and show a static prompt.
                        cv2.fillPoly(canvas,[np.array([[W//2-34,70],[W//2+34,70],[W//2,6]],np.int32)],(0,230,255))
                        text='Look at the camera lens (above the screen) and hold still'
                        tw_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.8,2)[0][0]
                        cv2.rectangle(canvas,(W//2-tw_//2-20,H//2-40),(W//2+tw_//2+20,H//2+20),(20,20,20),-1)
                        put(text,W//2-tw_//2,H//2,.8,(0,230,255),2)
                    if state in ('recenter_settle','recenter_collect'):
                        cx,cy=CAL_POINTS[4]
                        # v9: cv2.circle(canvas,(round(cx*W),round(cy*H)),13,(0,230,255),-1)
                        draw_dot(round(cx*W),round(cy*H))
                    if state in ('edge_settle','edge_collect'):
                        cx,cy=CAL_POINTS[edge_dots()[edge_step]]
                        # v9: cv2.circle(canvas,(round(cx*W),round(cy*H)),13,(0,230,255),-1)
                        draw_dot(round(cx*W),round(cy*H))
                    if state in ('settle','collect','test'):
                        target=order[step]
                        cx,cy=(CENTRES if state=='test' else CAL_POINTS)[target]
                        # v9: cv2.circle(canvas,(round(cx*W),round(cy*H)),13,(0,230,255),-1)
                        draw_dot(round(cx*W),round(cy*H))
                    cyan=(0,230,255); white=(235,235,235)
                    # During calibration, recentring and testing only the target dot is shown,
                    # so no text can hide or distract from it. Progress is printed to the console.
                    # v9: status lines removed from the screen (original block kept, disabled):
# v9:                     if state not in ('settle','collect','recenter_settle','recenter_collect','test','eye_settle','eye_collect','edge_settle','edge_collect'):
# v9:                         y0=H-80
# v9:                         second='Dwell '+('ON' if dwell_enabled else 'OFF')+' | '+reason.replace('_',' ')+' | '+('3D geometric (G)' if method=='geo' else 'regression (G)')
# v9:                         if grid: second+=' | Text: '+(typed[-20:] if typed else '(empty)')
# v9:                         chin_txt='chin rest '+(CHINREST_LEVELS[chin_level][0] if chinrest_on else 'off')+' (K, T, P)'
# v9:                         if pose is None: head_txt='Head: no pose'
# v9:                         elif pose_ref is None or head_dev is None:
# v9:                             head_txt='Head logged (yaw %+.0f pitch %+.0f roll %+.0f); no reference yet: C/R/O or P' % tuple(pose[:3])
# v9:                         else: head_txt='Head: %.0f mm, %.1f deg from reference' % (head_dev[0],head_dev[2])
# v9:                         if method=='geo': head_txt+=' | gain L %.2f R %.2f' % (geo.gain_left,geo.gain_right)
# v9:                         status([(head_txt+' | '+chin_txt,y0-30,.5,white),(message,y0,.54,white),(second,y0+33,.6,white)])
                    if hold or blocked:
                        # Visible realign prompt (the trial/calibration clock is frozen while it shows).
                        line1='HEAD MOVED: '+hold_hint
                        line2='Return to your starting position  (%.0f mm, %.1f deg away)' % (head_dev[0],head_dev[2])
                        wd=max(cv2.getTextSize(line1,cv2.FONT_HERSHEY_SIMPLEX,.62,2)[0][0],
                               cv2.getTextSize(line2,cv2.FONT_HERSHEY_SIMPLEX,.5,1)[0][0])
                        cv2.rectangle(canvas,(W//2-wd//2-16,4),(W//2+wd//2+16,64),(20,20,20),-1)
                        cv2.rectangle(canvas,(W//2-wd//2-16,4),(W//2+wd//2+16,64),(0,180,255),2)
                        centred(line1,W/2,28,.62,(0,180,255),2)
                        centred(line2,W/2,52,.5,(235,235,235),1)
                # Filter only the visual cursor; raw validation and dwell are unchanged.
                live=state in ('track','letters') and screen is not None and not blocked
                cursor_x=cursor_filter.update(float(screen[0]) if live else None,now)
                cursor_y=cursor_filter_y.update(float(screen[1]) if live and grid else None,now)
                if SHOW_GAZE_CURSOR and cursor_x is not None and (not grid or cursor_y is not None):   # v9: SHOW_GAZE_CURSOR
                    gaze_x=cursor_x*W
                    gaze_y=round(cursor_y*H) if grid else round(CENTRES[0][1]*H)
                    if 0<=gaze_x<W:
                        position=(round(gaze_x),gaze_y)
                        cv2.circle(canvas,position,18,(255,255,0),3,cv2.LINE_AA)
                        cv2.circle(canvas,position,3,(255,255,0),-1,cv2.LINE_AA)
                        # v9: if not grid: centred('Smoothed horizontal gaze',W/2,H-15,.48,(255,255,0))
                if diagnostic and state in ('track','letters'):
                    cv2.imshow(diagnostic_name,eyes_preview(frame,xy))
                key=show()
                if key in (27,ord('q'),ord('Q')): break
                if key in (ord('i'),ord('I')):
                    diagnostic=not diagnostic
                    if not diagnostic:
                        try: cv2.destroyWindow(diagnostic_name)
                        except cv2.error: pass
                if key in (ord('b'),ord('B')):
                    if one_point: model=None; one_point=False; cal_type='none'
                    state='track'; dwell.reset(); cooldown=now+1.; recent=[]
                elif key in (ord('l'),ord('L')) and state=='track':
                    # Calibration targets differ per layout, so the old mapping is discarded.
                    configure('grid' if LAYOUT=='row' else 'row')
                    model=None; blocks=[]; order=[]; dwell_enabled=False; one_point=False
                    cal_type='geo_uncalibrated' if method=='geo' else 'none'
                    geo.shift=np.zeros(2); geo.reset()
                    dwell.reset(); cursor_filter.reset(); cursor_filter_y.reset(); recent=[]
                    pose_ref=None; calib_resid_mm=None; fix_reset=True
                    message='Layout: %s. %s' % (LAYOUT_NAMES[LAYOUT],'3-D model needs no calibration; O aligns it.' if method=='geo' else 'Calibrate again (C or H).')
                    print(message,flush=True)
# v9 disabled:                 elif key in (ord('g'),ord('G')) and state=='track':
# v9 disabled:                     dwell_enabled=False; dwell.reset(); cursor_filter.reset(); cursor_filter_y.reset(); recent=[]
# v9 disabled:                     one_point=False; calib_resid_mm=None; fix_reset=True
# v9 disabled:                     if method=='reg':
# v9 disabled:                         method='geo'; reg_cal_type=cal_type; cal_type='geo_uncalibrated'
# v9 disabled:                         geo.shift=np.zeros(2); geo.reset()
# v9 disabled:                         message='3-D geometric mode: no calibration needed. Optional: E (look at camera lens), O (centre dot).'
# v9 disabled:                     else:
# v9 disabled:                         method='reg'; cal_type=reg_cal_type
# v9 disabled:                         message='Regression mode. '+('Calibrated.' if model is not None else 'Calibrate with C/H, or O for one-point.')
# v9 disabled:                     print(message,flush=True)
                elif key in (ord('e'),ord('E')) and state=='track' and method=='geo':
                    state='eye_settle'; phase=time.perf_counter(); collected_total=0
                    geo.capture=False; geo.eye_buffer=[]; dwell_enabled=False
                    dwell.reset(); cursor_filter.reset(); cursor_filter_y.reset(); recent=[]
                elif key in (ord('x'),ord('X')) and state=='track' and method=='geo':
                    edge_step=0; edge_gains=[None,None]; state='edge_settle'; phase=time.perf_counter()
                    collected_total=0; dwell_enabled=False
                    dwell.reset(); cursor_filter.reset(); cursor_filter_y.reset(); recent=[]
                elif key in (ord('x'),ord('X')) and state=='track':
                    message='X (edge stretch) is for the 3-D model: press G first.'
                elif key in (ord(']'),ord('['),ord("'"),ord(';')) and state=='track' and method=='geo':
                    delta=.05 if key in (ord(']'),ord("'")) else -.05
                    if key in (ord(']'),ord('[')): geo.gain_right=min(5.,max(.5,geo.gain_right+delta))
                    else: geo.gain_left=min(5.,max(.5,geo.gain_left+delta))
                    message="Gain: left x%.2f, right x%.2f  (] [ right, ' ; left, X: auto)." % (geo.gain_left,geo.gain_right)
                    print(message,flush=True)
                elif key in (ord('o'),ord('O')) and state=='track' and method=='geo':
                    state='recenter_settle'; phase=time.perf_counter()
                    collected=[]; collected_total=0; dwell_enabled=False
                    dwell.reset(); cursor_filter.reset(); cursor_filter_y.reset(); recent=[]
                elif key in (ord('c'),ord('C'),ord('h'),ord('H')) and method=='geo':
                    message='3-D geometric mode needs no calibration (O: align to centre dot, G: back to regression).'
# v9 disabled:                 elif key in (ord('o'),ord('O')) and state=='track':
# v9 disabled:                     # One-point calibration: saved full model + a single look at the centre dot.
# v9 disabled:                     try:
# v9 disabled:                         model=load_model(LAYOUT,actual_shape,np); calib_resid_mm=None; fix_reset=True
# v9 disabled:                         one_point=True; state='recenter_settle'; phase=time.perf_counter()
# v9 disabled:                         collected=[]; collected_total=0; dwell_enabled=False
# v9 disabled:                         dwell.reset(); cursor_filter.reset(); cursor_filter_y.reset(); recent=[]
# v9 disabled:                         print('One-point calibration: loaded saved',LAYOUT_NAMES[LAYOUT],'model. Look at the centre dot.',flush=True)
# v9 disabled:                     except (ValueError,OSError,KeyError) as exc:
# v9 disabled:                         message=str(exc); print(message,flush=True)
# v9 disabled:                 elif key in (ord('c'),ord('C'),ord('h'),ord('H')):
# v9 disabled:                     one_point=False
# v9 disabled:                     head_calibration=key in (ord('h'),ord('H'))
# v9 disabled:                     cal_id+=1; model=None; blocks=[]; order=[]; dwell_enabled=False; dwell.reset()
# v9 disabled:                     pose_ref=None; ref_samples=[]
# v9 disabled:                     # Centre first, then the other eight positions (one row, or the 3x3 grid).
# v9 disabled:                     order=[4,0,1,2,3,5,6,7,8]
# v9 disabled:                     step=0; state='settle'; collected=[]; collected_total=0; retries=0
# v9 disabled:                     phase=time.perf_counter(); message='Follow the dots. No Space key needed.'
                elif key in (ord('r'),ord('R')) and state in ('track','letters'):
                    if model is None and method!='geo': message='Calibrate first with C or H.'
                    else:
                        state='recenter_settle'; phase=time.perf_counter()
                        collected=[]; collected_total=0; dwell_enabled=False
                        dwell.reset(); cursor_filter.reset(); cursor_filter_y.reset(); recent=[]
                elif key in (ord('k'),ord('K')) and state=='track':
                    chinrest_on=not chinrest_on
                    message='Virtual chin rest %s (%s).' % ('ON' if chinrest_on else 'OFF',CHINREST_LEVELS[chin_level][0])
                    if chinrest_on and pose_ref is None: message+=' No reference yet: C, R/O or P sets it.'
                    print(message,flush=True)
                elif key in (ord('t'),ord('T')) and state=='track':
                    chin_level=(chin_level+1)%len(CHINREST_LEVELS)
                    lv=CHINREST_LEVELS[chin_level]
                    message='Chin-rest tolerance %s: %.0f mm sideways, %.0f mm depth, %.0f deg.' % (lv[0],lv[1],lv[2],lv[3])
                    print(message,flush=True)
                elif key in (ord('p'),ord('P')) and state=='track':
                    if len(pose_hist)>=15:
                        pose_ref=np.median(np.asarray(pose_hist),axis=0)
                        message='Head reference set at the current position. K: chin rest on/off | T: tolerance.'
                    else: message='No steady face pose yet; look at the screen and press P again.'
                    print(message,flush=True)
                elif key in (ord('m'),ord('M')) and state=='track':
                    movement_test=not movement_test
                    message='Next test: '+('small head movements' if movement_test else 'head still')
                elif key in (ord('d'),ord('D')) and state=='track':
                    if model is None and method!='geo': message='Calibrate first with C.'
                    else:
                        dwell_enabled=not dwell_enabled; dwell.reset(); cooldown=now+1.
                        message='Dwell '+('ON: look at a group for one second.' if dwell_enabled else 'OFF.')
                elif key in (ord('v'),ord('V')) and state=='track' and (model is not None or method=='geo'):
                    test_id+=1; order=list(range(4))*5; random.shuffle(order); step=0
                    votes=[]; trial_rows=[]; phase=time.perf_counter(); state='test'; dwell_enabled=False; recent=[]
                    trial_points=[]; trial_poses=[]; trial_acc=[]; hold_time=0.
                    fix_reset=True; trial_fix=[]; trial_win_start=None; trial_fixsum=[]; hold_total=0.
                elif state=='letters' and ord('1')<=key<=ord('7'):
                    symbol=GROUPS[active_group][key-ord('1')]
                    typed=typed[:-1] if symbol=='DELETE' else typed+(' ' if symbol=='SPACE' else symbol)
                    state='track'; cooldown=now+1.; recent=[]
                if state not in ('track','letters'):
                    # Prevent an old diagnostic window remaining visible during targets.
                    try: cv2.destroyWindow(diagnostic_name)
                    except cv2.error: pass
        finally:
            if bs_handle is not None: bs_handle.close()
            if mesh is not None: mesh.close()
            if cap is not None: cap.release()
            cv2.destroyAllWindows()
            print('Frame log:',framefile,'\nTrial log:',trialfile,flush=True)
            print('Final manually entered text:',typed,flush=True)


if __name__=='__main__':
    try:
        worker() if '--worker' in sys.argv else launch()
    except Exception:
        details=traceback.format_exc()
        print(details,flush=True)
        try:
            with (ROOT/'mx_brio_error.txt').open('a',encoding='utf-8') as f:
                f.write('\n'+datetime.now().isoformat()+'\n'+details)
        except OSError: pass
        sys.exit(1)
