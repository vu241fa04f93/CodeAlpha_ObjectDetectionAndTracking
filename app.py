"""
OBJECT VISION: Intelligent Object Detection & Multi-Object Tracking
Professional Computer Vision Dashboard.
Architecture: Ultralytics YOLO (YOLOv8n / YOLOv8s) + Class-Aware SORT.
Design System: Modern Dark AI Monitoring (#0B1020, #111827, #161F32, #38BDF8, #273449)
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Add workspace to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    COCO_CLASSES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_FRAME_SKIP,
    DEFAULT_IMG_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_NMS_IOU_THRESHOLD,
    DEFAULT_RESOLUTION_WIDTH,
    SAMPLE_VIDEOS_DIR,
    SORT_IOU_THRESHOLD,
    SORT_MAX_AGE,
    SORT_MIN_HITS,
    SUPPORTED_IMG_SIZES,
    SUPPORTED_MODELS,
    SUPPORTED_RESOLUTIONS,
    THEME_COLORS,
    get_optimal_device,
)
from detector import YOLOv8Detector
from tracker import ObjectTracker
from utils.performance import PerformanceMonitor
from utils.visualization import draw_bounding_boxes, draw_tracking_hud, resize_frame


# -----------------------------------------------------------------------------
# Streamlit Page Setup & Custom CSS Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OBJECT VISION | Intelligent Object Detection & Tracking",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply exact design system and typography requirements
st.markdown(f"""
<style>
    /* Global Base */
    .stApp {{
        background-color: {THEME_COLORS["bg"]};
        color: {THEME_COLORS["text"]};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}

    /* Main Container Constraint (Max 1200px) */
    .main .block-container {{
        max-width: 1200px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        margin: 0 auto !important;
    }}

    /* Sidebar (~250px) */
    section[data-testid="stSidebar"] {{
        background-color: {THEME_COLORS["sidebar"]} !important;
        border-right: 1px solid {THEME_COLORS["border"]} !important;
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding: 1.2rem 1rem !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown h3, 
    section[data-testid="stSidebar"] .stMarkdown h4 {{
        font-size: 14px !important;
        color: {THEME_COLORS["accent"]} !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        margin-bottom: 0.4rem !important;
    }}

    /* Header Component */
    .header-box {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding-bottom: 14px;
        border-bottom: 1px solid {THEME_COLORS["border"]};
        margin-bottom: 16px;
    }}
    .header-left .main-heading {{
        font-size: 28px !important;
        font-weight: 800 !important;
        color: {THEME_COLORS["text"]} !important;
        letter-spacing: 0.5px;
        margin: 0 !important;
        line-height: 1.2;
    }}
    .header-left .main-heading span {{
        color: {THEME_COLORS["accent"]};
    }}
    .header-left .sub-heading {{
        font-size: 15px !important;
        font-weight: 600 !important;
        color: {THEME_COLORS["secondary_accent"]} !important;
        margin-top: 2px !important;
    }}
    .header-left .desc-text {{
        font-size: 13px !important;
        color: {THEME_COLORS["secondary_text"]} !important;
        margin-top: 4px !important;
    }}

    /* Status Badges */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.6px;
    }}
    .status-live {{
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid {THEME_COLORS["success"]};
        color: #4ADE80;
    }}
    .status-live::before {{
        content: "";
        width: 7px;
        height: 7px;
        background-color: {THEME_COLORS["success"]};
        border-radius: 50%;
        box-shadow: 0 0 8px {THEME_COLORS["success"]};
    }}
    .status-paused {{
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid {THEME_COLORS["warning"]};
        color: #FBBF24;
    }}
    .status-offline {{
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid {THEME_COLORS["border"]};
        color: {THEME_COLORS["secondary_text"]};
    }}

    /* Metric Cards */
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 16px;
    }}
    .metric-card-box {{
        background-color: {THEME_COLORS["card"]};
        border: 1px solid {THEME_COLORS["border"]};
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }}
    .metric-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }}
    .metric-title {{
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        color: {THEME_COLORS["secondary_text"]} !important;
        letter-spacing: 0.5px;
    }}
    .metric-icon {{
        font-size: 14px;
        opacity: 0.8;
    }}
    .metric-value {{
        font-size: 26px !important;
        font-weight: 800 !important;
        color: {THEME_COLORS["text"]} !important;
        line-height: 1.1;
    }}
    .metric-subtitle {{
        font-size: 11px !important;
        color: {THEME_COLORS["secondary_text"]} !important;
        margin-top: 4px;
    }}

    /* Button Row */
    div.stButton > button {{
        font-size: 14px !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 6px 14px !important;
        border: 1px solid {THEME_COLORS["border"]} !important;
        background-color: {THEME_COLORS["card"]} !important;
        color: {THEME_COLORS["text"]} !important;
        transition: all 0.15s ease-in-out;
    }}
    div.stButton > button:hover {{
        border-color: {THEME_COLORS["accent"]} !important;
        color: {THEME_COLORS["accent"]} !important;
    }}
    div.stButton > button[kind="primary"] {{
        background-color: #0284C7 !important;
        border-color: {THEME_COLORS["accent"]} !important;
        color: #FFFFFF !important;
    }}

    /* Video Center Viewport (Max 800px) */
    .video-center-container {{
        max-width: 800px;
        margin: 0 auto;
        background-color: #000000;
        border: 1px solid {THEME_COLORS["border"]};
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }}

    /* Performance and Statistics Boxes */
    .section-title {{
        font-size: 18px !important;
        font-weight: 700 !important;
        color: {THEME_COLORS["text"]} !important;
        margin: 16px 0 8px 0 !important;
        letter-spacing: 0.3px;
    }}
    .info-bar {{
        background-color: {THEME_COLORS["card"]};
        border: 1px solid {THEME_COLORS["border"]};
        border-radius: 6px;
        padding: 10px 14px;
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        font-size: 13px;
        color: {THEME_COLORS["secondary_text"]};
    }}
    .info-bar strong {{
        color: {THEME_COLORS["text"]};
    }}

    /* Clean Styled Table */
    .styled-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        background-color: {THEME_COLORS["card"]};
        border: 1px solid {THEME_COLORS["border"]};
        border-radius: 6px;
        overflow: hidden;
    }}
    .styled-table th {{
        background-color: #111827;
        color: {THEME_COLORS["accent"]};
        text-align: left;
        padding: 8px 12px;
        font-weight: 600;
        border-bottom: 1px solid {THEME_COLORS["border"]};
    }}
    .styled-table td {{
        padding: 8px 12px;
        color: {THEME_COLORS["text"]};
        border-bottom: 1px solid #1E293B;
    }}
    .styled-table tr:last-child td {{
        border-bottom: none;
    }}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Singleton Model Loader (Loaded ONCE, supports model selection switch)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_detector(model_file_path: str) -> YOLOv8Detector:
    """Returns singleton detector instance for the selected model path."""
    device = get_optimal_device()
    return YOLOv8Detector(model_name_or_path=model_file_path, device=device)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "is_paused" not in st.session_state:
    st.session_state.is_paused = False
if "tracker" not in st.session_state:
    st.session_state.tracker = ObjectTracker(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESHOLD)
if "perf_monitor" not in st.session_state:
    st.session_state.perf_monitor = PerformanceMonitor()
if "frame_counter" not in st.session_state:
    st.session_state.frame_counter = 0


# -----------------------------------------------------------------------------
# Left Sidebar: Compact Collapsible Sections (Width ~250px)
# -----------------------------------------------------------------------------
with st.sidebar:
    # SECTION 1: VIDEO SOURCE
    with st.expander("VIDEO SOURCE", expanded=True):
        source_type = st.radio(
            "Source Type",
            options=["Webcam", "Upload Video", "Sample Video"],
            index=0,
            label_visibility="collapsed"
        )
        video_path = None
        webcam_index = 0
        if source_type == "Webcam":
            webcam_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
        elif source_type == "Upload Video":
            uploaded_file = st.file_uploader("Choose Video", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                video_path = temp_path
        elif source_type == "Sample Video":
            sample_files = sorted(list(Path(SAMPLE_VIDEOS_DIR).glob("*.mp4")) + list(Path(SAMPLE_VIDEOS_DIR).glob("*.avi")))
            if sample_files:
                sample_options = [f.name for f in sample_files]
                default_idx = sample_options.index("pedestrian_bus_demo.mp4") if "pedestrian_bus_demo.mp4" in sample_options else 0
                selected_sample = st.selectbox("Sample Clip", sample_options, index=default_idx)
                video_path = str(SAMPLE_VIDEOS_DIR / selected_sample)
            else:
                st.caption("No sample clips found.")

    # SECTION 2: DETECTION MODEL
    with st.expander("DETECTION MODEL", expanded=True):
        model_selection = st.selectbox(
            "Model Variant",
            options=list(SUPPORTED_MODELS.keys()),
            index=0
        )
        selected_model_path = SUPPORTED_MODELS[model_selection]

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=1.00,
            value=DEFAULT_CONF_THRESHOLD,
            step=0.05
        )

        img_size = st.selectbox(
            "Image Size (imgsz)",
            options=SUPPORTED_IMG_SIZES,
            index=0
        )

        filter_classes = st.checkbox("Filter Classes", value=False)
        selected_classes = None
        if filter_classes:
            popular = ["person", "car", "motorcycle", "bus", "truck", "dog", "chair", "bottle", "laptop"]
            chosen = st.multiselect("Active Classes", COCO_CLASSES, default=popular)
            selected_classes = [COCO_CLASSES.index(c) for c in chosen if c in COCO_CLASSES]

    # SECTION 3: TRACKING
    with st.expander("TRACKING", expanded=False):
        st.caption("Algorithm: **SORT (Class-Aware)**")
        max_age = st.slider("Max Track Age", 5, 60, SORT_MAX_AGE, 5)
        min_hits = st.slider("Min Hits", 1, 5, SORT_MIN_HITS, 1)
        iou_thresh = st.slider("IoU Threshold", 0.1, 0.7, SORT_IOU_THRESHOLD, 0.05)

    # SECTION 4: PERFORMANCE
    with st.expander("PERFORMANCE", expanded=False):
        device_name = get_optimal_device().upper()
        st.write(f"Device: **{device_name}**")
        frame_skip = st.slider("Frame Skipping", 1, 8, DEFAULT_FRAME_SKIP)
        res_choice = st.selectbox(
            "Resolution",
            options=list(SUPPORTED_RESOLUTIONS.keys()),
            index=0
        )
        target_width = SUPPORTED_RESOLUTIONS[res_choice]
        show_motion_trails = st.checkbox("Motion Trails", value=True)
        show_hud_banner = st.checkbox("In-Frame HUD", value=True)


# -----------------------------------------------------------------------------
# Main Content: Header & Proper Content Alignment
# -----------------------------------------------------------------------------
# Status indicator string
if st.session_state.is_running and not st.session_state.is_paused:
    status_html = '<span class="status-pill status-live">LIVE</span>'
elif st.session_state.is_running and st.session_state.is_paused:
    status_html = '<span class="status-pill status-paused">PAUSED</span>'
else:
    status_html = '<span class="status-pill status-offline">OFFLINE</span>'

st.markdown(f"""
<div class="header-box">
    <div class="header-left">
        <div class="main-heading">OBJECT <span>VISION</span></div>
        <div class="sub-heading">Intelligent Object Detection & Multi-Object Tracking</div>
        <div class="desc-text">Detect, classify, and track multiple objects in real time using deep learning and computer vision.</div>
    </div>
    <div class="header-right">
        {status_html}
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Controls Row: Start, Pause, Resume, Stop in one row
# -----------------------------------------------------------------------------
ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1, 1, 1, 4])

with ctrl_c1:
    btn_start = st.button("Start", use_container_width=True, type="primary", disabled=st.session_state.is_running and not st.session_state.is_paused)

with ctrl_c2:
    pause_label = "Resume" if st.session_state.is_paused else "Pause"
    btn_pause = st.button(pause_label, use_container_width=True, disabled=not st.session_state.is_running)

with ctrl_c3:
    btn_stop = st.button("Stop", use_container_width=True, disabled=not st.session_state.is_running)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Metric Cards: Four Equal Cards
# -----------------------------------------------------------------------------
m_card1, m_card2, m_card3, m_card4 = st.columns(4)

with m_card1:
    c1_holder = st.empty()
    c1_holder.markdown("""
    <div class="metric-card-box">
        <div class="metric-header"><span class="metric-title">LIVE FPS</span><span class="metric-icon">⚡</span></div>
        <div class="metric-value">0.0</div>
        <div class="metric-subtitle">Real-time processing rate</div>
    </div>
    """, unsafe_allow_html=True)

with m_card2:
    c2_holder = st.empty()
    c2_holder.markdown("""
    <div class="metric-card-box">
        <div class="metric-header"><span class="metric-title">ACTIVE TRACKS</span><span class="metric-icon">🎯</span></div>
        <div class="metric-value">0</div>
        <div class="metric-subtitle">Currently in field of view</div>
    </div>
    """, unsafe_allow_html=True)

with m_card3:
    c3_holder = st.empty()
    c3_holder.markdown("""
    <div class="metric-card-box">
        <div class="metric-header"><span class="metric-title">UNIQUE OBJECTS</span><span class="metric-icon">🔢</span></div>
        <div class="metric-value">0</div>
        <div class="metric-subtitle">Session cumulative count</div>
    </div>
    """, unsafe_allow_html=True)

with m_card4:
    c4_holder = st.empty()
    c4_holder.markdown("""
    <div class="metric-card-box">
        <div class="metric-header"><span class="metric-title">INFERENCE LATENCY</span><span class="metric-icon">⏱️</span></div>
        <div class="metric-value">0.0 ms</div>
        <div class="metric-subtitle">Neural network forward pass</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Video Preview (Centered, Max Width 800px)
# -----------------------------------------------------------------------------
video_col_l, video_col_m, video_col_r = st.columns([1, 7, 1])
with video_col_m:
    st.markdown('<div class="video-center-container">', unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Detection Statistics (Clean Table: Object Class | Count | Tracking IDs)
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">Detection Statistics</div>', unsafe_allow_html=True)
table_placeholder = st.empty()
table_placeholder.markdown("""
<table class="styled-table">
    <thead>
        <tr>
            <th>Object Class</th>
            <th>Count</th>
            <th>Tracking IDs</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan="3" style="text-align: center; color: #94A3B8;">Start stream to observe live detected classes.</td>
        </tr>
    </tbody>
</table>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Performance Section
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">System Performance</div>', unsafe_allow_html=True)
perf_placeholder = st.empty()
perf_placeholder.markdown(f"""
<div class="info-bar">
    <div>Average FPS: <strong>0.0 FPS</strong></div>
    <div>YOLO Latency: <strong>0.0 ms</strong></div>
    <div>Processing Resolution: <strong>{res_choice}</strong></div>
    <div>Current Model: <strong>{model_selection.split('(')[0].strip()}</strong></div>
    <div>Compute Device: <strong>{device_name}</strong></div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Button Event Handling
# -----------------------------------------------------------------------------
if btn_start:
    st.session_state.is_running = True
    st.session_state.is_paused = False
    st.session_state.tracker.reset()
    st.session_state.perf_monitor.reset()
    st.session_state.frame_counter = 0
    st.rerun()

if btn_pause:
    st.session_state.is_paused = not st.session_state.is_paused
    st.rerun()

if btn_stop:
    st.session_state.is_running = False
    st.session_state.is_paused = False
    st.rerun()


# -----------------------------------------------------------------------------
# Real-Time Processing Loop
# -----------------------------------------------------------------------------
if st.session_state.is_running:
    # 1. Initialize detector for selected model variant (cached singleton)
    try:
        detector = get_cached_detector(selected_model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.session_state.is_running = False
        st.stop()

    # 2. Open Video Stream
    cap = None
    try:
        if source_type == "Webcam":
            cap = cv2.VideoCapture(int(webcam_index), cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if not cap.isOpened():
                cap = cv2.VideoCapture(int(webcam_index))
        else:
            if not video_path or not os.path.exists(video_path):
                st.error("Invalid video path. Please select or upload a valid file.")
                st.session_state.is_running = False
                st.stop()
            cap = cv2.VideoCapture(video_path)

        if not cap or not cap.isOpened():
            st.error("Failed to open video source. Ensure camera or file is available.")
            st.session_state.is_running = False
            st.stop()

        # Update tracker parameters
        st.session_state.tracker.sort.max_age = max_age
        st.session_state.tracker.sort.min_hits = min_hits
        st.session_state.tracker.sort.iou_threshold = iou_thresh

        while st.session_state.is_running:
            if st.session_state.is_paused:
                time.sleep(0.06)
                continue

            st.session_state.perf_monitor.start_frame()
            ret, frame = cap.read()

            if not ret or frame is None:
                st.info("Video stream completed.")
                st.session_state.is_running = False
                break

            st.session_state.frame_counter += 1
            frame_idx = st.session_state.frame_counter

            # Aspect-ratio preserved resizing (or native full frame if target_width <= 0)
            processed_frame = resize_frame(frame, target_width=target_width)

            # Determine detection vs Kalman tracking frame
            is_detection_frame = (frame_idx % frame_skip == 0) or (frame_idx == 1)

            inference_ms = 0.0
            t_track_start = time.perf_counter()

            if is_detection_frame:
                # Full-frame YOLO inference
                detections, inference_ms = detector.detect(
                    frame=processed_frame,
                    conf_threshold=conf_threshold,
                    iou_threshold=DEFAULT_NMS_IOU_THRESHOLD,
                    img_size=img_size,
                    selected_classes=selected_classes
                )
                # Class-aware SORT update
                tracked_objects = st.session_state.tracker.process_frame(
                    detections=detections,
                    is_detection_frame=True,
                    class_names_map=detector.names
                )
            else:
                # Kalman motion projection across skipped frame
                tracked_objects = st.session_state.tracker.process_frame(
                    detections=np.empty((0, 6)),
                    is_detection_frame=False,
                    class_names_map=detector.names
                )

            t_track_end = time.perf_counter()
            tracking_ms = (t_track_end - t_track_start) * 1000.0

            # Record metrics
            current_fps = st.session_state.perf_monitor.end_frame(
                inference_time_ms=inference_ms,
                tracking_time_ms=tracking_ms,
                was_detected=is_detection_frame
            )

            # Render bounding boxes and visual indicators
            annotated_frame = draw_bounding_boxes(
                processed_frame,
                tracked_objects=tracked_objects,
                show_trajectories=show_motion_trails
            )

            if show_hud_banner:
                annotated_frame = draw_tracking_hud(
                    annotated_frame,
                    fps=st.session_state.perf_monitor.current_fps,
                    active_tracks=len(tracked_objects),
                    total_unique=st.session_state.tracker.total_unique_objects,
                    is_detection_frame=is_detection_frame,
                    inference_ms=inference_ms
                )

            # Convert BGR to RGB and render in center container
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Update Metric Cards & Tables every 2 frames
            if frame_idx % 2 == 0:
                fps_val = f"{st.session_state.perf_monitor.current_fps:.1f}"
                tracks_val = f"{len(tracked_objects)}"
                unique_val = f"{st.session_state.tracker.total_unique_objects}"
                lat_val = f"{inference_ms:.1f} ms"

                c1_holder.markdown(f"""
                <div class="metric-card-box">
                    <div class="metric-header"><span class="metric-title">LIVE FPS</span><span class="metric-icon">⚡</span></div>
                    <div class="metric-value">{fps_val}</div>
                    <div class="metric-subtitle">Real-time processing rate</div>
                </div>
                """, unsafe_allow_html=True)

                c2_holder.markdown(f"""
                <div class="metric-card-box">
                    <div class="metric-header"><span class="metric-title">ACTIVE TRACKS</span><span class="metric-icon">🎯</span></div>
                    <div class="metric-value">{tracks_val}</div>
                    <div class="metric-subtitle">Currently in field of view</div>
                </div>
                """, unsafe_allow_html=True)

                c3_holder.markdown(f"""
                <div class="metric-card-box">
                    <div class="metric-header"><span class="metric-title">UNIQUE OBJECTS</span><span class="metric-icon">🔢</span></div>
                    <div class="metric-value">{unique_val}</div>
                    <div class="metric-subtitle">Session cumulative count</div>
                </div>
                """, unsafe_allow_html=True)

                c4_holder.markdown(f"""
                <div class="metric-card-box">
                    <div class="metric-header"><span class="metric-title">INFERENCE LATENCY</span><span class="metric-icon">⏱️</span></div>
                    <div class="metric-value">{lat_val}</div>
                    <div class="metric-subtitle">Neural network forward pass</div>
                </div>
                """, unsafe_allow_html=True)

                # Detection Statistics Table: Object Class | Count | Tracking IDs
                class_map = {}
                for obj in tracked_objects:
                    cname = obj.class_name.upper()
                    if cname not in class_map:
                        class_map[cname] = []
                    class_map[cname].append(f"#{obj.track_id:02d}")

                if class_map:
                    rows_html = ""
                    for cname, id_list in sorted(class_map.items()):
                        ids_str = ", ".join(id_list)
                        rows_html += f"<tr><td><strong>{cname}</strong></td><td>{len(id_list)}</td><td>{ids_str}</td></tr>"

                    table_placeholder.markdown(f"""
                    <table class="styled-table">
                        <thead>
                            <tr>
                                <th>Object Class</th>
                                <th>Count</th>
                                <th>Tracking IDs</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                    """, unsafe_allow_html=True)
                else:
                    table_placeholder.markdown("""
                    <table class="styled-table">
                        <thead>
                            <tr>
                                <th>Object Class</th>
                                <th>Count</th>
                                <th>Tracking IDs</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td colspan="3" style="text-align: center; color: #94A3B8;">No objects detected in current frame.</td>
                            </tr>
                        </tbody>
                    </table>
                    """, unsafe_allow_html=True)

                # Update Performance Section
                res_label = f"{processed_frame.shape[1]}x{processed_frame.shape[0]}"
                perf_placeholder.markdown(f"""
                <div class="info-bar">
                    <div>Average FPS: <strong>{fps_val} FPS</strong></div>
                    <div>YOLO Latency: <strong>{lat_val}</strong></div>
                    <div>Processing Resolution: <strong>{res_label}</strong></div>
                    <div>Current Model: <strong>{model_selection.split('(')[0].strip()}</strong></div>
                    <div>Compute Device: <strong>{device_name}</strong></div>
                </div>
                """, unsafe_allow_html=True)

            # Prevent thread lock & allow instant button response
            time.sleep(0.001)

    except Exception as ex:
        st.error(f"Runtime processing error: {ex}")
        st.session_state.is_running = False

    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            print("[App] VideoCapture released.")
