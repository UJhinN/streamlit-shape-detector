import streamlit as st
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import time
import os

# --- Constants ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
URL_READ_TIMEOUT = 5  # seconds

# --- Function Definitions ---

def detect_shapes(frame, threshold_val, min_area):
    """
    Processes a single video frame to detect basic shapes and draw contours.
    
    Args:
        frame: The input video frame (BGR).
        threshold_val: The threshold value for binarization.
        min_area: The minimum area to consider a contour.

    Returns:
        A tuple containing:
        - original_frame_rgb: The resized original frame (RGB).
        - processed_frame_rgb: The frame with detected shapes and labels (RGB).
        - shape_counts: A dictionary containing the count of each detected shape.
    """
    resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    gray_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Binarize the image using thresholding
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    processed_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    shape_counts = {"Triangle": 0, "Square": 0, "Circle": 0}
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            shape = "Unknown"
            
            if len(approx) == 3:
                shape = "Triangle"
                shape_counts["Triangle"] += 1
            elif len(approx) == 4:
                # Check aspect ratio to distinguish squares from rectangles
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                    shape = "Square"
                    shape_counts["Square"] += 1
                else:
                    shape = "Rectangle"
            else:
                # Check if the shape is a circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                circle_area = np.pi * (radius ** 2)
                if abs(area - circle_area) / area < 0.2:
                    shape = "Circle"
                    shape_counts["Circle"] += 1
            
            # Draw contour and text on the frame
            cv2.drawContours(processed_frame_rgb, [approx], 0, (0, 255, 0), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(processed_frame_rgb, shape, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    original_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    return original_frame_rgb, processed_frame_rgb, shape_counts

def open_video_capture(source, file=None, url=None):
    """Opens a video capture object based on the selected source."""
    if source == "upload":
        if file is None:
            return None
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        return cv2.VideoCapture(tfile.name)
    elif source == "webcam":
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    elif source == "url":
        if not url:
            return None
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    return None

# --- Main App ---

# --- UI Styling ---
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .st-emotion-cache-18e3th9 {padding: 2rem 1rem 1rem 1rem;}
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<h1 style='color:#B388FF; text-align:center;'>Real-time Shape Detector</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color:gray; text-align:center;'>Using OpenCV and Streamlit</h4>", unsafe_allow_html=True)

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("## Adjust Shape Detection")
    threshold_val = st.slider("Threshold Value", 0, 255, 128)
    min_area = st.slider("Minimum Object Area", 0, 5000, 1500)

# --- Video Source Selection ---
st.markdown("#### Select Video Source")
cols = st.columns(3)
if cols[0].button("Upload Video", key="btn_upload"): st.session_state.video_source = "upload"
if cols[1].button("Webcam", key="btn_cam"): st.session_state.video_source = "webcam"
if cols[2].button("Stream URL", key="btn_url"): st.session_state.video_source = "url"

video_source = st.session_state.get("video_source", None)
uploaded_file, stream_url = None, None

if video_source == "upload":
    uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "avi", "mov"])
elif video_source == "url":
    stream_url = st.text_input("Enter the stream URL", placeholder="https://... or rtsp://...")

# --- Video Processing and Display ---
cap = open_video_capture(video_source, file=uploaded_file, url=stream_url)

if cap and cap.isOpened():
    placeholder_cols = st.columns(2)
    placeholder_orig = placeholder_cols[0].empty()
    placeholder_detect = placeholder_cols[1].empty()
    placeholder_graph = st.empty()

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1 / fps if fps > 0 else 0.03

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        
        if time.time() - start_time > URL_READ_TIMEOUT and video_source == 'url':
            st.error("Stream connection timed out. 🌐")
            break
        if not ret:
            st.warning("Video has ended or cannot be read.")
            break

        original, processed, shape_counts = detect_shapes(frame, threshold_val, min_area)
        
        placeholder_orig.image(original, caption="Original Frame")
        placeholder_detect.image(processed, caption="Detected Shapes")
        # Create and display a bar chart of detected shapes
        fig, ax = plt.subplots(figsize=(6,3), facecolor='#0E1117')
        shapes = list(shape_counts.keys())
        counts = list(shape_counts.values())
        
        ax.bar(shapes, counts, color=['#FFA07A', '#98FB98', '#ADD8E6'])
        ax.set_facecolor('#0E1117')
        ax.tick_params(colors='white')
        ax.set_title('Detected Shape Counts', color='#B388FF')
        ax.set_xlabel('Shape', color='white')
        ax.set_ylabel('Count', color='white')
        
        placeholder_graph.pyplot(fig)
        plt.close(fig)

        time.sleep(delay)
    cap.release()
else:
    if video_source == "upload" and uploaded_file is None:
        st.info("Please upload a video file to start. 📁")
    elif video_source == "url" and (not stream_url or stream_url.strip() == ""):
        st.info("Please enter a stream URL. 🌐")
    elif video_source == "webcam" and not (cap and cap.isOpened()):
        st.error("Cannot access webcam. 📸 Please check if it's connected and not used by another app.")
    elif video_source and not (cap and cap.isOpened()):
         st.error(f"Error opening source: {video_source}. Please check the file or URL.")
    else:
        st.info("Please select a video source above. 👆")