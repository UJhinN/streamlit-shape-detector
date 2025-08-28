# Real-time Shape Detector

![Project_Screenshot](https://i.imgur.com/kYq3QWc.png)

This project is a real-time web application built with **Streamlit** and **OpenCV** that detects basic shapes (triangles, squares, and circles) from a live video feed. Users can customize detection parameters and view the results, including a real-time graph of the detected shapes.

---

### Key Features

* **Flexible Video Source:** Supports video from a webcam, an uploaded video file, or a direct streaming URL (e.g., RTSP).
* **Real-time Processing:** Detects and labels shapes in real time.
* **Customizable Parameters:** Users can adjust the **threshold** and **minimum object area** through an interactive sidebar.
* **Visual Output:** Displays both the original video feed and the processed feed with detected shapes.
* **Data Visualization:** Includes a **bar chart** that shows the count of each detected shape, updating with every frame.

---

### Prerequisites

Before running the application, make sure you have Python installed. Then, install the required libraries by running the following command:

```bash
pip install streamlit opencv-python matplotlib numpy