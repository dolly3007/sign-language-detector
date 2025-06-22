import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load ASL model
@st.cache_resource
def load_asl_model():
    return load_model("best_model.keras")

model = load_asl_model()

# Class names (match training)
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Custom transformer class
class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        self.class_names = class_names
        self.model = model
        self.rect_size = 224

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")

        # Mirror the image horizontally
        img = cv2.flip(img, 1)

        h, w, _ = img.shape
        x, y = w // 2 - 112, h // 2 - 112

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + self.rect_size, y + self.rect_size), (0, 255, 0), 2)
        roi = img[y:y + self.rect_size, x:x + self.rect_size]

        if roi.shape[:2] == (self.rect_size, self.rect_size):
            resized = cv2.resize(roi, (128, 128))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_img = np.expand_dims(rgb, axis=0)
            prediction = self.model.predict(input_img)
            idx = int(np.argmax(prediction))
            conf = float(np.max(prediction))
            label = f"{self.class_names[idx]} ({conf:.2f})"
            cv2.putText(img, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return img

# Disable audio & display UI
st.title("ASL Sign Language Detection (WebRTC)")
webrtc_streamer(
    key="asl",
    video_processor_factory=ASLTransformer,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False},  # 🔇 Audio disabled
)
