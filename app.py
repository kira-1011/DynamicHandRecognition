import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import av
import threading
import queue
import mediapipe as mp
import cupy as cp

# Page config
st.set_page_config(
    page_title="🤚 Hand Gesture Recognition",
    page_icon="🤚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🤚 Real-Time Hand Gesture Recognition")
st.markdown("**AI-powered gesture recognition using LSTM and MediaPipe**")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Model settings
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
buffer_size = st.sidebar.slider("Frame Buffer Size", 10, 50, 30, 5)
process_every_n_frames = st.sidebar.slider("Process Every N Frames", 1, 5, 2, 1)

# Gesture class mappings (adjust these based on your actual classes)
gesture_names = {
    0: "No Gesture",
    10: "Slide Down", 
    11: "Slide Left",
    12: "Slide Right",
    13: "Slide Up", 
    14: "Stop Sign"
}

# Add the missing functions from your notebook
def build_landmarker():
    """Build MediaPipe hand landmarker"""
    try:
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Download the model file if it doesn't exist
        import os
        import urllib.request
        
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            st.info("Downloading MediaPipe hand landmark model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            st.success("Model downloaded!")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )

        return HandLandmarker.create_from_options(options)
    except Exception as e:
        st.error(f"Error building landmarker: {e}")
        return None

def landmarks_to_vector(landmarks):
    """Convert landmarks to vector"""
    try:
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    except:
        return np.zeros(63, dtype=np.float64)

def frame_to_landmark_vec_gpu(bgr_image, landmark_detector):
    """Extract hand landmarks from frame using GPU acceleration"""
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect hand landmarks
        detection_result = landmark_detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            # Get first hand landmarks
            landmarks = detection_result.hand_landmarks[0]
            
            # Convert to vector
            vec = landmarks_to_vector(landmarks)
            
            # Try GPU acceleration if available
            try:
                gpu_vec = cp.asarray(vec, dtype=cp.float64)
                return gpu_vec
            except:
                # Fallback to CPU
                return vec.astype(np.float64)
        else:
            # Return zeros if no hand detected
            try:
                return cp.zeros(63, dtype=cp.float64)
            except:
                return np.zeros(63, dtype=np.float64)
                
    except Exception as e:
        # Return zeros on error
        try:
            return cp.zeros(63, dtype=cp.float64)
        except:
            return np.zeros(63, dtype=np.float64)

# Load model and mappings
@st.cache_resource
def load_model_and_mappings():
    """Load the trained model and label mappings"""
    try:
        # Load your saved model
        model = tf.keras.models.load_model('gesture_lstm_model.keras')
        
        # Create label mappings (adjust based on your training)
        unique_labels = [0, 10, 11, 12, 13, 14]  # Your actual labels
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        
        st.success("✅ Model loaded successfully!")
        return model, reverse_mapping, label_mapping
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

# Load MediaPipe detector
@st.cache_resource
def load_detector():
    """Load MediaPipe hand landmark detector"""
    try:
        detector = build_landmarker()
        if detector:
            st.success("✅ MediaPipe detector loaded!")
        return detector
    except Exception as e:
        st.error(f"❌ Error loading detector: {e}")
        return None

# Video transformer class for streamlit-webrtc
class GestureRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.detector = None
        self.reverse_mapping = None
        self.frame_buffer = deque(maxlen=30)
        self.frame_count = 0
        self.current_prediction = "No Gesture"
        self.current_confidence = 0.0
        self.prediction_buffer = deque(maxlen=5)
        
    def set_model_and_detector(self, model, detector, reverse_mapping):
        self.model = model
        self.detector = detector
        self.reverse_mapping = reverse_mapping
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.model is None or self.detector is None:
            # Just return the frame if model not loaded
            cv2.putText(img, "Loading model...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img
        
        self.frame_count += 1
        
        # Process every N frames for speed
        if self.frame_count % process_every_n_frames == 0:
            try:
                # Extract landmarks using your existing function
                landmark_vec = frame_to_landmark_vec_gpu(img, self.detector)
                
                if landmark_vec is not None:
                    # Convert CuPy to NumPy if needed
                    if hasattr(landmark_vec, 'get'):
                        landmark_vec = landmark_vec.get()
                    
                    self.frame_buffer.append(landmark_vec)
                    
                    # Predict when we have enough frames
                    if len(self.frame_buffer) >= 15:  # Minimum frames
                        # Pad sequence to required length
                        current_sequence = list(self.frame_buffer)
                        while len(current_sequence) < 30:
                            current_sequence.insert(0, np.zeros(63, dtype=np.float32))
                        
                        sequence = np.expand_dims(np.array(current_sequence[:30], dtype=np.float32), axis=0)
                        
                        # Make prediction
                        predictions = self.model(sequence, training=False)
                        predicted_class = tf.argmax(predictions[0]).numpy()
                        confidence = tf.reduce_max(predictions[0]).numpy()
                        
                        # Add to prediction buffer for smoothing
                        self.prediction_buffer.append((predicted_class, confidence))
                        
                        # Get smoothed prediction
                        if confidence > confidence_threshold:
                            recent_classes = [p[0] for p in self.prediction_buffer if p[1] > confidence_threshold]
                            if recent_classes:
                                most_common = max(set(recent_classes), key=recent_classes.count)
                                avg_confidence = np.mean([p[1] for p in self.prediction_buffer if p[0] == most_common])
                                
                                original_label = self.reverse_mapping[most_common]
                                self.current_prediction = gesture_names.get(original_label, f"Gesture_{original_label}")
                                self.current_confidence = avg_confidence
                
            except Exception as e:
                pass  # Ignore errors in real-time processing
        
        # Draw predictions on frame
        cv2.putText(img, f"Gesture: {self.current_prediction}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Confidence: {self.current_confidence:.3f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Buffer: {len(self.frame_buffer)}/{buffer_size}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return img

# Simplified camera input version (more reliable)
def camera_input_app():
    st.subheader("📷 Camera Input Approach")
    
    # Load model and detector
    model, reverse_mapping, label_mapping = load_model_and_mappings()
    detector = load_detector()
    
    if model is None or detector is None:
        return
    
    # Camera input
    camera_input = st.camera_input("Take a picture of your gesture")
    
    if camera_input is not None:
        # Convert to OpenCV format
        bytes_data = camera_input.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Process the image
        try:
            landmark_vec = frame_to_landmark_vec_gpu(cv2_img, detector)
            
            if landmark_vec is not None:
                if hasattr(landmark_vec, 'get'):
                    landmark_vec = landmark_vec.get()
                
                # Create a sequence (repeat the single frame)
                sequence = np.tile(landmark_vec, (30, 1))
                sequence = np.expand_dims(sequence, axis=0)
                
                # Make prediction
                predictions = model(sequence, training=False)
                predicted_class = tf.argmax(predictions[0]).numpy()
                confidence = tf.reduce_max(predictions[0]).numpy()
                
                original_label = reverse_mapping[predicted_class]
                gesture_name = gesture_names.get(original_label, f"Gesture_{original_label}")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(camera_input, caption="Captured Image")
                
                with col2:
                    st.subheader("Prediction Results")
                    if confidence > confidence_threshold:
                        st.success(f"🤚 **{gesture_name}**")
                        st.metric("Confidence", f"{confidence:.3f}")
                    else:
                        st.warning("Low confidence prediction")
                        st.metric("Confidence", f"{confidence:.3f}")
                        
                    # Show all predictions
                    st.subheader("📊 All Predictions")
                    pred_data = []
                    for i, pred in enumerate(predictions[0].numpy()):
                        original_label = reverse_mapping[i]
                        gesture_name = gesture_names.get(original_label, f"Gesture_{original_label}")
                        pred_data.append({"Gesture": gesture_name, "Confidence": pred})
                    
                    pred_df = pd.DataFrame(pred_data).sort_values('Confidence', ascending=False)
                    st.dataframe(pred_df, use_container_width=True)
            else:
                st.error("No hand landmarks detected!")
                
        except Exception as e:
            st.error(f"Error processing image: {e}")

# Main function with WebRTC
def main():
    # Load model and detector
    model, reverse_mapping, label_mapping = load_model_and_mappings()
    detector = load_detector()
    
    if model is None or detector is None:
        st.error("Cannot proceed without model and detector. Please check the error messages above.")
        return
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        
        try:
            # WebRTC configuration
            RTC_CONFIGURATION = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            })
            
            # Create transformer
            transformer = GestureRecognitionTransformer()
            transformer.set_model_and_detector(model, detector, reverse_mapping)
            
            # WebRTC streamer
            webrtc_ctx = webrtc_streamer(
                key="gesture-recognition",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_transformer_factory=lambda: transformer,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
        except Exception as e:
            st.error(f"WebRTC Error: {e}")
            st.info("WebRTC might not work in all environments. Try the Camera Input mode instead.")
    
    with col2:
        st.subheader("📊 Prediction Results")
        st.info("Real-time predictions will appear here when using video mode")
        
        # Gesture legend
        st.subheader("🎯 Gesture Classes")
        for key, value in gesture_names.items():
            st.write(f"**{key}:** {value}")

# App selection
app_mode = st.sidebar.selectbox(
    "Choose App Mode",
    ["Camera Input", "Real-time Video"]  # Changed order to make Camera Input default
)

if app_mode == "Real-time Video":
    main()
else:
    camera_input_app()

# Instructions
with st.sidebar:
    st.markdown("---")
    st.subheader("📖 Instructions")
    st.markdown("""
    1. **Choose Camera Input mode** for best compatibility
    2. **Allow camera access** when prompted
    3. **Position your hand** clearly in the camera
    4. **Make distinct gestures** for best results
    5. **Adjust confidence threshold** if needed
    """)
    
    st.markdown("---")
    st.subheader("⚡ Performance Tips")
    st.markdown("""
    - **Good lighting** improves detection
    - **Clear background** works better
    - **Single hand** in frame
    - **Hold gesture** for 1-2 seconds
    """)

# Footer
st.markdown("---")
st.markdown("**Built with Streamlit, TensorFlow, and MediaPipe** 🚀")
