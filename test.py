import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile

def load_model():
    model = YOLO("best.pt")  # Replace 'best.pt' with your YOLOv8 model file path
    return model

def process_frame(frame, model, confidence_threshold=0.75):
    results = model(frame)  # Run inference on the frame
    detections = results[0].boxes.data.cpu().numpy()  # Get detection results
    
    # Filter detections based on confidence threshold
    filtered_detections = [d for d in detections if d[4] >= confidence_threshold]  # d[4] is confidence score
    
    if filtered_detections:
        annotated_frame = results[0].plot()  # Annotate frame only if detections exceed threshold
        return annotated_frame, True
    else:
        return frame, False  # Return original frame if no detections exceed threshold

def main():
    st.title("Fall Detection - YOLOv8 Live Prediction")
    
    # Sidebar for selecting input source
    st.sidebar.header("Select Input Source")
    input_source = st.sidebar.radio("Choose an input source:", ("Upload Video", "Live Webcam"))

    # Upload video file option
    if input_source == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            model = load_model()
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()  # Placeholder for displaying frames

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, fall_detected = process_frame(frame, model)

                # Convert frame to RGB and display it
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                stframe.image(frame_pil, caption="Live Fall Detection" if fall_detected else "No Fall Detected")

            cap.release()
            st.success("Video processing completed!")
        else:
            st.warning("Please upload a video to start predictions.")

    elif input_source == "Live Webcam":
        model = load_model()
        cap = cv2.VideoCapture(0)  # 0 for default webcam
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("No webcam detected.")
                break

            annotated_frame, fall_detected = process_frame(frame, model)

            # Convert frame to RGB and display it
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            stframe.image(frame_pil, caption="Live Webcam Fall Detection" if fall_detected else "No Fall Detected")

        cap.release()
        st.success("Live webcam processing stopped.")

if __name__ == "__main__":
    main()
