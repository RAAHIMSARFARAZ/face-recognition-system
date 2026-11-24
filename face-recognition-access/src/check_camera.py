import cv2
import requests
import numpy as np
import time
import os

# ============================================================
# ESP32-CAM
# ============================================================

ESP32_IP = "192.168.0.115"   # set your board IP
URL = f"http://{ESP32_IP}/capture"


# ============================================================
# Models
# ============================================================

YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "models/face_recognition_sface_2021dec.onnx"


# ============================================================
# Check model files
# ============================================================

if not os.path.exists(YUNET_MODEL):
    print("ERROR: YuNet model not found:")
    print(YUNET_MODEL)
    input("Press Enter to exit...")
    raise SystemExit

if not os.path.exists(SFACE_MODEL):
    print("ERROR: SFace model not found:")
    print(SFACE_MODEL)
    input("Press Enter to exit...")
    raise SystemExit


# ============================================================
# Load YuNet
# ============================================================

print("Loading YuNet...")

detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 240),
    0.7,
    0.3,
    5000
)

print("YuNet loaded.")


# ============================================================
# Load SFace
# ============================================================

print("Loading SFace...")

recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL,
    ""
)

print("SFace loaded.")
print()
print("Everything loaded successfully!")
print("Starting camera...")
print("Press Q to quit.")


# ============================================================
# Main loop
# ============================================================

while True:

    try:

        # ----------------------------------------------------
        # Get image from ESP32
        # ----------------------------------------------------

        response = requests.get(
            URL,
            timeout=3
        )

        if response.status_code != 200:
            print(
                "HTTP error:",
                response.status_code
            )
            continue


        # ----------------------------------------------------
        # JPEG → OpenCV image
        # ----------------------------------------------------

        data = np.frombuffer(
            response.content,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            data,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            print("Failed to decode image")
            continue


        # ----------------------------------------------------
        # Tell YuNet the image size
        # ----------------------------------------------------

        height, width = frame.shape[:2]

        detector.setInputSize(
            (width, height)
        )


        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        _, faces = detector.detect(frame)

        face_count = 0

        if faces is not None:

            face_count = len(faces)

            for face in faces:

                # ============================================
                # Face bounding box
                # ============================================

                x, y, w, h = face[:4].astype(int)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )


                # ============================================
                # Face confidence
                # ============================================

                confidence = face[-1]

                cv2.putText(
                    frame,
                    f"Face {confidence:.2f}",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )


                # ============================================
                # Align face for SFace
                # ============================================

                aligned_face = recognizer.alignCrop(
                    frame,
                    face
                )


                # ============================================
                # Generate face feature
                # ============================================

                feature = recognizer.feature(
                    aligned_face
                )


                # ============================================
                # Display feature information
                # ============================================

                cv2.putText(
                    frame,
                    "SFace OK",
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )


        # ====================================================
        # Face count
        # ====================================================

        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ====================================================
        # Display
        # ====================================================

        cv2.imshow(
            "ESP32-CAM - YuNet + SFace",
            frame
        )


        # ====================================================
        # Quit
        # ====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


        # About 5 FPS
        time.sleep(0.2)


    except requests.RequestException as e:

        print(
            "Connection error:",
            e
        )

        time.sleep(1)


    except Exception as e:

        print(
            "Error:",
            e
        )

        time.sleep(1)


cv2.destroyAllWindows()
