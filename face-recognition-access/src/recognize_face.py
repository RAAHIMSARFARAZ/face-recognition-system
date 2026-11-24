
import cv2
import numpy as np
import requests
import sqlite3
import os
import time


# ============================================================
# CONFIGURATION
# ============================================================

ESP32_IP = "192.x.x.x" #Webserver IP goes here.
CAPTURE_URL = f"http://{ESP32_IP}/capture"

DATABASE_FILE = "database/access.db"
EMBEDDINGS_DIR = "database/embeddings"

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"

CONFIDENCE_THRESHOLD = 0.45

FRAME_WIDTH = 240
FRAME_HEIGHT = 176

MIN_DETECTION_SCORE = 0.80
MIN_FACE_SIZE = 40


# ============================================================
# ESP32-CAM GPIO 4 FLASH LED CONTROL
# ============================================================

# None means the Python program has not synchronized LED state yet.
LED_STATE = None


def set_led(state):
    global LED_STATE

    # Avoid repeatedly sending the same successful command.
    if state == LED_STATE:
        return

    endpoint = "on" if state else "off"
    url = f"http://{ESP32_IP}/led/{endpoint}"

    try:
        response = requests.get(url, timeout=1)
        response.raise_for_status()

        LED_STATE = state
        print(f"Flash LED: {'ON' if state else 'OFF'}")

    except requests.exceptions.RequestException as error:
        print(f"LED control error: {error}")


# ============================================================
# CHECK DATABASE AND LOAD EMPLOYEE RECORDS
# ============================================================

if not os.path.exists(DATABASE_FILE):
    print("Database does not exist.")
    print("Run setup_database.py and enroll.py first.")
    raise SystemExit(1)

with sqlite3.connect(DATABASE_FILE) as connection:
    employees = connection.execute(
        """
        SELECT employee_id, name, job_title, authorized
        FROM employees
        """
    ).fetchall()

if not employees:
    print("Database is empty.")
    print("Run enroll.py to register an employee.")
    raise SystemExit(1)

print()
print("=" * 50)
print("FACE ACCESS CONTROL")
print("=" * 50)
print()
print("Registered employees:")

for employee_id, name, job_title, authorized in employees:
    status = "AUTHORIZED" if authorized else "UNAUTHORIZED"
    print(f"  {employee_id} - {name} ({job_title}): {status}")

print()
print(f"Cosine similarity threshold: {CONFIDENCE_THRESHOLD}")
print()


# ============================================================
# LOAD FACE EMBEDDINGS
# ============================================================

known_people = {}

for employee_id, name, job_title, authorized in employees:
    embedding_file = os.path.join(
        EMBEDDINGS_DIR,
        f"{employee_id}.npy"
    )

    if not os.path.exists(embedding_file):
        print(
            f"WARNING: Embeddings missing for "
            f"{name} ({employee_id})"
        )
        continue

    try:
        embeddings = np.load(embedding_file)

        if embeddings.size == 0:
            print(
                f"WARNING: Empty embeddings for "
                f"{name} ({employee_id})"
            )
            continue

        known_people[employee_id] = {
            "name": name,
            "job_title": job_title,
            "authorized": bool(authorized),
            "embeddings": embeddings
        }

        print(f"Loaded embeddings: {name} ({employee_id})")

    except (OSError, ValueError) as error:
        print(
            f"WARNING: Could not load embeddings "
            f"for {name}: {error}"
        )

if not known_people:
    print()
    print("No usable face embeddings were found.")
    print("Check database/embeddings and enroll employees again.")
    raise SystemExit(1)

print()
print(f"Loaded {len(known_people)} employees with face embeddings.")
print()


# ============================================================
# LOAD FACE DETECTION AND RECOGNITION MODELS
# ============================================================

if not os.path.exists(DETECTOR_MODEL):
    print(f"Face detection model not found: {DETECTOR_MODEL}")
    raise SystemExit(1)

if not os.path.exists(RECOGNIZER_MODEL):
    print(f"Face recognition model not found: {RECOGNIZER_MODEL}")
    raise SystemExit(1)

detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL,
    "",
    (FRAME_WIDTH, FRAME_HEIGHT),
    MIN_DETECTION_SCORE,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNIZER_MODEL,
    ""
)

print("Face detection model loaded.")
print("Face recognition model loaded.")
print("Connecting to ESP32-CAM...")
print("Press Q in the camera window to quit.")
print()


# ============================================================
# IDENTIFY A FACE
# ============================================================

def identify_face(feature):
    best_employee_id = None
    best_name = "Unknown"
    best_job_title = ""
    best_score = -1.0
    best_authorized = False

    # Compare the detected face against every saved embedding.
    for employee_id, info in known_people.items():
        for known_embedding in info["embeddings"]:
            score = recognizer.match(
                feature,
                known_embedding,
                cv2.FaceRecognizerSF_FR_COSINE
            )

            if score > best_score:
                best_score = float(score)
                best_employee_id = employee_id
                best_name = info["name"]
                best_job_title = info["job_title"]
                best_authorized = info["authorized"]

    # Accept the best match only if it reaches the threshold.
    if best_score >= CONFIDENCE_THRESHOLD:
        return (
            best_employee_id,
            best_name,
            best_job_title,
            best_score,
            best_authorized
        )

    # A low similarity score is not a confirmed identity.
    return (
        None,
        "Unknown",
        "",
        best_score,
        False
    )


# ============================================================
# MAIN CAMERA AND RECOGNITION LOOP
# ============================================================

try:
    while True:
        try:
            # Fetch a JPEG image from the ESP32-CAM.
            response = requests.get(
                CAPTURE_URL,
                timeout=5
            )
            response.raise_for_status()

            image_data = np.frombuffer(
                response.content,
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                image_data,
                cv2.IMREAD_COLOR
            )

            if frame is None:
                print("Could not decode the camera image.")
                set_led(False)
                time.sleep(0.5)
                continue

            frame_height, frame_width = frame.shape[:2]

            # Set YuNet to the actual camera image dimensions.
            detector.setInputSize(
                (frame_width, frame_height)
            )

            # Detect faces in the image.
            _, faces = detector.detect(frame)

            # If there are no detected faces, turn off the LED.
            if faces is None or len(faces) == 0:
                set_led(False)

                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 165, 255),
                    1
                )

                cv2.imshow(
                    "ESP32 Face Access Control",
                    frame
                )

                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break

                continue

            # Track whether any face in this frame is recognized.
            recognized_face_found = False

            # Process every detected face.
            for face in faces:
                x, y, w, h = face[:4].astype(int)

                # Ignore detections that are too small.
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    continue

                # Align and crop the face before recognition.
                aligned_face = recognizer.alignCrop(
                    frame,
                    face
                )

                feature = recognizer.feature(aligned_face)

                (
                    employee_id,
                    name,
                    job_title,
                    score,
                    authorized
                ) = identify_face(feature)

                # Any matched identity counts as recognized.
                if name != "Unknown":
                    recognized_face_found = True

                # Decide the access status separately from identity.
                if name == "Unknown":
                    access_status = "UNKNOWN PERSON"
                    color = (0, 0, 255)

                elif authorized:
                    access_status = "ACCESS GRANTED"
                    color = (0, 255, 0)

                else:
                    access_status = "ACCESS DENIED"
                    color = (0, 0, 255)

                # Print recognition results in the terminal.
                if name == "Unknown":
                    print(
                        f"Unknown person | "
                        f"Best similarity: {score:.3f}"
                    )
                else:
                    print(
                        f"{name} | ID: {employee_id} | "
                        f"Job: {job_title} | "
                        f"Similarity: {score:.3f} | "
                        f"{access_status}"
                    )

                # Keep the face rectangle inside the image.
                x = max(0, x)
                y = max(0, y)

                box_w = min(w, frame_width - x)
                box_h = min(h, frame_height - y)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + box_w, y + box_h),
                    color,
                    2
                )

                # Display identity above the face.
                if name == "Unknown":
                    identity_label = f"Unknown | {score:.2f}"
                else:
                    identity_label = (
                        f"{name} ({employee_id}) | {score:.2f}"
                    )

                cv2.putText(
                    frame,
                    identity_label,
                    (x, max(14, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1
                )

                # Display job title or access status below the face.
                detail_label = job_title if job_title else access_status

                detail_y = min(
                    frame_height - 20,
                    y + box_h + 14
                )

                cv2.putText(
                    frame,
                    detail_label,
                    (x, max(12, detail_y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1
                )

                status_y = min(
                    frame_height - 5,
                    max(12, detail_y + 13)
                )

                cv2.putText(
                    frame,
                    access_status,
                    (x, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1
                )

            # Update the LED once per frame rather than once per face.
            # It remains on if at least one identity was recognized.
            set_led(recognized_face_found)

            # Display the processed camera image.
            cv2.imshow(
                "ESP32 Face Access Control",
                frame
            )

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

        except requests.exceptions.RequestException as error:
            print(f"ESP32-CAM connection error: {error}")
            set_led(False)
            time.sleep(1)

        except (cv2.error, ValueError, TypeError) as error:
            print(f"Image processing error: {error}")
            time.sleep(0.5)

except KeyboardInterrupt:
    print("\nRecognition stopped by user.")

finally:
    # Attempt to turn the flash LED off when the program exits.
    set_led(False)
    cv2.destroyAllWindows()
    print("Face recognition stopped.")
