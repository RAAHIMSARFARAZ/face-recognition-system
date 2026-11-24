import cv2
import numpy as np
import requests
import sqlite3
import time
import re
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

ESP32_IP = "192.x.x.x" #Webserver IP goes here.
CAPTURE_URL = f"http://{ESP32_IP}/capture"

DETECTOR_MODEL = Path("models/face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = Path("models/face_recognition_sface_2021dec.onnx")

DB_PATH = Path("database/access.db")
EMBEDDINGS_DIR = Path("database/embeddings")

TARGET_SAMPLES = 25
MIN_FACE_SIZE = 40          # Minimum face width and height in pixels
MIN_DETECTION_SCORE = 0.80  # YuNet detection confidence
SAMPLE_INTERVAL = 0.6       # Seconds between accepted samples
FRAME_WIDTH = 240
FRAME_HEIGHT = 176


def main():
    print("=" * 55)
    print("EMPLOYEE FACE ENROLLMENT")
    print("=" * 55)

    # Check that the project files are where this script expects them.
    for model_path in (DETECTOR_MODEL, RECOGNIZER_MODEL):
        if not model_path.is_file():
            print(f"\nERROR: Model file not found: {model_path}")
            print("Run this script from your pc project folder and check models/.")
            return

    if not DB_PATH.parent.is_dir():
        print(f"\nERROR: Database folder not found: {DB_PATH.parent}")
        print("Run setup_database.py first.")
        return

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    employee_id = input("\nEmployee ID (e.g. EMP001): ").strip()
    if not employee_id or not re.fullmatch(r"[A-Za-z0-9_-]+", employee_id):
        print("ERROR: Use a non-empty ID containing only letters, numbers, _ or -.")
        return

    name = input("Full name: ").strip()
    if not name:
        print("ERROR: Name cannot be empty.")
        return

    job_title = input("Job title: ").strip()
    if not job_title:
        print("ERROR: Job title cannot be empty.")
        return

    while True:
        answer = input("Authorize this employee? (y/n): ").strip().lower()
        if answer in ("y", "n"):
            authorized = 1 if answer == "y" else 0
            break
        print("Please enter y or n.")

    # Ensure the employee ID is not already registered.
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                job_title TEXT NOT NULL,
                authorized INTEGER NOT NULL DEFAULT 0
                    CHECK (authorized IN (0, 1)),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        existing = connection.execute(
            "SELECT 1 FROM employees WHERE employee_id = ?",
            (employee_id,)
        ).fetchone()

    if existing:
        print(f"\nERROR: Employee ID {employee_id} is already registered.")
        print("No files or records were changed.")
        return

    embedding_path = EMBEDDINGS_DIR / f"{employee_id}.npy"
    if embedding_path.exists():
        print(f"\nERROR: An embedding file already exists: {embedding_path}")
        print("Rename or back up that file before enrolling this ID.")
        return

    print("\nLoading YuNet...")
    detector = cv2.FaceDetectorYN.create(
        str(DETECTOR_MODEL), "", (FRAME_WIDTH, FRAME_HEIGHT)
    )

    print("Loading SFace...")
    recognizer = cv2.FaceRecognizerSF.create(
        str(RECOGNIZER_MODEL), ""
    )

    samples = []
    last_sample_time = 0.0
    session = requests.Session()

    print(f"\nEnrollment will collect {TARGET_SAMPLES} valid samples.")
    print("Keep one face in view. Slowly vary your head angle and expression.")
    print("Press Q to cancel. The employee record is saved only after all samples are collected.\n")

    try:
        while len(samples) < TARGET_SAMPLES:
            try:
                response = session.get(CAPTURE_URL, timeout=5)
                response.raise_for_status()
            except requests.RequestException as exc:
                print(f"Camera connection error: {exc}")
                time.sleep(1)
                continue

            image_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if frame is None:
                print("Received an invalid image; trying again.")
                time.sleep(0.2)
                continue

            height, width = frame.shape[:2]
            detector.setInputSize((width, height))
            _, faces = detector.detect(frame)

            message = "No face detected"
            valid_face = None

            if faces is not None and len(faces) > 1:
                message = "Only one person should be in frame"
            elif faces is not None and len(faces) == 1:
                face = faces[0]
                x, y, w, h = face[:4].astype(int)
                score = float(face[14])

                if score < MIN_DETECTION_SCORE:
                    message = "Face detection confidence too low"
                elif w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    message = "Move closer to the camera"
                elif x < 2 or y < 2 or x + w > width - 2 or y + h > height - 2:
                    message = "Keep your whole face in frame"
                else:
                    valid_face = face
                    message = "Face looks usable"

            now = time.monotonic()
            if valid_face is not None and now - last_sample_time >= SAMPLE_INTERVAL:
                try:
                    aligned_face = recognizer.alignCrop(frame, valid_face)
                    feature = recognizer.feature(aligned_face)

                    if feature is not None and np.isfinite(feature).all():
                        samples.append(feature.copy())
                        last_sample_time = now
                        print(f"Collected sample {len(samples)}/{TARGET_SAMPLES}")
                    else:
                        message = "Invalid embedding; try again"
                except cv2.error:
                    message = "Could not align face; adjust your position"

            # Preview and progress display.
            if valid_face is not None:
                x, y, w, h = valid_face[:4].astype(int)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame, f"Samples: {len(samples)}/{TARGET_SAMPLES}",
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )
            cv2.putText(
                frame, message, (5, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
            )
            cv2.putText(
                frame, "Q = cancel", (5, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
            cv2.imshow("Employee Enrollment", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nEnrollment cancelled. No employee record was saved.")
                return

    finally:
        session.close()
        cv2.destroyAllWindows()

    embeddings = np.asarray(samples, dtype=np.float32)

    # Save embeddings first, then insert the employee record.
    # If the database insert fails, remove the newly created embedding file.
    try:
        np.save(embedding_path, embeddings)

        with sqlite3.connect(DB_PATH) as connection:
            connection.execute("""
                INSERT INTO employees
                    (employee_id, name, job_title, authorized)
                VALUES (?, ?, ?, ?)
            """, (employee_id, name, job_title, authorized))

    except Exception as exc:
        if embedding_path.exists():
            embedding_path.unlink()
        print(f"\nERROR: Could not save enrollment: {exc}")
        print("The new embedding file was removed if it had been created.")
        return

    print("\n" + "=" * 55)
    print("ENROLLMENT COMPLETE")
    print("=" * 55)
    print(f"Employee ID: {employee_id}")
    print(f"Name:        {name}")
    print(f"Job title:   {job_title}")
    print(f"Authorized:  {'Yes' if authorized else 'No'}")
    print(f"Samples:     {len(samples)}")
    print(f"Embeddings:  {embedding_path.as_posix()}")
    print(f"Database:    {DB_PATH.as_posix()}")
    print("\nNote: recognition code must be updated to read employees from SQLite.")


if __name__ == "__main__":
    main()
