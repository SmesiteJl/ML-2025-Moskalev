import cv2
import mediapipe as mp
from fer import FER
import threading

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
hand_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
emotion_detector = FER(mtcnn=True)

current_emotion = "..."
emotion_lock = threading.Lock()

def analyze_emotion(face_img):
    global current_emotion
    try:
        results = emotion_detector.detect_emotions(cv2.resize(face_img, (224, 224)))
        if results:
            emotions = results[0]["emotions"]
            dominant = max(emotions, key=emotions.get)
            with emotion_lock:
                current_emotion = dominant
    except:
        with emotion_lock:
            current_emotion = "N/A"

def count_fingers(hand_landmarks):
    ids_tip = [4, 8, 12, 16, 20]
    ids_joint = [3, 6, 10, 14, 18]
    count = 0

    if hand_landmarks[ids_tip[0]].x < hand_landmarks[ids_joint[0]].x:
        count += 1

    for i in range(1, 5):
        if hand_landmarks[ids_tip[i]].y < hand_landmarks[ids_joint[i]].y:
            count += 1

    return count

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    face_results = face_detector.process(rgb)
    hand_results = hand_detector.process(rgb)

    fingers = 0
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2)
            )
            fingers = count_fingers(hand_landmarks.landmark)

    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            box_w, box_h = int(bbox.width * w), int(bbox.height * h)
            x, y = max(0, x), max(0, y)
            face_crop = frame[y:y+box_h, x:x+box_w]

            if fingers == 1:
                label = "Nikita"
            elif fingers == 2:
                label = "Moskalev"
            elif fingers == 3:
                threading.Thread(target=analyze_emotion, args=(face_crop,), daemon=True).start()
                with emotion_lock:
                    label = f"Mood: {current_emotion}"
            else:
                label = "Verified"

            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (255, 100, 50), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
