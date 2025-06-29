import cv2
import os
from deepface import DeepFace
import pandas as pd
from datetime import datetime

# ====== CẤU HÌNH ======
dataset_path = "dataset/NguyenVanVung"
student_name = "Nguyễn Văn Vững"
model_name = "Facenet"
detector_backend = "opencv"

# ====== ĐỌC ẢNH MẪU ======
image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)
               if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

# ====== CSV ======
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(attendance_file, index=False)

# ====== KHỞI TẠO WEBCAM ======
cap = cv2.VideoCapture(0)
recognized = False

print("🟢 Đang chạy điểm danh - Nhấn Q để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        if not recognized:
            for img_path in image_paths:
                result = DeepFace.verify(
                    frame,
                    img_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=False
                )

                if result["verified"]:
                    print(f"✅ {student_name} đã điểm danh!")
                    recognized = True

                    # Ghi log điểm danh
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    df = pd.read_csv(attendance_file)
                    df.loc[len(df)] = [student_name, now]
                    df.to_csv(attendance_file, index=False)

                    # Phát hiện và vẽ khung khuôn mặt
                    face_objs = DeepFace.extract_faces(
                        frame, detector_backend=detector_backend, enforce_detection=False
                    )
                    for face in face_objs:
                        x, y, w, h = face["facial_area"].values()
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, student_name, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    break

    except Exception as e:
        print("❌ Lỗi nhận diện:", e)

    cv2.imshow("Face Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
