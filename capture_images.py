import cv2
import os

# ====== CẤU HÌNH ======
student_name = "Nguyễn Văn Vững"
save_dir = f"dataset/{student_name}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
max_images = 10  # Số ảnh muốn chụp

print("📸 Nhấn 'S' để chụp ảnh - Nhấn 'Q' để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Hiển thị preview từ webcam
    cv2.imshow("Capture Face Images", frame)

    key = cv2.waitKey(1) & 0xFF

    # Nhấn S để lưu ảnh
    if key == ord('s'):
        filename = os.path.join(save_dir, f"{student_name.replace(' ', '')}{count+1}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ Đã lưu ảnh: {filename}")
        count += 1

        if count >= max_images:
            print("📦 Đã chụp đủ số lượng ảnh.")
            break

    # Nhấn Q để thoát sớm
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
