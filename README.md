# HỆ THỐNG GIÁM SÁT AN NINH BẰNG HÌNH ẢNH

Ứng dụng sử dụng hình ảnh thu được từ camera để phát hiện người (hoặc vật thể) di chuyển, từ đó đưa ra thông báo cảnh báo **realtime** .

---

## Cách setup dự án trên Windows

#### 1. Môi trường yêu cầu

* **Python** phiên bản 3.9 trở lên.
* Visual Studio Code (VSCode) hoặc môi trường IDE tương đương.
* Windows 64-bit

#### 2. Mở dự án bằng Visual Studio Code

* Clone hoặc tải thư mục dự án về máy.
* Mở  **VSCode** , sau đó chọn **Open Folder** và dẫn đến thư mục dự án đã tải.

#### 3. Cài đặt các thư viện cần thiết

Các thư viện (packages) cần thiết được mô tả trong tệp `requirements.txt`. Để cài đặt, thực hiện các bước:

- Mở **Terminal** trong VSCode và điều hướng đến thư mục `backend` (nơi chứa `requirements.txt`):

  ```bash

  cd backend

  ```
- Cài đặt thư viện:

  ```bash

  pip install -r requirements.txt

  ```
- Chờ quá trình cài đặt hoàn tất.

#### 4. Chạy ứng dụng

- Di chuyển vào thư mục `app/`, chạy:

```bash

cd ../app

```

- Chạy file `testFinal.py` để khởi động ứng dụng  **Security Camera**

```bash

python testFinal.py

```

Sau khi chạy, giao diện ứng dụng sẽ hiển thị:

* Khung video (phát trực tiếp từ webcam hoặc từ nguồn video tuỳ chỉnh).
* Khu vực thông báo bên cạnh khung video.
* Các nút điều khiển (như  **Thoát ứng dụng** , v.v.).

> **Thoát ứng dụng** : Nhấn nút tương ứng để dừng và đóng giao diện chương trình.

---

## Lưu ý

* **Video mẫu đầu vào** được đặt trong thư mục `assets/inputs/` của dự án. Mặc định, chương trình lấy dữ liệu từ webcam.
* Để thay đổi nguồn video (sử dụng video mẫu), chỉnh đường dẫn trong đoạn code lựa chọn nguồn video tương ứng trong file **`testFinal.py`** (từ dòng 223).

```python

# Mở VideoCapture từ camera hoặc video file (chọn 1 trong 2 nguồn)

try:
    # Mở video từ file
    # cap = cv2.VideoCapture('./assets/inputs/shop1.mp4')

    # Mở camera
    cap = cv2.VideoCapture(0)

    ifnot cap.isOpened():
        raise Exception("Không thể mở camera")
    else:
        print("Camera đã được mở thành công.")

except Exception as e:
    print(f"Lỗi: {e}")
    messagebox.showerror("Lỗi", f"Không thể mở camera: {e}")
    root.destroy()
    exit()

```

* Mô hình YOLO được tải từ file `yolov8n.pt` trong thư mục `backend/`. File này được tự động sinh ra trong quá trình chạy và cần được đặt đường dẫn thích hợp trong mã nguồn.

---

## Chức năng chính của Hệ thống giám sát an ninh bằng hình ảnh

1. **Phát hiện và theo dõi chuyển động** :

* Sử dụng kỹ thuật trừ nền (background subtraction) để xác định khu vực, đối tượng đang di chuyển.

2. **Nhận dạng vật thể với mô hình YOLOv8** :

* Kết hợp YOLOv8 để tăng độ chính xác khi phát hiện và theo dõi vật thể.

3. **Khoanh vùng khu vực quan sát** :

* Người dùng có thể chọn một vùng cụ thể trong khung hình để tập trung giám sát.

4. **Ghi và xem lại bản record** :

* Ứng dụng tự động lưu lại video khi phát hiện đối tượng; cho phép xem lại sau đó.

5. **Giao diện trực quan với Tkinter** :

* Hiển thị khung video, thời gian, thông báo khi phát hiện chuyển động hoặc vật thể (kèm thời điểm).
* Chức năng chọn khu vực tập trung và xem lại bản record cũng được cung cấp trực tiếp trên giao diện.

---

**Chúc bạn cài đặt và sử dụng ứng dụng thành công!**

Nếu có thắc mắc hoặc gặp lỗi trong quá trình cài đặt và chạy, hãy kiểm tra lại các bước hoặc liên hệ người phát triển để được hỗ trợ.
