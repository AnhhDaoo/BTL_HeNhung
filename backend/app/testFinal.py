import cv2
import threading
import queue
import tkinter as tk
from tkinter import filedialog, Listbox, Button, messagebox
from PIL import Image, ImageTk
import time
from ultralytics import YOLO
import numpy as np
from scipy.spatial import distance as dist
import pygame
import os
from datetime import datetime

# Biến toàn cục gán ID duy nhất cho đối tượng mới được phát hiện và ánh xạ ID từ YOLO sang đối tượng được theo dõi
nextObjectID = 0
yolo_id_mapping = {}

# Lớp CentroidTracker quản lý việc theo dõi các đối tượng chuyển động
class CentroidTracker:
    # Hàm khởi tạo đối tượng
    def __init__(self, maxDisappeared=30):
        self.nextObjectID = 0
        self.objects = {}  # Lưu trữ ID và centroid của đối tượng
        self.disappeared = {}  # Số khung hình đối tượng đã biến mất
        self.maxDisappeared = maxDisappeared

    # Hàm đăng ký: gán ID cho đối tượng mới và thêm vào danh sách theo dõi
    def register(self, centroid):
        global nextObjectID
        self.objects[nextObjectID] = centroid
        self.disappeared[nextObjectID] = 0
        nextObjectID += 1

    # Hàm hủy đăng ký: xóa đối tượng khỏi danh sách theo dõi khi nó biến mất quá lâu
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    # Hàm cập nhật: cập nhật vị trí của các đối tượng dựa trên các bounding box mới
    def update(self, rects):
        
        # Nếu không có bounding boxes nào, tăng số lần biến mất cho tất cả đối tượng
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # Nếu vượt quá ngưỡng biến mất, xóa đối tượng
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Tính tọa độ tâm (centroid) từ các bounding boxes
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            inputCentroids[i] = (cX, cY)

        # Nếu không có đối tượng nào được theo dõi, đăng ký tất cả centroid
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
                
        # Nếu đã có đối tượng, gán centroid mới cho đối tượng dựa trên khoảng cách gần nhất giữa centroid hiện tại và centroid mới
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

# Khởi tạo pygame và tải âm thanh cảnh báo
pygame.init()
pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound('././assets/sounds/alert_sound.mp3')
except pygame.error as e:
    print(f"Lỗi khi tải âm thanh: {e}")
    alarm_sound = None

# Biến toàn cục cho việc theo dõi vật thể bằng YOLO
fgbg = cv2.createBackgroundSubtractorMOG2()

# Tải mô hình YOLO
try:
    model = YOLO('./backend/yolov8n.pt')
    print("Mô hình YOLO đã được tải thành công.")
except Exception as e:
    print(f"Lỗi khi tải mô hình YOLO: {e}")
    model = None

# Khởi tạo ứng dụng Tkinter
root = tk.Tk()
root.title("Hệ thống giám sát an ninh bằng hình ảnh")
root.geometry("1200x800")

# Các nhãn của video và thời gian
video_label = tk.Label(root)
video_label.pack(side="left")
time_label = tk.Label(root, font=('Helvetica', 12))
time_label.pack(anchor='ne')

# Listbox để hiển thị danh sách sự kiện
event_label = tk.Label(root, text="Danh sách sự kiện:")
event_label.pack(anchor='nw')
event_listbox = Listbox(root, width=50)
event_listbox.pack()

# Biến trạng thái để tạm dừng và tiếp tục ứng dụng
paused = False

# Chức năng tạm dừng chạy ứng dụng
def pause_program():
    global paused
    paused = True
    print("Ứng dụng đã được tạm dừng.")

# Chức năng tiếp tục chạy ứng dụng
def resume_program():
    global paused
    paused = False
    print("Ứng dụng đã được tiếp tục.")

# Các nút điều khiển play/stop trên màn hình
control_frame = tk.Frame(root)
control_frame.pack(side="top", fill="x")

play_button = Button(control_frame, text="Play", command=resume_program)
play_button.pack(side="left", padx=5)
stop_button = Button(control_frame, text="Stop", command=pause_program)
stop_button.pack(side="left", padx=5)

# Chức năng xem lại lịch sử
def view_history():
    # Tạo cửa sổ mới để hiển thị lịch sử đã ghi
    history_window = tk.Toplevel(root)
    history_window.title("Lịch sử đã ghi")
    history_window.geometry("600x400")

    scrollbar = tk.Scrollbar(history_window)
    scrollbar.pack(side="right", fill="y")

    video_listbox = tk.Listbox(history_window, yscrollcommand=scrollbar.set)
    video_listbox.pack(fill="both", expand=True)

    # Lấy danh sách các video đã ghi
    recorded_videos_dir = '././assets/recorded_videos'
    videos = [f for f in os.listdir(recorded_videos_dir) if f.endswith('.avi')]
    for video in videos:
        video_listbox.insert(tk.END, video)

    scrollbar.config(command=video_listbox.yview)

    def play_selected_video():
        selected = video_listbox.curselection()
        if selected:
            video_path = os.path.join(recorded_videos_dir, video_listbox.get(selected))
            play_recorded_video(video_path)

    def go_back():
        history_window.destroy()

    play_button_history = tk.Button(history_window, text="Phát Video", command=play_selected_video)
    play_button_history.pack(side="left")

    back_button = tk.Button(history_window, text="Trở về", command=go_back)
    back_button.pack(side="right")

history_button = Button(control_frame, text="Xem lại lịch sử", command=view_history)
history_button.pack(side="left", padx=5)

# Chức năng chọn khu vực cụ thể để theo dõi
roi = None
mask = None
displayed_frame = None

# Tạo luồng chọn ROI
def select_roi():
    threading.Thread(target=_select_roi_thread).start()

# Hàm chọn ROI
def _select_roi_thread():
    global roi, displayed_frame, mask
    if displayed_frame is not None:
        roi = cv2.selectROI("Chọn Vùng Quan Tâm", displayed_frame, False, False)
        cv2.destroyWindow("Chọn Vùng Quan Tâm")
        mask = np.zeros(displayed_frame.shape[:2], dtype="uint8")
        x, y, w, h = roi
        mask[int(y):int(y+h), int(x):int(x+w)] = 255
        print(f"ROI đã được chọn: x={x}, y={y}, w={w}, h={h}")
    else:
        print("Chưa có khung hình để chọn ROI")

# Nút chọn ROI
select_roi_button = Button(control_frame, text="Chọn ROI", command=select_roi)
select_roi_button.pack(side="left", padx=5)

# Mở VideoCapture từ camera hoặc video file (chọn 1 trong 2 nguồn)
try:
    # Mở video từ file
    # cap = cv2.VideoCapture('./assets/inputs/shop1.mp4')
    
    # Mở camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Không thể mở camera")
    else:
        print("Camera đã được mở thành công.")
except Exception as e:
    print(f"Lỗi: {e}")
    messagebox.showerror("Lỗi", f"Không thể mở camera: {e}")
    root.destroy()
    exit()

# Biến toàn cục cho xử lý video
detected_objects = {}
motion_objects = {}
stop_program = False
recording = False
out = None
previous_yolo_ids = set()

# Tạo thư mục lưu trữ video nếu chưa có
if not os.path.exists('././assets/recorded_videos'):
    os.makedirs('././assets/recorded_videos')
    print("Thư mục 'recorded_videos' đã được tạo.")

# Đối tượng chuyển động được theo dõi với số khung hình biến mất tối đa là 15
ct = CentroidTracker(maxDisappeared=15)

# Hàng đợi để truyền khung hình giữa các luồng
frame_queue = queue.Queue()
notification_queue = queue.Queue()

# Hàm xử lý video với chức năng: phát hiện chuyển động, nhận diện và theo dõi vật thể, ghi lại lịch sử
def video_processing():
    global stop_program, nextObjectID, recording, out, frame, mask, paused, previous_yolo_ids
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Lấy FPS từ webcam
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0:
        fps = 20  # Giá trị mặc định nếu không lấy được
    print(f"FPS từ webcam: {fps}")
    
    frame_delay = 1.0 / fps
    # Dừng ghi sau 5 giây không có đối tượng
    max_no_object_frames = int(5 * fps) 
    
    # Sử dụng bộ đệm để lưu trữ khung hình trước khi bắt đầu ghi
    buffer_frames = []
    buffer_size = int(2 * fps)  # Lưu trữ 2 giây khung hình trước khi phát hiện sự kiện

    # Vòng lặp xử lý video cho đến khi nhận được lệnh dừng
    while not stop_program:
        if paused:
            time.sleep(0.1)
            continue
        
        # Đọc khung hình từ camera, nếu không thể đọc được thì thoát khỏi vòng lặp
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình từ camera")
            break

        # Thay đổi kích thước khung hình cho phù hợp
        frame = cv2.resize(frame, (640, 480))

        # Áp dụng mask nếu ROI được chọn
        if mask is not None:
            frame_masked = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            frame_masked = frame.copy()

        # Xử lý hình ảnh để tìm chuyển động
        fgmask = fgbg.apply(frame_masked)
        _, thresh = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Tìm các bounding box của chuyển động
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            rects.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Cập nhật các đối tượng chuyển động
        objects = ct.update(rects)

        # Chức năng nhận diện chuyển động và theo dõi bằng thuật toán
        # Kiểm tra xem có sự kiện chuyển động mới được phát hiện hay không, nếu có thì thêm vào danh sách sự kiện và phát âm thanh cảnh báo
        event_detected = False
        for (objectID, centroid) in objects.items():
            if objectID not in motion_objects:
                motion_objects[objectID] = time.time()
                notification_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(motion_objects[objectID]))
                event_description = f"Motion ID: {objectID} detected at: {notification_time_str}\n"
                notification_queue.put(event_description)
                event_listbox.insert(tk.END, event_description)
                # Phát âm thanh cảnh báo
                if alarm_sound:
                    alarm_sound.play()
                print(f"Sự kiện chuyển động được phát hiện: {event_description}")
                event_detected = True

        # Xóa các đối tượng không còn chuyển động khỏi danh sách
        for objectID in list(motion_objects.keys()):
            if objectID not in objects:
                del motion_objects[objectID]

        # Nhận diện vật thể và theo dõi bằng YOLO
        if model is not None:
            try:
                results = model.track(frame_masked, persist=True)
            except Exception as e:
                print(f"Lỗi trong quá trình nhận diện bằng YOLO: {e}")
                results = []
        else:
            results = []
            print("Mô hình YOLO không được tải. Bỏ qua nhận diện vật thể.")

        current_yolo_ids = set()

        # Xử lý kết quả từ YOLO, vẽ bounding box, cập nhật danh sách, theo dõi và phát âm thanh cảnh báo
        for result in results:
            for box in result.boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    yolo_box_id = int(box.id.item()) if box.id is not None else None

                    if yolo_box_id is not None:
                        current_yolo_ids.add(yolo_box_id)
                        if yolo_box_id not in yolo_id_mapping:
                            yolo_id_mapping[yolo_box_id] = nextObjectID
                            detection_time = time.time()
                            detected_objects[nextObjectID] = detection_time
                            notification_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(detection_time))
                            event_description = f"YOLO ID: {nextObjectID} detected at: {notification_time_str}\n"
                            notification_queue.put(event_description)
                            event_listbox.insert(tk.END, event_description)
                            nextObjectID += 1
                            # Phát âm thanh cảnh báo
                            if alarm_sound:
                                alarm_sound.play()
                            print(f"Sự kiện YOLO được phát hiện: {event_description}")
                            event_detected = True
                        track_id = yolo_id_mapping[yolo_box_id]
                    else:
                        track_id = "N/A"

                    if track_id != "N/A":
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                except Exception as e:
                    print(f"Lỗi khi xử lý bounding box YOLO: {e}")

        # Xóa các YOLO ID đã biến mất ra khỏi danh sách
        vanished_yolo_ids = previous_yolo_ids - current_yolo_ids
        for yolo_id in vanished_yolo_ids:
            if yolo_id in yolo_id_mapping:
                objectID = yolo_id_mapping[yolo_id]
                if objectID in detected_objects:
                    del detected_objects[objectID]
                del yolo_id_mapping[yolo_id]
        previous_yolo_ids = current_yolo_ids

        # Thêm khung hình hiện tại vào bộ đệm
        buffer_frames.append(frame.copy())
        if len(buffer_frames) > buffer_size:
            buffer_frames.pop(0)

        # Kiểm tra xem có sự kiện được phát hiện hay không
        has_object = event_detected

        # Xử lý ghi lại video khi có sự kiện
        if has_object:
            if not recording:
                # Khởi tạo VideoWriter trước khi ghi buffer_frames
                filename = f'././assets/recorded_videos/recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.avi'
                try:
                    out = cv2.VideoWriter(filename, fourcc, fps, (frame.shape[1], frame.shape[0]))
                    if not out.isOpened():
                        print("Không thể mở VideoWriter")
                        out = None
                    else:
                        print(f"Bắt đầu ghi video: {filename}")
                        recording = True
                        # Ghi lại các khung hình trong bộ đệm trước sự kiện
                        for buffered_frame in buffer_frames:
                            out.write(buffered_frame)
                        buffer_frames = []  
                except Exception as e:
                    print(f"Lỗi khi khởi tạo VideoWriter: {e}")
                    out = None
            
            # Reset bộ đếm khung hình không có đối tượng
            no_object_frames = 0  
        else:
            if recording:
                no_object_frames += 1
                if no_object_frames >= max_no_object_frames:
                    print("Dừng ghi video do không có sự kiện.")
                    recording = False
                    no_object_frames = 0
                    if out is not None:
                        out.release()
                        out = None

        # Ghi khung hình nếu đang ghi
        if recording and out is not None:
            try:
                out.write(frame)
            except Exception as e:
                print(f"Lỗi khi ghi khung hình vào VideoWriter: {e}")

        # Đặt khung hình vào hàng đợi
        frame_queue.put(frame)

    # Sau khi kết thúc vòng lặp
    cap.release()
    if out is not None:
        out.release()
    print("Video processing thread đã kết thúc.")

# Hàm cập nhật khung hình và giao diện
def update_frame():
    global displayed_frame
    if stop_program:
        return

    try:
        frame = frame_queue.get_nowait()
    except queue.Empty:
        pass
    else:
        # Lưu khung hình hiện tại để sử dụng cho chức năng chọn ROI
        displayed_frame = frame.copy()

        # Cập nhật giao diện với khung hình mới
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        time_label.config(text=current_time)

    # Gọi lại hàm sau một khoảng thời gian (ms)
    video_label.after(10, update_frame)

# Chức năng phát video đã ghi
def play_recorded_video(video_path=None):
    if video_path is None:
        video_path = filedialog.askopenfilename(initialdir="././assets/recorded_videos/", title="Chọn video đã ghi")
    if video_path:
        # Mở cửa sổ mới để hiển thị video
        video_window = tk.Toplevel(root)
        video_window.title("Phát lại video")
        video_label_playback = tk.Label(video_window)
        video_label_playback.pack()
        # Cờ để kiểm tra cửa sổ đã đóng chưa
        video_window.closed = False 

        def on_close():
            video_window.closed = True
            video_window.destroy()

        video_window.protocol("WM_DELETE_WINDOW", on_close)

        # Chạy luồng phát video
        threading.Thread(target=_play_video_thread, args=(video_path, video_label_playback, video_window), daemon=True).start()

# Hàm định nghĩa luồng phát video đã ghi
def _play_video_thread(video_path, video_label_playback, video_window):
    try:
        cap_video = cv2.VideoCapture(video_path)
        if not cap_video.isOpened():
            print(f"Không thể mở video: {video_path}")
            return

        fps = cap_video.get(cv2.CAP_PROP_FPS)
        if fps == 0.0:
            fps = 20  # Giá trị mặc định nếu không lấy được
        print(f"Phát lại video với FPS: {fps}")
        frame_delay = 1.0 / fps

        while cap_video.isOpened() and not stop_program and not video_window.closed:
            ret, frame = cap_video.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label_playback.imgtk = imgtk
            video_label_playback.configure(image=imgtk)
            video_window.update()
            time.sleep(frame_delay)  # Chờ theo FPS
    except Exception as e:
        print(f"Lỗi trong quá trình phát video: {e}")
    finally:
        cap_video.release()
        if not video_window.closed:
            video_window.destroy()
        print("Đã kết thúc phát video.")

# Chức năng dừng video đã ghi
def stop_recorded_video():
    global stop_program
    stop_program = True
    print("Đã dừng chương trình.")

# Hàm xử lý khi đóng cửa sổ chính
def on_closing():
    exit_program()

# Hàm thoát chương trình
def exit_program():
    global stop_program
    stop_program = True
    print("Đang thoát ứng dụng...")
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    root.quit()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Nút thoát ứng dụng
exit_button = Button(control_frame, text="Thoát ứng dụng", command=exit_program, fg="red")
exit_button.pack(side="left", padx=5)

# Khởi tạo và bắt đầu luồng video
video_thread = threading.Thread(target=video_processing, daemon=True)
video_thread.start()
print("Đã bắt đầu luồng xử lý video.")

# Vòng lặp chính
update_frame()
root.mainloop()
print("Giao diện Tkinter đã được đóng.")
cap.release()
cv2.destroyAllWindows()
