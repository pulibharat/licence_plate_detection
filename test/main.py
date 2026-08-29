# from ultralytics import YOLO

# model = YOLO("model/exp.pt")

# results = model("car.jpg", conf=0.5)

# results[0].show()

# vid

# from ultralytics import YOLO

# model = YOLO("model/exp.pt")

# results = model(
#     "cars_extended.mp4",
#     conf=0.5,
#     save=True
# )


# from ultralytics import YOLO

# model = YOLO("model/exp.pt")

# results = model(
#     "cars_extended.mp4",
#     conf=0.5,
#     show=True,
#     stream=True
# )

# for r in results:
#     pass


# from ultralytics import YOLO
# import torch

# print("CUDA available:", torch.cuda.is_available())
# print("GPU:", torch.cuda.get_device_name(0)
#       if torch.cuda.is_available() else "CPU")

# model = YOLO("model/exp.pt")

# results = model(
#     "cars_extended.mp4",
#     conf=0.5,
#     show=True,
#     stream=True,
#     device=0
# )

# for r in results:
#     pass


from ultralytics import YOLO
import torch
import cv2

print("CUDA available:", torch.cuda.is_available())

print(
    "GPU:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

model = YOLO("model/exp.pt")

cap = cv2.VideoCapture("data/locatiopn1.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run plate detector
    results = model(
        frame,
        conf=0.5,
        device=0,
        verbose=False
    )

    for r in results:

        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            # Crop the license plate
            plate_crop = frame[y1:y2, x1:x2]

            # Display plate crop
            cv2.imshow("Plate", plate_crop)

            # Draw detection
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    cv2.imshow("CCTV", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
