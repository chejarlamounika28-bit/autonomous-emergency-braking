from ultralytics import YOLO
import cv2
import numpy as np
import time

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

MAX_SPEED = 80
BRAKE_DISTANCE_CM = 50
SAFE_DISTANCE_CM = 150
SWITCH_COOLDOWN = 3
last_switch_time = 0

STATIC_THRESHOLD = 5
prev_positions = {}

def get_lane_boundaries(frame):
    height, width = frame.shape[:2]
    left_lane = [(0, height), (width//3, height//2), (width//3, height)]
    center_lane = [(width//3, height), (2*width//3, height//2), (2*width//3, height)]
    right_lane = [(2*width//3, height), (width, height//2), (width, height)]
    return left_lane, center_lane, right_lane

blocking_labels = ["car","truck","bus","person","bicycle","motorbike","dog","cat","chair","traffic cone"]

print("🚗 Press ESC to quit.")
current_lane = "center"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape
    your_car_pos = (width//2, height)

    left_lane, center_lane, right_lane = get_lane_boundaries(frame)

    lane_dict = {"left": False, "center": False, "right": False}
    lane_distance = {"left": float('inf'), "center": float('inf'), "right": float('inf')}
    red_light_detected = False

    results = model(frame, stream=True)
    objects = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[cls]

            cx = (x1+x2)//2
            cy = (y1+y2)//2
            box_height = y2 - y1
            distance_cm = 7000 / (box_height+1)

            if label in ["traffic light"]:
                tl_roi = frame[y1:y2, x1:x2]
                hsv = cv2.cvtColor(tl_roi, cv2.COLOR_BGR2HSV)

                red_lower1 = np.array([0,120,70]); red_upper1 = np.array([10,255,255])
                red_lower2 = np.array([170,120,70]); red_upper2 = np.array([180,255,255])
                yellow_lower = np.array([15,100,100]); yellow_upper = np.array([35,255,255])
                green_lower = np.array([40,40,40]); green_upper = np.array([90,255,255])

                mask_red = cv2.inRange(hsv, red_lower1, red_upper1)+cv2.inRange(hsv, red_lower2, red_upper2)
                mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
                mask_green = cv2.inRange(hsv, green_lower, green_upper)

                if cv2.countNonZero(mask_red) > 20:
                    red_light_detected = True
                    cv2.putText(frame,"RED LIGHT",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                elif cv2.countNonZero(mask_yellow) > 20:
                    cv2.putText(frame,"YELLOW LIGHT",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,165,255),2)
                elif cv2.countNonZero(mask_green) > 20:
                    cv2.putText(frame,"GREEN LIGHT",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            display_label = label if label in blocking_labels else "object"

            prev_pos = prev_positions.get((label, cx, cy), (cx, cy))
            movement = np.sqrt((cx-prev_pos[0])**2 + (cy-prev_pos[1])**2)
            prev_positions[(label, cx, cy)] = (cx, cy)
            is_static = movement < STATIC_THRESHOLD

            color = (0,255,0) if label in blocking_labels else (0,165,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{display_label} {int(distance_cm)}cm",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

            if label in blocking_labels:
                objects.append((label, conf, cx, cy, x1, y1, x2, y2, distance_cm, is_static))
                if is_static:
                    if cx < width//3:
                        lane_dict["left"] = True
                        lane_distance["left"] = min(lane_distance["left"], distance_cm)
                    elif cx < 2*width//3:
                        lane_dict["center"] = True
                        lane_distance["center"] = min(lane_distance["center"], distance_cm)
                    else:
                        lane_dict["right"] = True
                        lane_distance["right"] = min(lane_distance["right"], distance_cm)
            else:
                objects.append((display_label, conf, cx, cy, x1, y1, x2, y2, distance_cm, is_static))

    nearest_distance = lane_distance[current_lane]

    if red_light_detected:
        speed = 0
        braking = "STOP for RED LIGHT!"
    elif nearest_distance >= SAFE_DISTANCE_CM or nearest_distance == float('inf'):
        speed = MAX_SPEED
        braking = "None"
    elif nearest_distance <= BRAKE_DISTANCE_CM:
        speed = 0
        braking = "Emergency Brake!"
    else:
        speed = MAX_SPEED * (nearest_distance - BRAKE_DISTANCE_CM)/(SAFE_DISTANCE_CM-BRAKE_DISTANCE_CM)
        braking = "Slow Brake"

    current_time = time.time()
    lane_text = ""
    if lane_dict[current_lane] and nearest_distance <= SAFE_DISTANCE_CM and not red_light_detected and (current_time-last_switch_time>SWITCH_COOLDOWN):
        for lane_option in ["left","center","right"]:
            if lane_option != current_lane and not lane_dict[lane_option]:
                current_lane = lane_option
                lane_text = f"⚙️ Shifting to {current_lane.upper()} lane"
                break
        else:
            lane_text = "⚠️ All lanes blocked! Brake HARD"
        last_switch_time = current_time
    else:
        lane_text = f"Staying in {current_lane.upper()} lane | Nearest object: {int(nearest_distance) if nearest_distance!=float('inf') else 'None'} cm"

    cv2.polylines(frame,[np.array(left_lane,np.int32)],True,(255,255,0),2)
    cv2.polylines(frame,[np.array(center_lane,np.int32)],True,(0,255,255),2)
    cv2.polylines(frame,[np.array(right_lane,np.int32)],True,(255,0,255),2)
    cv2.circle(frame,your_car_pos,5,(0,0,255),-1)
    cv2.putText(frame,"Your Car",(your_car_pos[0]-40,your_car_pos[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    cv2.putText(frame,f"Speed: {int(speed)} km/h | Braking: {braking}",(20,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(frame,lane_text,(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.imshow("Smart Lane & Obstacle Monitoring",frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
