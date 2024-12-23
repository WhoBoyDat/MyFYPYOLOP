import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio
import pygame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image
import odrive

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm

#YOLO
from ultralytics import YOLO
from collections import Counter
from realsense_depth import *
from ultralytics.utils.plotting import Annotator, colors

#JETSON PINS
import Jetson.GPIO as GPIO
import busio
import board
import digitalio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

#ESP
from fastapi import FastAPI, WebSocket
import uvicorn
import json

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# Initialize Pygame for PS4 controller vibration
pygame.init()

#Run Fast API
app = FastAPI()

# Store connected WebSocket clients (if needed for multi-client scenarios)
connected_clients = []

#Direction to turn
dir = None  
angleToTurn = None

# global cap for camera
cap = DepthCamera()

# Load the model
modely = YOLO("best.pt")
namesy = modely.names

#object detection class_id
class_ids=[]

run_detection = True

#Manual_mode
is_recording = False
manual_mode = True
start_mode = True
exit_flag = False
# Timing variables
last_press_time = 0
double_tap_threshold = 0.3  # Maximum time (seconds) between taps to count as a double-tap

# #Actuator
i2c=busio.I2C(board.SCL_1, board.SDA_1)
time.sleep(0.1)  # Small delay to stabilize the connection
ads=ADS.ADS1115(i2c, address=0x48)

chan1 = AnalogIn(ads, ADS.P0)
chan2=AnalogIn(ads, ADS.P3)
print(chan1.value, chan1.voltage)
print(chan2.value, chan2.voltage)

# Create directories for output if they don't exist
output_image_dir = 'output_images'
output_video_dir = 'output_videos'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_video_dir, exist_ok=True)

# Get the highest existing image counter in the output directory
existing_images = [f for f in os.listdir(output_image_dir) if f.startswith("opencv_frame_") and f.endswith(".png")]
if existing_images:
    img_counter = max([int(f.split('_')[2].split('.')[0]) for f in existing_images]) + 1
else:
    img_counter = 0

# Get the highest existing video counter in the output directory
existing_videos = [f for f in os.listdir(output_video_dir) if f.startswith("output_video_") and f.endswith(".mp4")]
if existing_videos:
    vid_counter = max([int(f.split('_')[2].split('.')[0]) for f in existing_videos]) + 1
else:
    vid_counter = 0

joysticks = {}

        
# Function to map chan1 value to a specific range (e.g., 10 to 100)
def map_potentiometer_value(value, input_min=0, input_max=26230, output_min=0, output_max=1023):
    return ((value - input_min) * (output_max - output_min)) / (input_max - input_min) + output_min

#Initialise GPIO Pins
pwm = digitalio.DigitalInOut(board.D12)
steering = digitalio.DigitalInOut(board.D13)
pwm.direction = digitalio.Direction.OUTPUT
steering.direction = digitalio.Direction.OUTPUT

#Initial Low 
pwm.value = True  # Set GPIO12 high
steering.value = True  # Set GPIO13 low

# Function to connect to ODrive
def connect_to_odrive():
    try:
        odrv = odrive.find_any()
        if odrv is None:
            print("ODrive not found. Exiting.")
            sys.exit()
        return odrv
    except Exception as e:
        print(f"Error connecting to ODrive: {e}")
        sys.exit()

#gets centre of bounding bvox
def get_center_of_bbox(box):
    """ Calculate the center point of the bounding box """
    x_center = int((box[0] + box[2]) / 2)
    y_center = int((box[1] + box[3]) / 2)
    return (x_center, y_center)
    

# Emergency stop function
def emergency_stop(odrv):
    try:
        odrv.axis0.controller.input_vel = 0
        odrv.axis1.controller.input_vel = 0
        odrv.axis0.requested_state = 1  # Set to idle state
        odrv.axis1.requested_state = 1
        print("Emergency Stop Activated!")
    except Exception as e:
        print(f"Error during emergency stop: {e}")


# Analyze segmentation results for drivable area and lane position
def process_segmentation(da_seg_mask, ll_seg_mask, img_det):
    h, w = da_seg_mask.shape
    vehicle_center = w // 2

    # Detect drivable area
    front_row = da_seg_mask[int(h * 0.8):, :]  # Bottom 20% of the frame
    drivable = np.sum(front_row) > 0

    # Analyze lane position
    bottom_row = ll_seg_mask[-1, :]  # Bottom row of the lane mask
    lane_pixels = np.where(bottom_row > 0)[0]

    if len(lane_pixels) >= 2:
        left_lane = lane_pixels[0]
        right_lane = lane_pixels[-1]
        lane_center = (left_lane + right_lane) // 2

        if vehicle_center < left_lane:
            lane_position = "left"
            cv2.putText(img_det, "Lane Shift Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        elif vehicle_center > right_lane:
            lane_position = "right"
            cv2.putText(img_det, "Lane Shift Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        else:
            lane_position = "center"
            cv2.putText(img_det, "Good Lane Keeping", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        lane_position = "unknown"

    return drivable, lane_position

def Manual_drive(joysticks, odrv0, img_det, annotated_frame):
    # global exit variable
    global img_counter, vid_counter, Steering, start_mode, last_press_time, double_tap_threshold, manual_mode, exit_flag, is_recording, vid_name, vid_name1, out ,out1
    try:
        pot_value = map_potentiometer_value(chan1.value)
    except OSError as e:
        print(f"Error reading potentiometer: {e}")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_flag = True  # Flag that we are done so we exit this loop.

        if event.type == pygame.JOYBUTTONDOWN:
            # Square button
            if event.button == 3:
                start_mode = not start_mode
                #Start motor
                if start_mode:
                    print("Starting...")
                    odrv0.axis0.requested_state = 8  # Start Motor
                    odrv0.axis1.requested_state = 8 # Start Motor
                    odrv0.axis1.controller.input_vel = 0
                    odrv0.axis0.controller.input_vel = 0


                #Motor Idle
                if start_mode is False: 
                    print("Resetting")
                    odrv0.axis1.controller.input_vel = 0
                    odrv0.axis0.controller.input_vel = 0
                    if odrv0.axis0.controller.input_vel == 0 or odrv0.axis1.controller.input_vel == 0:
                        odrv0.axis1.requested_state = 1  # Set ODrive to idle state
                        odrv0.axis0.requested_state = 1  # Set ODrive to idle state
            
            # Circle Button
            if event.button == 1:
                print("Status:")
                print ("recording:",[{is_recording}])

            # Triangle Button
            if event.button == 2:
                if not is_recording:
                    print("Recording started...")
                    vid_name = os.path.join(output_video_dir, f"output_video_{vid_counter}.mp4")  # Set video name
                    vid_name1 = os.path.join(output_video_dir, f"lane_video_{vid_counter}.mp4")  # Set video name
                    
                    # Define the codec and create a VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(vid_name, fourcc, 20.0, (annotated_frame.shape[1], annotated_frame.shape[0]))
                    out1 = cv2.VideoWriter(vid_name1, fourcc, 20.0, (img_det.shape[1], img_det.shape[0]))
                    is_recording = True 
                else:
                    print("Recording stopped.")
                    is_recording = False
                    print(f"{vid_name} written!")
                    print(f"{vid_name1} written!")

                    vid_counter += 1
                    out.release()  # Stop recording and release the output file
                    out1.release()
            
            # X Button
            if event.button == 0:
                print ("Image Captured")
                img_name_annotated = os.path.join(output_image_dir, f"object_frame_{img_counter}.png")
                img_name_combined = os.path.join(output_image_dir, f"lane_frame_{img_counter}.png")
                cv2.imwrite(img_name_annotated, annotated_frame)  # Save YOLO-detected frame
                print(f"Images saved: {img_name_annotated} and {img_name_combined}")
                img_counter += 1

            #Left axis Motion Button
            if event.button == 11:
                odrv0.axis1.controller.input_vel = 0
                odrv0.axis0.controller.input_vel = 0
                print ('STOP')
            #Button R1    
            if event.button == 7:
                manual_mode = not manual_mode
                print (f"Manual: {manual_mode}")
                odrv0.axis1.controller.input_vel = 0
                odrv0.axis1.controller.input_vel = 0

            if event.button == 10:  # Power Button
                print("Exiting...")
                odrv0.axis1.controller.input_vel = 0
                odrv0.axis1.requested_state = 1  # Set ODrive to idle state
                odrv0.axis1.controller.input_vel = 0
                odrv0.axis0.requested_state = 1  # Set ODrive to idle state
                exit_flag = True       

        if event.type == pygame.JOYBUTTONUP:
            print("Joystick button released.")
        # Handle hotplugging
        if event.type == pygame.JOYDEVICEADDED:
            # This event will be generated when the program starts for every
            # joystick, filling up the list without needing to create them manually.
            joy = pygame.joystick.Joystick(event.device_index)
            joysticks[joy.get_instance_id()] = joy
            print(f"Joystick {joy.get_instance_id()} connencted")

        if event.type == pygame.JOYDEVICEREMOVED:
            del joysticks[event.instance_id]
            print(f"Joystick {event.instance_id} disconnected")

        # Hat Motion
        if event.type == pygame.JOYHATMOTION:
            if event.instance_id in joysticks:
                joy = joysticks[event.instance_id]
                hat_index = event.hat
                hat_position = joy.get_hat(hat_index)

                current_time = time.time()
                time_since_last_press = current_time - last_press_time

                # Example logic for specific hat positions
                if hat_position == (0, 1):  # Up
                    if time_since_last_press <= double_tap_threshold:
                        odrv0.axis1.controller.input_vel = -1
                        odrv0.axis0.controller.input_vel = -1
                        print('Fast FORWARD')

                    else:
                        odrv0.axis1.controller.input_vel = -0.7
                        odrv0.axis0.controller.input_vel = -0.7
                        print ('FORWARD')

                elif hat_position == (0, -1):  # Down
                    if time_since_last_press <= double_tap_threshold:
                        odrv0.axis1.controller.input_vel = 1
                        odrv0.axis0.controller.input_vel = 1             
                        print ('Fast BACKWARD')
                    else:
                        odrv0.axis1.controller.input_vel = 0.7
                        odrv0.axis0.controller.input_vel = 0.7
                        print ('BACKWARD')
                if hat_position == (-1, 0):  # Left
                    pwm.value = False
                    steering.value = True
                    if pot_value is not None:
                        print(f"Steering Left: Potentiometer Value: {pot_value}")
                    Steering = True
                    print ("Left")
                    time.sleep(0.1)
                elif hat_position == (1, 0):  # Right
                    pwm.value = False
                    steering.value = False
                    if pot_value is not None:
                        print(f"Steering Right: Potentiometer Value: {pot_value}")
                    Steering = True
                    print ("Right") 
                    time.sleep(0.1)
                elif hat_position == (0, 0):  # Centered/Neutral
                    pwm.value = True
                    steering.value = True

                    Steering = False
                    print ("No Steering")
                # Update the last press time
                last_press_time = current_time
        # Axis Motion
        if event.type == pygame.JOYAXISMOTION:
                # Handle joystick axis motion
            if event.instance_id in joysticks:
                joy = joysticks[event.instance_id]            
                x_axis = joy.get_axis(0)  # Horizontal axis
                y_axis = joy.get_axis(1)  # Vertical axis
                AXIS_THRESHOLD = 0.4 # Adjust
                HIGH_AXIS_THRESHOLD = 0.8
                """Determine the joystick direction based on axis values."""
                if abs(x_axis) < AXIS_THRESHOLD and abs(y_axis) < AXIS_THRESHOLD: # Steering Idle
                    pwm.value = True
                    steering.value = True
                    if pot_value is not None:
                        print(f"No Steering: Potentiometer Value: {pot_value}")
                    Steering = False
                if y_axis < -AXIS_THRESHOLD:   # Move Forward
                    odrv0.axis1.controller.input_vel = -0.7
                    odrv0.axis0.controller.input_vel = -0.7
                    print ('FORWARD')

                if y_axis > AXIS_THRESHOLD: # Move BackWard
                    odrv0.axis1.controller.input_vel = 0.7
                    odrv0.axis0.controller.input_vel = 0.7           
                    print ('BACKWARD')

                if y_axis < -HIGH_AXIS_THRESHOLD:   # Move Forward
                    odrv0.axis1.controller.input_vel = -1
                    odrv0.axis0.controller.input_vel = -1
                    print ('FAST FORWARD')

                if y_axis > HIGH_AXIS_THRESHOLD: # Move BackWard
                    odrv0.axis1.controller.input_vel = 1
                    odrv0.axis0.controller.input_vel = 1             
                    print ('FAST BACKWARD')

                if x_axis < -AXIS_THRESHOLD and pot_value <= 800:  # Steer Left
                    pwm.value = False
                    steering.value = True
                    if pot_value is not None:
                        print(f"Steering Left: Potentiometer Value: {pot_value}")
                    Steering = True
                    print ("Left")

                if x_axis > AXIS_THRESHOLD and pot_value >= 300:  # Steer Right
                    pwm.value = False
                    steering.value = False
                    if pot_value is not None:
                        print(f"Steering Right: Potentiometer Value: {pot_value}")
                    Steering = True
                    print ("Right")    
        if exit_flag:
            break

def object_detection(class_ids, odrv0, distance):
        global detected, run_detection
        # Control motors based on object detections
        people_detected = False
        # People or stop sign detected
        if 4 in class_ids or 8 in class_ids:
            print("Stop")
            run_detection = True
            detected = True
            people_detected = True
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis0.controller.input_vel = 0

        # Slow or Speed sign detected
        elif 6 in class_ids or 7 in class_ids:
            detected = True
            print("Slow")
            odrv0.axis1.controller.input_vel = -0.8
            odrv0.axis0.controller.input_vel = -0.8

        # Hump or Pedestrain Sign detected
        if 1 in class_ids or 3 in class_ids:
            detected = True
            print("Pedestrain")
            odrv0.axis1.controller.input_vel = -0.8
            odrv0.axis0.controller.input_vel = -0.8
        
        if 10 in class_ids and distance <= 500:
            print("CAR")
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis0.controller.input_vel = 0
        
        elif 10 in class_ids and distance >= 500 and not people_detected:
            detected = False

        if not class_ids:
            detected = False

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint to receive data from BLE and send commands to sensors."""
    await websocket.accept()
    connected_clients.append(websocket)
    print("WebSocket client connected")

    try:
        if True:
            # Wait for messages from the WebSocket client
            data = await websocket.receive_text()
            print(f"Received from WebSocket: {data}")
            # Parse the incoming JSON data
            try:
                message = json.loads(data)
                print(f"Parsed message: {message}")
                if isinstance(message, str):
                    print("Parsed message is not a dictionary")
                    message = json.loads(message)
                    print(message)
                    print(f"Type of data: {type(message)}")

                if "vehicleControl" in message:
                    dir = message["modifier"]
                    angleToTurn = message["angleToTurn"] 
                    print("Direction to turn to: ", dir)
                    print(f"Turn {angleToTurn} degrees")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON format: {e}")
                
            # Optionally send a response back to the BLE system or other clients
            response = {"status": "processed", "originalMessage": message}
            await websocket.send_text(json.dumps(response))
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connected_clients.remove(websocket)
        print("WebSocket client disconnected")

# Main detection and control loop
def detect(cfg, opt):
    #global variables to pass
    global class_ids, annotated_frame, run_detection, detected, is_recording, vid_counter, img_counter, img_det, vid_name , vid_name1, out , out1
    detected = False
    run_detection = True
    distance = 0
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')
    device = select_device(logger, opt.device)
    # Set the width and height of the screen (width, height), and name the window.
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Joystick Info")

    # Prepare output directory
    if os.path.exists(opt.save_dir):
        shutil.rmtree(opt.save_dir)
    os.makedirs(opt.save_dir)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device).eval()

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size

    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors1 = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Connect to ODrive
    odrv0 = connect_to_odrive()
    odrv0.axis0.requested_state = 8  # Closed-loop velocity control
    odrv0.axis1.requested_state = 8  # Closed-loop velocity control
    # Start inference
    for path, img, img_det, vid_cap, shapes in dataset:
        img = transform(img).to(device).unsqueeze(0)

        # Inference
        _, da_seg_out, ll_seg_out = model(img)

        # Process segmentation outputs
        _, _, height, width = img.shape
        da_seg_mask = torch.nn.functional.interpolate(da_seg_out, scale_factor=1, mode='bilinear').argmax(1).squeeze().cpu().numpy()
        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, scale_factor=1, mode='bilinear').argmax(1).squeeze().cpu().numpy()

        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

        # Get Camera
        ret, success, frame = cap.get_frame()
        idx = 0

        if not ret:
            print("Failed to grab frame")
            return

        # Run YOLO detection if enabled
        if run_detection:
            results = modely(frame, conf=0.5, verbose=False, device=0)  # Running detection
            annotated_frame = results[0].plot()  # Annotate frame with segmentaion mask
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(annotated_frame, line_width=2, example=namesy)

            if boxes is not None:
                for box, cls in zip(boxes, clss):
                    idx += 1
                    annotator.box_label(box, color=colors(int(cls), True), label=namesy[int(cls)])

                    crop_obj = annotated_frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
                    center_point = get_center_of_bbox(box)

                    cv2.circle(annotated_frame, center_point, 5, (0, 0, 255), -1)
                    distance = success[center_point[1], center_point[0]]
                    cv2.putText(annotated_frame, "{}mm".format(distance), (center_point[0], center_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

            # Process and print class names for each detection
            for result in results:
                class_ids = result.boxes.cls.tolist()
                class_names = [modely.names[int(id)] for id in class_ids]  # Map IDs to names
            
            # Count occurrences of each class name
            class_counts = Counter(class_names)
            for class_name, count in class_counts.items():
                print(f"{count} {class_name} detected.")
        else:
            results = modely(frame, conf=0.5, verbose=False)  # Running detection
            annotated_frame = results[0].plot()

        if not is_recording:
            print("Recording started...")
            vid_name = os.path.join(output_video_dir, f"output_video_{vid_counter}.mp4")  # Set video name
            vid_name1 = os.path.join(output_video_dir, f"lane_video_{vid_counter}.mp4")  # Set video name
            
            # Define the codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_name, fourcc, 20.0, (annotated_frame.shape[1], annotated_frame.shape[0]))
            out1 = cv2.VideoWriter(vid_name1, fourcc, 20.0, (img_det.shape[1], img_det.shape[0]))
            is_recording = True
        pot_value = map_potentiometer_value(chan1.value)

        # Analyze results
        drivable, lane_position = process_segmentation(da_seg_mask, ll_seg_mask, img_det)
        Manual_drive(joysticks, odrv0, img_det, annotated_frame)
        if not manual_mode:
        # Control logic
            object_detection(class_ids, odrv0, distance)
            if not detected:
                if drivable:
                    print("Drivable area detected.")
                    odrv0.axis0.controller.input_vel = -1.0  # Move forward
                    odrv0.axis1.controller.input_vel = -1.0  # Move forward
                    cv2.putText(img_det, "Drivable Area", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                else:
                    print("No drivable area detected. Stopping.")
                    odrv0.axis0.controller.input_vel = 0  # Stop
                    odrv0.axis1.controller.input_vel = 0  # Stop

                if lane_position == "left" and not detected:
                    print("Vehicle drifting left. Slowing down.")
                    odrv0.axis0.controller.input_vel = -0.8
                    odrv0.axis1.controller.input_vel = -1   # Stop
                    pwm.value = False
                    steering.value = True
                    if pot_value is not None:
                        print(f"Steering Left: Potentiometer Value: {pot_value}")
                    
                elif lane_position == "right" and not detected:
                    print("Vehicle drifting right. Slowing down.")
                    odrv0.axis0.controller.input_vel = -1
                    odrv0.axis1.controller.input_vel = -0.8   # Stop
                    pwm.value = False
                    steering.value = False
                    if pot_value is not None:
                        print(f"Steering Right: Potentiometer Value: {pot_value}")

                elif lane_position == "center":
                    print("Vehicle centered in lane.")

                if pot_value <= 300 and not lane_position == "left":  #Steer Right Limit
                    pwm.value = True
                    steering.value = True
                    if pot_value is not None:
                        print(f"No Steering: Potentiometer Value: {pot_value}")

                if pot_value >= 800 and not lane_position == "right": #Steer Left Limit
                    pwm.value = True
                    steering.value = True
                    if pot_value is not None:
                        print(f"No Steering: Potentiometer Value: {pot_value}")

                if lane_position == "center" or not lane_position: # Joystick IDLE
                    pwm.value = True
                    steering.value = True
                    if pot_value is not None:
                        print(f"No Steering: Potentiometer Value: {pot_value}")
                if pot_value == None:
                    continue
                            # GPU memory usage
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # In MB
        reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)    # In MB
        free_memory = reserved_memory - allocated_memory                      # Free within reserved

        # print(f"Allocated Memory: {allocated_memory:.2f} MB")
        # print(f"Reserved Memory: {reserved_memory:.2f} MB")
        # print(f"Free Memory within Reserved: {free_memory:.2f} MB")
        cv2.imshow('YOLOP Inference', img_det)
        # Display the annotated frame
        cv2.imshow('YOLOv8 Detection', annotated_frame)
        if pot_value is not None:
            print(f"No Steering: Potentiometer Value: {pot_value}")
        if is_recording:
            out.write(annotated_frame)
            out1.write(img_det)
        key = cv2.waitKey(1) & 0xFF  # Adjust delay based on video FPS / change to 1 or 0

        # 's' key to capture an image
        if key == ord('s'):
            img_name = os.path.join(output_image_dir, f"opencv_frame_{img_counter}.png")  # Set image name
            img_name1 = os.path.join(output_image_dir, f"lanes_frame_{img_counter}.png")  # Set image name

            cv2.imwrite(img_name, annotated_frame)  # Write image file
            cv2.imwrite(img_name1, img_det)  # Write image file

            print(f"{img_name} written!")
            print(f"{img_name1} written!")

            img_counter += 1
        # Stop on key press
        if key == ord('q'):
            emergency_stop(odrv0)
            break
        if exit_flag:
            break

    emergency_stop(odrv0)
    print("Inference completed.")
    print("Recording stopped.")
    is_recording = False
    print(f"{vid_name} written!")
    print(f"{vid_name1} written!")

    vid_counter += 1
    out.release()  # Stop recording and release the output file
    out1.release()
    pygame.quit
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/End-to-end.pth', help='Path to model weights')
    parser.add_argument('--source', type=str, default='6', help='Input source (file/folder)')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size (pixels)')
    parser.add_argument('--device', default='0,1,2,3', help='Device: cpu or cuda')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='Directory to save results')
    opt = parser.parse_args()

    with torch.no_grad():
        import threading
        detect_thread = threading.Thread(target=detect, args=(cfg, opt))
        detect_thread.start()

        # # Thread for running the FastAPI server
        # unicorn_thread = threading.Thread(target=uvicorn.run, kwargs={"app": app, "host": "127.0.0.1", "port": 8765})
        # unicorn_thread.start()

        # Optionally, join threads if needed
        detect_thread.join()
        # unicorn_thread.join()