# main.py
import odrive
import cv2
import pygame
import time
import os
from realsense_depth import *
import Jetson.GPIO as GPIO
import busio
import board
import digitalio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# list_board_pins()
i2c=busio.I2C(board.SCL_1, board.SDA_1)
time.sleep(0.1)  # Small delay to stabilize the connection
ads=ADS.ADS1115(i2c, address=0x48)

chan1 = AnalogIn(ads, ADS.P0)
chan2=AnalogIn(ads, ADS.P3)
print(chan1.value, chan1.voltage)
print(chan2.value, chan2.voltage)

cap = cv2.VideoCapture(6)
#  Function to map chan1 value to a specific range (e.g., 10 to 100)
def map_value(value, input_min, input_max, output_min, output_max):
    return ((value - input_min) * (output_max - output_min)) / (input_max - input_min) + output_min

#Initialise GPIO Pins
pwm = digitalio.DigitalInOut(board.D22)
steering = digitalio.DigitalInOut(board.D13)
pwm.direction = digitalio.Direction.OUTPUT
steering.direction = digitalio.Direction.OUTPUT

#Initialise actuator IDLE 
pwm.value = False  # Set GPIO12 high
steering.value = False  # Set GPIO13 high

# Global variable to track drivable area detection and object detection
Steering = False
start_mode = True
steering_angle = 0
pot_value = 0
# global cap for camera
# cap = DepthCamera()

# Directory setup for recording images and videos
output_image_dir = 'output_images'
output_video_dir = 'output_videos'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_video_dir, exist_ok=True)

# Get the highest existing image counter in the output directory
existing_images = [f for f in os.listdir(output_image_dir) if f.startswith("object_frame_") and f.endswith(".png")]
if existing_images:
    img_counter = max([int(f.split('_')[2].split('.')[0]) for f in existing_images]) + 1
else:
    img_counter = 0

# Get the highest existing video counter in the output directory
existing_videos = [f for f in os.listdir(output_video_dir) if f.startswith("object_video_") and f.endswith(".mp4")]
if existing_videos:
    vid_counter = max([int(f.split('_')[2].split('.')[0]) for f in existing_videos]) + 1
else:
    vid_counter = 0

# Initialize counters and flags for recording and image capture
recording = False
out_annotated = None

# Initialize pygame
pygame.init()

# Define window size (To capture events)
screen = pygame.display.set_mode((640,480))



# Calibration for odrive wheels
def calibration(odrv, duration=20):
    """
    Move axis0 and axis1 of the given ODrive instance in a defined pattern
    for a specified duration.
    
    Parameters:
        odrv: ODrive instance
        duration: Total duration of the movement in seconds
    """
    # Define the key states
    states = [(2, -2), (0, 0), (-2, 2)]
    state_duration = duration / len(states)  # Duration for each state

    start_time = time.time()
    
    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time

        # Determine the current state based on elapsed time
        current_state_index = int(elapsed_time // state_duration) % len(states)
        input_vel = states[current_state_index]

        # Set input velocities for the axes
        odrv.axis0.controller.input_vel = input_vel[1]  # Axis 0
        odrv.axis1.controller.input_vel = input_vel[0]  # Axis 1
        time.sleep(2)  # Small delay to prevent busy waiting

    # Set both axes to zero velocity at the end
    odrv.axis0.controller.input_vel = 0
    odrv.axis1.controller.input_vel = 0

# Connect to ODrive
def connect_to_odrive():
    print("Finding odrive")
    try:
        odrv = odrive.find_any()
        if odrv is None:
            print("ODrive not found. Exiting program.")
            exit()
        print("ODrive connected successfully.")
        return odrv
    except Exception as e:
        print("Error connecting to ODrive:", e)
        exit()

def camera(cap):
    global frame
    if True:
        ret, frame = cap.read()
        idx = 0

        if not ret:
            print("Failed to grab frame")
            return
        annotated_frame = frame
        # Display the annotated frame
        cv2.imshow('YOLOv8 Detection', frame)

# Connect to ODrive and ESP
odrv0 = connect_to_odrive()
last_voltage_print_time = time.time()  # Initialize the timer

# Display drive voltage
print("ODrive Voltage:", odrv0.vbus_voltage)
odrv0.axis0.controller.config.control_mode = 2 # Vel Mode
odrv0.axis0.controller.config.input_mode = 2 #Vel Mode
odrv0.axis0.requested_state = 8  # Start Motor
odrv0.axis1.requested_state = 8 # Start Motor


def Manual_drive(keys):
    global Steering, start_mode, pot_value, steering_angle

    # Read potentiometer value
    try:
        pot_value = map_value(chan1.value, 0, 26230, 0, 1023)
        steering_angle = map_value(pot_value, 0, 1023, -40, 40)
    except OSError as e:
        print(f"Error reading potentiometer: {e}")
        pot_value, steering_angle = None, None

    # Default behavior: Stop motors if no keys are pressed
    odrv0.axis1.controller.input_vel = 0
    odrv0.axis0.controller.input_vel = 0
    pwm.value = False
    steering.value = False

    # Handle specific key presses
    if keys[pygame.K_e]:  # Toggle motor start/stop
        start_mode = not start_mode

        if start_mode:  # Start motors
            print("Starting...")
            odrv0.axis0.requested_state = 8  # Start Motor
            odrv0.axis1.requested_state = 8  # Start Motor
        else:  # Stop motors
            print("Stopping...")
            odrv0.axis1.requested_state = 1  # Set ODrive to idle state
            odrv0.axis0.requested_state = 1  # Set ODrive to idle state

    if keys[pygame.K_w]:  # Move forward
        odrv0.axis1.controller.input_vel = -1
        odrv0.axis0.controller.input_vel = -1
        print("FORWARD")

    if keys[pygame.K_s]:  # Move backward
        odrv0.axis1.controller.input_vel = 1
        odrv0.axis0.controller.input_vel = 1
        print("BACKWARD")

    if keys[pygame.K_a] and steering_angle is not None and steering_angle <= 39:  # Steer Left
        pwm.value = True
        steering.value = False
        print(f"Steering Left: Potentiometer Value: {int(steering_angle)}")

    if keys[pygame.K_d] and steering_angle is not None and steering_angle >= -25:  # Steer Right
        pwm.value = True
        steering.value = True
        print(f"Steering Right: Potentiometer Value: {int(steering_angle)}")

    if pygame.key.get_mods() & pygame.KMOD_SHIFT:  # Brake
        odrv0.axis1.controller.input_vel = 0
        odrv0.axis0.controller.input_vel = 0
        print("STOP")
    if pot_value is None:
            return        
    if steering_angle is None:
            return
    return pot_value
        
def main():
    # global exit variable
    global exit_flag, recording, out_annotated, img_counter, vid_counter,frame,steering_angle, last_voltage_print_time
    # Start lane detection in a separate thread or process
    # Main event loop
    while not exit_flag:
        # Print the ODrive voltage at a 1-second interval
        current_time = time.time()
        if current_time - last_voltage_print_time >= 0.3:  # 0.3 second interval
            print("ODrive Voltage:", odrv0.vbus_voltage)
            last_voltage_print_time = current_time        # Poll for pygame events
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if steering_angle is None:
            return

        camera(cap)
        Manual_drive(keys)
        # Check for 'q' key in pygame to exit
        if keys[pygame.K_q]:
            print("Resetting")
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis0.controller.input_vel = 0
            if odrv0.axis0.controller.input_vel == 0 or odrv0.axis1.controller.input_vel == 0:
                odrv0.axis1.requested_state = 3  # Set ODrive to idle state
                odrv0.axis0.requested_state = 3  # Set ODrive to idle state

        pot_value = Manual_drive(keys)
        if pot_value == None:
            continue
        
        # If recording, write the frame to the video output file
        if recording:
            out_annotated.write(frame)

        # Break the loop if 'ESC'
        elif keys[pygame.K_ESCAPE]:  # ESC to exit and restart motor
            print("Exiting...")
            odrv0.axis1.controller.input_vel = 0
            odrv0.axis1.requested_state = 1  # Set ODrive to idle state
            odrv0.axis0.controller.input_vel = 0
            odrv0.axis0.requested_state = 1  # Set ODrive to idle state
            exit_flag = True       
            break

        #Recording functionality with 'r' key for both frames
        if keys[pygame.K_r]:
            if not recording:
                print("Recording started for both annotated and combined frames...")
                # Set up video writers for both annotated and combined frames
                vid_name_annotated = os.path.join(output_video_dir, f"object_video_{vid_counter}.mp4")
                vid_name_combined = os.path.join(output_video_dir, f"lane_video_{vid_counter}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                out_annotated = cv2.VideoWriter(vid_name_annotated, fourcc, 20.0, 
                                                (frame.shape[1], frame.shape[0]))
                recording = True
            else:
                print("Recording stopped.")
                recording = False
                out_annotated.release()  # Stop recording and release the output file for annotated frames
                print(f"{vid_name_annotated} and {vid_name_combined} saved!")
                vid_counter += 1

        # Image capture functionality with 't' key for both frames
        elif keys[pygame.K_t]:
            img_name_annotated = os.path.join(output_image_dir, f"object_frame_{img_counter}.png")
            img_name_combined = os.path.join(output_image_dir, f"lane_frame_{img_counter}.png")
            cv2.imwrite(img_name_annotated, frame)  # Save YOLO-detected frame
            print(f"Images saved: {img_name_annotated} and {img_name_combined}")
            img_counter += 1

        # 'l' key to see status
        elif keys[pygame.K_l]: 
            print("Status:")
            print ("recording:",[{recording}])

        cv2.waitKey(1)
    #Quit all components
    pygame.quit()
    # cap.release()
    cap.release()
    cv2.destroyAllWindows()  # Close windows after exiting
    print("Exiting all components.")

exit_flag = False

if  __name__ == '__main__':

    main()