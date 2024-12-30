from controller import Robot
import math

# Constants
MAX_SPEED = 12.3
STOP_DISTANCE = 0.5  
AXLE_LENGTH = 0.325  
WHEEL_RADIUS = 0.095  
PI = math.pi

# Create the Robot instance
robot = Robot()

# Get the time step of the current world
timestep = int(robot.getBasicTimeStep())

# Get wheel motor controllers
leftMotor = robot.getDevice('left wheel')
rightMotor = robot.getDevice('right wheel')
leftMotor.setPosition(float('inf'))  
rightMotor.setPosition(float('inf'))

# Get and enable lidar
lidar = robot.getDevice('Sick LMS 291')
lidar.enable(timestep)

# Function to rotate and detect the nearest wall
def rotate_and_detect():
    print("Starting full rotation for wall detection...")

    # Set motor speeds for rotation
    leftMotor.setVelocity(-MAX_SPEED * 0.2) 
    rightMotor.setVelocity(MAX_SPEED * 0.2)  

    # Initialize minimum distance and angle for the closest wall
    min_distance = float('inf')
    min_angle_index = -1

    # Rotate and scan for one full rotation 
    for _ in range(int(2 * PI / (MAX_SPEED * 0.2) * timestep)):
        robot.step(timestep)

        # Get current LIDAR measurements
        scan = lidar.getRangeImage()
        
        # Find the closest point during the rotation
        for i, distance in enumerate(scan):
            if 0 < distance < min_distance: 
                min_distance = distance
                min_angle_index = i  # Store index of the closest point

    # Stop the motors after rotation
    leftMotor.setVelocity(0.0)
    rightMotor.setVelocity(0.0)
    robot.step(timestep) 

    return min_distance, min_angle_index

# Function to move the robot straight a certain distance
def move_straight(distance):
    print(f"Moving straight for {distance:.2f} meters...")

    # Calculate the number of wheel rotations needed to move the specified distance
    wheel_circumference = 2 * PI * WHEEL_RADIUS
    num_rotations = distance / wheel_circumference

    # Calculate the duration for the movement
    move_duration = int((num_rotations / MAX_SPEED) * timestep)

    # Set motors to move straight
    leftMotor.setVelocity(MAX_SPEED * 0.5)
    rightMotor.setVelocity(MAX_SPEED * 0.5)
    
    # Move for the calculated duration
    for _ in range(move_duration):
        robot.step(timestep)

    # Stop the motors after moving
    leftMotor.setVelocity(0.0)
    rightMotor.setVelocity(0.0)

# Main logic
lidar.enablePointCloud()
# Perform rotation and detect nearest wall
minimum_distance, min_angle_index = rotate_and_detect()
print(f"Minimum distance to the wall detected: {minimum_distance:.2f} meters")

# Determine the target distance to stop 50 cm before the wall
target_distance = minimum_distance - STOP_DISTANCE

if target_distance > 0:
    # Rotate the robot to face the closest wall
    print("Rotating to face the closest wall...")

    # LiDAR index 0 is directly forward, 90 is directly to the right, and 180 is backward
    # Calculate the angle to rotate based on the index
    angle_to_rotate = (min_angle_index - 90) * (PI / 180)  # Convert index to radians
    
    # Set the motors to rotate towards the closest wall
    rotation_duration = abs(angle_to_rotate) / (MAX_SPEED * 0.2)
    leftMotor.setVelocity(-MAX_SPEED * 0.2)
    rightMotor.setVelocity(MAX_SPEED * 0.2)

    # Rotate for the calculated duration
    for _ in range(int(rotation_duration / timestep)):
        robot.step(timestep)

    # Stop the motors after rotating to the target direction
    leftMotor.setVelocity(0.0)
    rightMotor.setVelocity(0.0)
    robot.step(timestep)

    # Move towards the closest wall and stop 50 cm before
    move_straight(target_distance)
    print(f"Reached stopping distance from the wall: {STOP_DISTANCE:.2f} meters")
else:
    print(f"Already within 50 cm of the wall. No need to move.")
