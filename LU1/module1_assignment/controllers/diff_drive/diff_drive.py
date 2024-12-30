from controller import Robot
import math

MAX_SPEED = 12.3
PI = math.pi
WHEEL_RADIUS = 0.095  # Half the wheel diameter (0.19m)
AXLE_LENGTH = 0.325  # Distance between the wheels (0.325m)

# Create the Robot instance
robot = Robot()

# Get the time step of the current world
timestep = int(robot.getBasicTimeStep())

# Get wheel motor controllers
leftMotor = robot.getDevice('left wheel')  
rightMotor = robot.getDevice('right wheel') 
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

# Set initial wheel velocities
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Define movement parameters
turn_duration = 1000  # Duration for turning 1s
grid_size = 1.0  # Grid is 1 m
num_grids_to_move = 4  # Move 4 m

# Function to move robot straight
def move_straight(speed, duration):
    leftMotor.setVelocity(speed)
    rightMotor.setVelocity(speed)
    robot.step(duration)

# Function to rotate robot in place
def rotate(speed, duration):
    leftMotor.setVelocity(-speed)  # Left wheel moves backward
    rightMotor.setVelocity(speed)   # Right wheel moves forward
    robot.step(duration)

# Step 1: Turn to face the goal
def turn_to_goal():
    rotate(MAX_SPEED * 0.2, turn_duration)  

# Step 2: Move towards the goal
def move_to_goal():
    distance = grid_size * num_grids_to_move  
    wheel_circumference = 2 * PI * WHEEL_RADIUS  # Circumference of the wheel
    num_rotations = distance / wheel_circumference  # Total number of wheel rotations
    rotation_speed = MAX_SPEED * 0.5  # Half of max speed 
    
    # Calculate duration for the movement
    move_time = int((num_rotations / rotation_speed) * timestep * 100)  
    move_straight(rotation_speed, move_time) 

# Step 3: Turn by 360 degree (one rotation)
def final_turn_and_stop():
    angle = 2 * PI 
    angular_velocity = (2 * MAX_SPEED * 0.2) / AXLE_LENGTH
    turn_duration = int((angle / angular_velocity) * 1000)

    # Rotate clockwise (right)
    leftMotor.setVelocity(MAX_SPEED * 0.6)  
    rightMotor.setVelocity(-MAX_SPEED * 0.6) # Right wheel moves backward
    robot.step(turn_duration)

    # Stop the robot after the turn
    leftMotor.setVelocity(0.0)
    rightMotor.setVelocity(0.0)
    
    # Allow a short delay to ensure it stops
    robot.step(100)  


# Main loop
while robot.step(timestep) != -1:
    turn_to_goal()          # Step 1: Turn to face the goal
    move_to_goal()         # Step 2: Move straight for 4 meters
    final_turn_and_stop()   # Step 3: Final turn to face the target orientation
   
    break  
 