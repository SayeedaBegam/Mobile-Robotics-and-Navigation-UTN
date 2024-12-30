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
grid_size = 1.0  # Grid is 1 m
num_grids_to_move = 4  # Move 4 m

# Step 1: Turn and move to the goal in one action
def turn_and_move_to_goal():
    # Calculate the distance to move
    distance = grid_size * num_grids_to_move  
    wheel_circumference = 2 * PI * WHEEL_RADIUS  # Circumference of the wheel
    num_rotations = distance / wheel_circumference  # Total number of wheel rotations
    rotation_speed = MAX_SPEED * 0.5  # Half of max speed 
    
    # Calculate duration for the movement
    move_time = int((num_rotations / rotation_speed) * timestep * 100)  
    
    # Set velocities for turning and moving forward in one step
    leftMotor.setVelocity(rotation_speed * 0.8)  # Left wheel moves slower to turn slightly
    rightMotor.setVelocity(rotation_speed)       # Right wheel moves faster to create a slight turn

    # Let the robot move towards the goal
    robot.step(move_time)

# Step 2: Stop and turn back to the initial orientation
def stop_and_turn_back():
    # Calculate the angle to turn back (180 degrees or PI radians)
    angle = PI  # 180 degrees in radians
    angular_velocity = (2 * MAX_SPEED * 0.5) / AXLE_LENGTH  # Angular velocity based on wheel speeds
    
    # Calculate turn duration
    turn_duration = int((angle / angular_velocity) * timestep * 100)

    # Set velocities to rotate back to initial direction
    leftMotor.setVelocity(-MAX_SPEED * 0.5)  # Left wheel moves backward
    rightMotor.setVelocity(MAX_SPEED * 0.5)  # Right wheel moves forward
    
    # Let the robot turn
    robot.step(turn_duration)

    # Stop after turning
    leftMotor.setVelocity(0.0)
    rightMotor.setVelocity(0.0)
    robot.step(100)

# Main loop
while robot.step(timestep) != -1:
    turn_and_move_to_goal()  # Step 1: Turn and move towards the goal in one step
    stop_and_turn_back()     # Step 2: Stop and turn back to initial orientation
    
    break