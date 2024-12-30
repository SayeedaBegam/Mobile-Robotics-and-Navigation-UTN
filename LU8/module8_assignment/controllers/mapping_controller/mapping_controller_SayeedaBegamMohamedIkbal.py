"""mapping_controller controller."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math
import sys
import copy
from abc import ABC, abstractmethod

from controller import Robot
from controller import Supervisor
from controller import Keyboard

MAX_SPEED = 12.3

        
############################### HELPER FUNCTIONS ##################################


# Normalizes the angle theta in range (-pi, pi)
def normalize_angle(theta):
    if (theta > np.pi):
        return theta - 2*np.pi
    if (theta < -np.pi):
        return theta + 2*np.pi
    return theta


def log_odds(p):
    return np.log(p/(1-p))
    
    
def from_log_odds(l):
    return 1.0 - 1/(1.0 + np.exp(l))
    
    
from_log_odds = np.vectorize(from_log_odds)


norm_angle_arr = np.vectorize(normalize_angle)


def get_curr_pose(trans_field, rot_field):
    values = trans_field.getSFVec3f()
    rob_theta = np.sign(rot_field.getSFRotation()[2])*rot_field.getSFRotation()[3]
    rob_x = values[0]
    rob_y = values[1]
    return [rob_x, rob_y, rob_theta]
    

def get_pose_delta(last_pose, curr_pose):
    trans_delta = np.sqrt((last_pose[0]-curr_pose[0])**2 + (last_pose[1]-curr_pose[1])**2)
    theta_delta = abs(normalize_angle(last_pose[2]-curr_pose[2]))
    return trans_delta, theta_delta
    

def velFromKeyboard(keyboard):
    turn_base = 3.0
    linear_base = 6.0
    vel_left = 0.0
    vel_right = 0.0
    key = keyboard.getKey()
    while (key != -1):
        if (key==Keyboard.UP):
            vel_left += linear_base
            vel_right += linear_base
        if (key==Keyboard.DOWN):
            vel_left += -linear_base
            vel_right += -linear_base
        if (key==Keyboard.LEFT):
            vel_left += -turn_base
            vel_right += turn_base
        if (key==Keyboard.RIGHT):
            vel_left += turn_base
            vel_right += -turn_base
        key = keyboard.getKey()
    return vel_left, vel_right


############################### GridMapBase ##################################    
    

class GridMapBase(ABC):
    
    def __init__(self, fs_x, fs_y, cell_size, p_prior):
        self.cell_size = cell_size
        self.p_prior = p_prior
        n_cells_half_x = int((fs_x/2.0)//cell_size) + 1
        n_cells_half_y = int((fs_y/2.0)//cell_size) + 1
        self.grid_size = [2*n_cells_half_x, 2*n_cells_half_y]
        self.grid_origin = [-n_cells_half_x*cell_size,-n_cells_half_y*cell_size]
        
        self.grid = np.empty((self.grid_size[0], self.grid_size[1]))
        self.grid.fill(p_prior)
        
        self.fig, ax = plt.subplots()
        self.im = ax.imshow(self.grid*255,cmap='gray', vmin = 0, vmax= 255)
        self.fig.show()
        
    @abstractmethod    
    def update_map(self, pose, scan):
        pass
        
    @abstractmethod   
    def vis_map(self):
        pass
        
        
    # adapted from http://playtechs.blogspot.com/2007/03/raytracing-on-grid.html
    # traverses cells in a 2D grid along a ray
    # start is the starting point of the ray as [x,y] in world coordinates
    # direction is the normalized direction vector of the ray
    # returns visited, intersections
    # visited is a 2D array, where each element is the index of a visited cell,
    # in order of traversal
    # intersections is a1D array that contains the distances to the visited cells 
    # along the ray
    def ray_traversal(self, start, direction):
        x0 = (start[0] - self.grid_origin[0])/self.cell_size
        y0 = (start[1] - self.grid_origin[1])/self.cell_size
        
        dx = abs(direction[0])
        dy = abs(direction[1])
        
        x = int(x0)
        y = int(y0)
        
        
        x_inc = 0
        y_inc = 0
        error = 0.0
        t_x = 0.0
        
        
        if (dx == 0):
            x_inc = 0
            error = float('inf')
            t_x = float('inf')
        elif (direction[0] > 0):
            x_inc = 1
            error = (math.floor(x0) + 1 - x0) * dy
            t_x = (math.floor(x0) + 1 - x0) / dx
        else:
            x_inc = -1
            error = (x0 - math.floor(x0)) * dy
            t_x = (x0 - math.floor(x0)) / dx
            
        t = abs(error)
        t_y = 0.0
        
        if (dy == 0):
            y_inc = 0
            error -= float('inf')
            t_y = float('inf')
            
        elif (direction[1] > 0):
            y_inc = 1
            error -= (math.floor(y0) + 1 - y0) * dx
            t_y = (math.floor(y0) + 1 - y0) / dy
        else:
            y_inc = -1
            error -= (y0 - math.floor(y0)) * dx
            t_y = (y0 - math.floor(y0)) / dy
            
        t_x = abs(t_x)
        t_y = abs(t_y)
        
        visited = []
        intersections = []
        t = 0.0
        while (x >= 0 and y >=0 and x < self.grid_size[0] and y < self.grid_size[1]):
            visited.append([y, x]);
            
            if (error > 0):
                y += y_inc
                error -= dx
                t_y += 1/dy
                t = t_y
            else:
                x += x_inc
                error += dy
                t_x += 1/dx
                t = t_x
            intersections.append(t*self.cell_size)
        
        return visited, intersections


############################### OccupancyMap ################################## 

    
class OccupancyMap(GridMapBase):

    def __init__(self, fs_x, fs_y, cell_size, p_prior):
        super().__init__(fs_x, fs_y, cell_size, p_prior)
        
        self.p_occ = 0.7
        self.p_free = 0.3
        self.r = 0.2
        
        self.grid.fill(-log_odds(p_prior))
        
     
    def vis_map(self):
        flipped_grid = np.flipud(self.grid)
        flipped_grid = 1.0 - from_log_odds(flipped_grid)
    
        self.im.set_data(flipped_grid*255)
        self.fig.canvas.draw()
            
        plt.pause(0.01)   
        
    # The inverse sensor model for LiDAR measurements
    # z is the range measurement from the LiDAR sensor (i.e., how far the sensor reads)
    # val is the distance to the particular cell that we are evaluating (i.e., how far the robot has scanned along the grid)
    # This function calculates the log-odds occupancy probability for a given cell in the grid

    def inverse_sensor_model(self, val, z):
        """
        Calculates the log-odds of a cell being occupied or free based on LiDAR distance measurements.
    
        Parameters:
         - val: The distance to the cell that the LiDAR beam intersects.
        - z: The range measurement obtained from the LiDAR sensor at a specific angle.
    
         Returns:
        - The log-odds value of the cell's occupancy probability.
        """
       
        # Check if the cell is in free space
        if val < (z - self.r):  # If the distance is significantly smaller than the LiDAR range (considered free)
            return log_odds(self.p_free)  # Return the log-odds for free space
        # Check if the cell is within the LiDAR range (occupied space)
        elif z - self.r <= val <= z + self.r:  # If the distance is within the LiDAR measurement (considered occupied)
            return log_odds(self.p_occ)  # Return the log-odds for occupied space
        # Otherwise, the cell is in an unknown state
        else:
            return log_odds(self.p_prior)  # Return the log-odds for prior information (no update)
            

    # Function to update the occupancy grid map using LiDAR scan data
    # pose is the current pose of the robot (x, y, theta)
    # scan is the current LiDAR scan containing range measurements
    def update_map(self, pose, scan):
        """
        Updates the occupancy grid based on the robot's current position and its LiDAR scan.
    
        Parameters:
        - pose: A tuple (x, y, theta), where x and y are the robot's position, and theta is its orientation.
        - scan: A list of range measurements from the LiDAR sensor, where each value represents the distance to the closest obstacle.
        """
  
        # Extract robot's position and orientation from the pose
        robot_x, robot_y, robot_theta = pose
    
        # Iterate over each range measurement (scan data)
        for i, z in enumerate(scan):
            # Calculate the angle for this particular LiDAR measurement
            # The angle is adjusted by the robot's current orientation and the position in the scan data
            angle = robot_theta + i * (2 * np.pi / len(scan)) - np.pi
        
            # Calculate the direction vector of the LiDAR beam based on this angle
            # The direction vector tells us the unit vector pointing in the direction the LiDAR is scanning
            direction = [math.cos(angle), math.sin(angle)]
        
            # Use ray tracing to find the grid cells the LiDAR ray intersects
            # `ray_traversal` returns two values: 
            # - visited: a list of cells (x, y) that are crossed by the ray
            # - intersections: a list of distances to those cells (how far the ray travels before hitting the cell)
            visited, intersections = self.ray_traversal([robot_x, robot_y], direction)
        
        # For each visited cell, update its occupancy probability
        for idx, cell in enumerate(visited):
            # Extract the coordinates of the visited cell
            cellx, celly = cell
            if 0 <= cellx <self.grid_size[1] and 0 <= celly < self.grid_size[0]: 
            # Get the distance to the cell from the intersections list
            # If the index exceeds the available intersections, assign the distance to infinity
                distance = intersections[idx] if idx < len(intersections) else float('inf')
            
            # Update the grid at the (cell_x, cell_y) position using the inverse sensor model
            # The inverse sensor model returns the log-odds of the cell's occupancy based on its distance
                self.grid[cellx, celly] += self.inverse_sensor_model(distance, z)

                
                
############################### ReflectanceMap ################################## 
        
        
class ReflectanceMap(GridMapBase):

    def __init__(self, fs_x, fs_y, cell_size, p_prior):
        super().__init__(fs_x, fs_y, cell_size, p_prior)
        
        self.hit_map = np.zeros((self.grid_size[0], self.grid_size[1]))
        self.miss_map = np.zeros((self.grid_size[0], self.grid_size[1]))
        

    def vis_map(self):
        prob_map = np.full_like(self.hit_map, self.p_prior)
        x_dim, y_dim = prob_map.shape
        for x in range(x_dim):
            for y in range(y_dim):
                h = self.hit_map[x][y]
                m = self.miss_map[x][y]
                if h + m > 0:
                    prob_map[x][y] = h / (h + m)
        flipped_grid = 1.0 - np.flipud(prob_map)
    
        self.im.set_data(flipped_grid*255)
        self.fig.canvas.draw()
         
        plt.pause(0.01)    
    
                
    # updates the reflectance grid map
    # pose is the current pose of the robot
    # scan is the current range measurement              
    def update_map(self, pose, scan):        
        # your code here
        
        """
        Updates the reflectance map using the robot's current pose (x, y, theta) and LiDAR scan data.
        The LiDAR scan gives distance measurements that are used to determine which cells are
        occupied (hit) and which cells are free (missed).
        
        - The 'hit_map' is updated where an obstacle is detected.
        - The 'miss_map' is updated for cells that the LiDAR passes through but no obstacle is detected.
        
        Args:
        - pose: A tuple (x, y, theta) representing the robot's position and orientation.
        - scan: A list of distance measurements from the LiDAR sensor.
        """
      
        
        # Extract the robot's current position and orientation from the pose
        robot_x, robot_y, robot_theta = pose

        # Iterate over each LiDAR scan measurement (representing the distance at a given angle)
        for i, z in enumerate(scan):
            # Calculate the angle of the LiDAR measurement relative to the robot's orientation
            # The LiDAR sensor typically scans 360 degrees, but the scan here is divided into multiple steps.
            angle = robot_theta + i * (2 * np.pi / len(scan)) - np.pi  # Calculate angle based on scan index

            # Create a direction vector for the LiDAR ray (unit vector pointing in the scan direction)
            direction = [math.cos(angle), math.sin(angle)]

            # Perform ray tracing to find the cells that the LiDAR ray intersects.
            # `ray_traversal` returns:
            # - visited: List of grid cells that the ray passed through.
            # - intersections: List of distances from the start to each visited cell.
            visited, intersections = self.ray_traversal([robot_x, robot_y], direction)

            # If the LiDAR ray intersects any cells, update the hit and miss maps accordingly
            if len(visited) > 0:
                # The last visited cell is considered a "hit" (where the LiDAR ray hits an obstacle)
                hit_cell = visited[-1]
                if 0 <= hit_cell[0] < self.grid_size[0] and 0 <= hit_cell[1] < self.grid_size[1]:
                    self.hit_map[hit_cell[0], hit_cell[1]] += 1  # Increment the hit count for this cell


                # All other cells in the path are considered "misses" (free space)
                for cell in visited[:-1]:
                    if 0 <= cell[0] < self.grid_size[0] and 0 <= cell[1] < self.grid_size[1]:
                        self.miss_map[cell[0], cell[1]] += 1 # Increment the miss count for these cells
                     
     
                    
                    

########################################### main ################################


def main():
    # create the Robot instance.
    robot = Supervisor()
    robot_node = robot.getFromDef("Pioneer3dx")

    # robot pose translation and rotation objects
    trans_field = robot_node.getField("translation")
    rot_field = robot_node.getField("rotation")
    
    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # init keyboard readings
    keyboard = Keyboard()
    keyboard.enable(10)
    
    # get wheel motor controllers
    leftMotor = robot.getDevice('left wheel')
    rightMotor = robot.getDevice('right wheel')
    leftMotor.setPosition(float('inf'))
    rightMotor.setPosition(float('inf'))

    # initialize wheel velocities
    leftMotor.setVelocity(0.0)
    rightMotor.setVelocity(0.0)
    
    # get and enable lidar
    lidar = robot.getDevice('Sick LMS 291')
    lidar.enable(60)
    lidar.enablePointCloud()
    
    # get map limits
    ground_node = robot.getFromDef("RectangleArena")
    floor_size_field = ground_node.getField("floorSize")
    fs_x = floor_size_field.getSFVec2f()[0]
    fs_y = floor_size_field.getSFVec2f()[1]
    

    # last pose used for odometry calculations
    last_pose = get_curr_pose(trans_field, rot_field)
 
    # translation threshold for odometry calculation
    trans_thr = 0.1
    
    # choose which map to use
    
    map = OccupancyMap(fs_x, fs_y, cell_size=0.05, p_prior=0.5)
    #map = ReflectanceMap(fs_x, fs_y, cell_size=0.05, p_prior=0.5)
    
    
    while robot.step(timestep) != -1:
        # key controls
        vel_left, vel_right = velFromKeyboard(keyboard)
        leftMotor.setVelocity(vel_left)
        rightMotor.setVelocity(vel_right)

        # read robot pose and compute difference to last used pose
        curr_pose = get_curr_pose(trans_field, rot_field)
        trans_delta, theta_delta = get_pose_delta(last_pose, curr_pose)

        # skip until translation change is big enough
        if (trans_delta < trans_thr):
            continue
         
        # get current lidar measurements
        scan = lidar.getRangeImage()
        # we use a reversed scan order in the sensor model
        scan.reverse()
        
        # update map 
        map.update_map(curr_pose, scan)
        # visualize map
        map.vis_map()

        last_pose = curr_pose

    plt.show('hold', visited)
    plt.pause(0.001)


if __name__ == "__main__":
    main()
  

#Reference : Probablistic Robitics Book Page : 286, 288.