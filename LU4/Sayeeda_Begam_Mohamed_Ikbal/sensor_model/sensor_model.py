# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt


# Function to calculate the intersection points of a beam and a circle
def beam_circle_intersection(A, B, C, r):
    """
    Returns the intersection points of a line (representing the beam) and a circle.
    A, B, C are parameters of the line equation (Ax + By + C = 0).
    r is the radius of the circle.
    
    If there is no intersection, returns an empty array.
    If the beam touches the circle at one point, returns that point.
    If there are two intersection points, returns both points.
    """
    d_0 = np.abs(C)  # Perpendicular distance from the center of the circle to the line
    if d_0 > r:
        return np.array([])  # No intersection if the distance is greater than the radius
    x_0 = -A * C
    y_0 = -B * C
    if math.isclose(d_0, r):  # Tangent case: one intersection
        return np.array([[x_0, y_0]])
    d = np.sqrt(r * r - C * C)  # Distance from the center to the intersection points
    # Calculate the two intersection points
    x_1 = x_0 + B * d
    y_1 = y_0 - A * d
    x_2 = x_0 - B * d
    y_2 = y_0 + A * d
    return np.array([[x_1, y_1], [x_2, y_2]])


# Function to calculate the distance to the closest intersection between a beam and circles
def distance_to_closest_intersection(x, y, theta, circles):
    """
    Returns the distance to the closest intersection between a beam and a set of circles.
    
    x, y: The starting point of the beam (robot's position).
    theta: The direction of the beam (angle in radians).
    circles: A list of circles, where each circle is represented by [x_c, y_c, r] (center and radius).
    
    If no intersection occurs, returns a large value (inf).
    """
    beam_dir_x = np.cos(theta)  # x-component of the beam direction vector
    beam_dir_y = np.sin(theta)  # y-component of the beam direction vector
    min_dist = float('inf')  # Initialize the minimum distance to infinity
    
    # Iterate over all circles to check for intersections
    for circle in circles:
        # Shift the coordinates to the circle's center for easier calculations
        x_shifted = x - circle[0]
        y_shifted = y - circle[1]
        
        # Calculate the normal vector (A, B) to the beam
        A = beam_dir_y
        B = -beam_dir_x
        C = -(A * x_shifted + B * y_shifted)  # Calculate the line equation constant
        
        # Find intersection points of the beam and circle
        intersections = beam_circle_intersection(A, B, C, circle[2])
        
        for isec in intersections:
            # Calculate the dot product to check if the intersection is in front of the robot
            dot_prod = (isec[0] - x_shifted) * beam_dir_x + (isec[1] - y_shifted) * beam_dir_y
            if dot_prod > 0.0:
                # Calculate the distance to the intersection point
                dist = np.sqrt(np.square(isec[0] - x_shifted) + np.square(isec[1] - y_shifted))
                # Update the minimum distance if this is closer
                if dist < min_dist:
                    min_dist = dist

    return min_dist


# Function to calculate the normalizer for the probability function
def normalizer(z_exp, b, z_max):
    """
    Returns the normalizer value for the hit-probability function.
    
    z_exp: The expected range (in cm).
    b: The variance of the measurement noise.
    z_max: The maximum range (in cm).
    
    The normalizer scales the probability function such that it integrates to 1 over the possible range.
    """
    std_dev = np.sqrt(b)  # Standard deviation from the variance
    # Calculate the cumulative probability density function (CDF) for the range 0 to z_max
    cdf_diff = norm(z_exp, std_dev).cdf(z_max) - norm(z_exp, std_dev).cdf(0.0)
    return 1.0 / cdf_diff  # Return the normalizing factor


# Main function to calculate the scan probability
def beam_based_model(z_scan, z_scan_exp, b, z_max):
    """
    Calculates the probability of a scan based on the simplified beam-based model.
    
    z_scan: The measured ranges (in cm) from the sensor.
    z_scan_exp: The expected ranges (in cm) based on the environment.
    b: The variance of the measurement noise.
    z_max: The maximum measurable range (in cm).
    
    Returns the total probability for the scan.
    """
    prob_scan = 1.0  # Initialize the total probability to 1.0
    # Iterate over all beams (scan points)
    for i in range(z_scan.size):
        if z_scan[i] > z_max:
            continue  # Skip if the measured range exceeds the maximum range
        eta = normalizer(z_scan_exp[i], b, z_max)  # Calculate the normalizer for this beam
        # Calculate the probability of the measurement based on the Gaussian distribution
        prob_z = (eta / np.sqrt(2.0 * np.pi * b)) * np.exp(-0.5 * np.square(z_scan[i] - z_scan_exp[i]) / b)
        prob_scan *= prob_z  # Multiply the individual beam probability to the total probability
    
    return prob_scan


# Main driver code
def main():
    # Define the circles (obstacles) in the map
    circles = np.array([[3.0, 0.0, 0.5], [4.0, 1.0, 0.8], [5.0, 0.0, 0.5], [0.7, -1.3, 0.5]])
    
    # Define the robot's pose (position and orientation)
    pose = np.array([1.0, 0.0, 0.0])  # Robot at (1, 0) with orientation 0 radians (facing along x-axis)
    
    # Define the beam directions (angles) from -90 to +90 degrees (21 beams)
    beam_directions = np.linspace(-np.pi / 2, np.pi / 2, 21)
    
    # Load the measured scan data (range readings) from a file
    z_scan = np.load('z_scan.npy')  # Measured range data
    
    # Compute the expected ranges for each beam based on the circle intersections
    z_scan_exp = np.zeros(beam_directions.shape)
    for i in range(beam_directions.size):
        z_scan_exp[i] = distance_to_closest_intersection(pose[0], pose[1], beam_directions[i], circles)

    # Set the maximum range (in cm) and the measurement noise variance (b)
    z_max = 1000.0  # Max range in cm
    b = 1.0         # Variance of the measurement noise

    # Compute the scan probability using the beam-based model
    prob = beam_based_model(z_scan * 100.0, z_scan_exp * 100.0, b, z_max * 100.0)  # Convert to cm
    print("The scan probability is", prob)

    ########### Visualization of the robot and environment ################
    # Set up the plot to visualize the environment and scan data
    plt.axes().set_aspect('equal')  # Equal scaling of x and y axes
    plt.xlim([-0, 6])  # Set x-axis limits
    plt.ylim([-2, 2])  # Set y-axis limits
    plt.plot(pose[0], pose[1], "bo")  # Plot the robot's position as a blue dot

    fig = plt.gcf()
    axes = fig.gca()

    # Plot each beam as a dashed red line
    for i in range(beam_directions.size):
        theta = beam_directions[i]
        x_points = [pose[0], pose[0] + 10 * np.cos(theta)]
        y_points = [pose[1], pose[1] + 10 * np.sin(theta)]
        plt.plot(x_points, y_points, linestyle='dashed', color='red', zorder=0)

    # Plot each circle (obstacle) as a black circle
    for circle in circles:
        circle_plot = plt.Circle((circle[0], circle[1]), radius=circle[2], color='black', zorder=1)
        axes.add_patch(circle_plot)

    # Plot the expected intersection points as red dots
    for i in range(beam_directions.size):
        if z_scan_exp[i] > z_max:
            continue  # Skip if the expected range exceeds the max range
        theta = beam_directions[i]
        hit_x = pose[0] + np.cos(theta) * z_scan_exp[i]
        hit_y = pose[1] + np.sin(theta) * z_scan_exp[i]
        plt.plot(hit_x, hit_y, "ro")  # Plot expected hits as red dots

    # Add labels and display the plot
    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.show()

    # Optionally save the scan data and expected scan data
    # np.save('z_scan_exp', z_scan_exp)
    # np.save('z_scan', z_scan)


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
