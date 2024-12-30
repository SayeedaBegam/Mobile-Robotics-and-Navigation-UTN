import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt

# Function to calculate the intersection of the beam with a circle
def beam_circle_intersection(A, B, C, r):
    d_0 = np.abs(C)
    if d_0 > r:
        return np.array([])
    x_0 = -A * C
    y_0 = -B * C
    if math.isclose(d_0, r):
        return np.array([[x_0, y_0]])
    d = np.sqrt(r * r - C * C)
    x_1 = x_0 + B * d
    y_1 = y_0 - A * d
    x_2 = x_0 - B * d
    y_2 = y_0 + A * d
    return np.array([[x_1, y_1], [x_2, y_2]])

# Function to calculate the distance to the closest intersection between a beam and the circles
def distance_to_closest_intersection(x, y, theta, circles):
    beam_dir_x = np.cos(theta)
    beam_dir_y = np.sin(theta)
    min_dist = float('inf')
    for circle in circles:
        x_shifted = x - circle[0]
        y_shifted = y - circle[1]
        A = beam_dir_y
        B = -beam_dir_x
        C = -(A * x_shifted + B * y_shifted)
        intersections = beam_circle_intersection(A, B, C, circle[2])
        for isec in intersections:
            dot_prod = (isec[0] - x_shifted) * beam_dir_x + (isec[1] - y_shifted) * beam_dir_y
            if dot_prod > 0.0:
                dist = np.sqrt(np.square(isec[0] - x_shifted) + np.square(isec[1] - y_shifted))
                if dist < min_dist:
                    min_dist = dist
    return min_dist

# Function to calculate the normalizer for the hit-probability function
def normalizer(z_exp, b, z_max):
    std_dev = np.sqrt(b)
    cdf_max = norm(z_exp, std_dev).cdf(z_max)
    cdf_0 = norm(z_exp, std_dev).cdf(0.0)
    # If the difference between the CDFs is zero, we return 1.0 as a safe fallback
    if cdf_max - cdf_0 == 0:
        print(f"Warning: CDF difference is zero for z_exp={z_exp}, b={b}, z_max={z_max}")
        return 1.0
    return 1.0 / (cdf_max - cdf_0)

# Function to calculate the beam-based model probability
def beam_based_model(z_scan, z_scan_exp, b, z_max):
    prob_scan = 1.0
    for i in range(z_scan.size):
        if z_scan[i] > z_max:
            continue
        eta = normalizer(z_scan_exp[i], b, z_max)
        # Calculate the probability for the scan
        prob_z = (eta / np.sqrt(2.0 * np.pi * b)) * np.exp(-0.5 * np.square(z_scan[i] - z_scan_exp[i]) / b)

        # Ensure that prob_z is valid (between 0 and 1)
        if np.isnan(prob_z) or np.isinf(prob_z):
            print(f"Invalid probability at index {i}, z_scan[i]: {z_scan[i]}, z_scan_exp[i]: {z_scan_exp[i]}")
            prob_z = 0  # Set to 0 to avoid invalid values in the final probability

        # Multiply the probabilities for all scans
        prob_scan *= prob_z

    # Ensure that the final probability is between 0 and 1
    prob_scan = max(0.0, min(1.0, prob_scan))
    return prob_scan

# Main function to calculate and visualize the scan probability
def main():
    circles = np.array([[3.0, 0.0, 0.5], [4.0, 1.0, 0.8], [5.0, 0.0, 0.5], [0.7, -1.3, 0.5]])
    pose = np.array([1.0, 0.0, 0.0])
    beam_directions = np.linspace(-np.pi / 2, np.pi / 2, 21)
    
    # Generating random placeholder data for z_scan as it is not found
    z_scan = np.random.uniform(0.0, 10.0, beam_directions.size)  # Random placeholder scan data
    
    # Compute the expected ranges using the intersection function
    z_scan_exp = np.zeros(beam_directions.shape)
    for i in range(beam_directions.size):
        z_scan_exp[i] = distance_to_closest_intersection(pose[0], pose[1], beam_directions[i], circles)

    z_max = 10.0  # Maximum range (cm)
    b = 1.0  # Variance (could be adjusted to fit your model)
    
    # Compute the scan probability using the beam-based model
    prob = beam_based_model(z_scan * 100.0, z_scan_exp * 100.0, b, z_max * 100.0)
    print("The scan probability is:", prob)

    ########### Visualization #################################
    plt.axes().set_aspect('equal')
    plt.xlim([0, 6])
    plt.ylim([-2, 2])
    plt.plot(pose[0], pose[1], "bo")

    fig = plt.gcf()
    axes = fig.gca()
    for i in range(beam_directions.size):
        theta = beam_directions[i]
        x_points = [pose[0], pose[0] + 10 * np.cos(theta)]
        y_points = [pose[1], pose[1] + 10 * np.sin(theta)]
        plt.plot(x_points, y_points, linestyle='dashed', color='red', zorder=0)

    for circle in circles:
        circle_plot = plt.Circle((circle[0], circle[1]), radius=circle[2], color='black', zorder=1)
        axes.add_patch(circle_plot)

    for i in range(beam_directions.size):
        if z_scan_exp[i] > z_max:
            continue
        theta = beam_directions[i]
        hit_x = pose[0] + np.cos(theta) * z_scan_exp[i]
        hit_y = pose[1] + np.sin(theta) * z_scan_exp[i]
        plt.plot(hit_x, hit_y, "ro")

    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.show()

if __name__ == "__main__":
    main()
