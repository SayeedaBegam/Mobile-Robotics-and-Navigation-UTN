import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data
from matplotlib.patches import Ellipse

#plot preferences, interactive plotting mode
fig = plt.figure()
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

def plot_state(mu, sigma, landmarks, map_limits):
    # Visualizes the state of the kalman filter.
    #
    # Displays the mean and standard deviation of the belief,
    # the state covariance sigma and the position of the 
    # landmarks.

    # landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean of belief as current estimate
    estimated_pose = mu

    #calculate and plot covariance ellipse
    covariance = sigma[0:2,0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    #get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:,max_ind]
    max_eigval = eigenvals[max_ind]

    #get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigvec = eigenvecs[:,min_ind]
    min_eigval = eigenvals[min_ind]

    #chi-square value for sigma confidence interval
    chisquare_scale = 2.2789  

    #calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale*max_eigval)
    height = 2 * np.sqrt(chisquare_scale*min_eigval)
    angle = np.arctan2(max_eigvec[1],max_eigvec[0])

    #generate covariance ellipse
    ell = Ellipse(xy=[estimated_pose[0],estimated_pose[1]], width=width, height=height, angle=angle/np.pi*180)
    ell.set_alpha(0.25)

    # plot filter state and covariance
    plt.clf()
    plt.gca().add_artist(ell)
    plt.plot(lx, ly, 'bo',markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy')
    plt.axis(map_limits)
    
    plt.pause(0.01)

def prediction_step(odometry, mu, sigma):
    # Updates the belief, i.e., mu and sigma, according to the motion 
    # model
    # 
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 
    
    x = mu[0]
    y = mu[1]
    theta = mu[2]

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    '''your code here'''
    '''***        ***'''

    # Adding the two rotations and translation to update the pose (x, y, theta).
    theta = theta + delta_rot1 + delta_rot2
    x = x + delta_trans * np.cos(theta + delta_rot1)
    y = y + delta_trans * np.sin(theta + delta_rot1)

    # New state estimate
    mu = np.array([x, y, theta])

    # Computing the Jacobian (G_t) of the motion model, which describes how the motion affects uncertainty.
    # G_t measures the partial derivatives of the motion model with respect to the state variables.
    G = np.array([[1, 0, -delta_trans * np.sin(theta + delta_rot1)],
                  [0, 1, delta_trans * np.cos(theta + delta_rot1)],
                  [0, 0, 1]])

   
    # Higher values indicate less confidence in motion readings.
    std_rot1 = 0.1  # Standard deviation for the first rotation
    std_trans = 0.1  # Standard deviation for the translation
    std_rot2 = 0.1  # Standard deviation for the second rotation
    R = np.array([[std_rot1**2, 0, 0], 
                  [0, std_trans**2, 0], 
                  [0, 0, std_rot2**2]])

    # Update the covariance matrix (sigma) based on the motion model.
    # Incorporating the Jacobian and process noise inorder to predict the uncertainty.
    sigma = np.dot(np.dot(G, sigma), G.T) + R

    return mu, sigma


def correction_step(sensor_data, mu, sigma, landmarks):
    # updates the belief, i.e., mu and sigma, according to the
    # sensor model
    # 
    # The employed sensor model is range-only
    #
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 
    landmarks = {key: list(map(float, value)) for key, value in landmarks.items()}
    x = mu[0]
    y = mu[1]
    theta = mu[2]

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    '''your code here'''
    '''***        ***'''
    # Computing the expected range measurements (z) for each observed landmark based on the robot's current state.
    z = []
    for i in range(len(ids)):
        landmark = landmarks[ids[i]]  # Get the coordinates of the current landmark
        dx = float(landmark[0] - x)  # Calculate the x difference
        dy = float(landmark[1] - y)  # Calculate the y difference
        dist = np.sqrt(dx**2 + dy**2)  # Computing the Euclidean distance to the landmark
        z.append(dist)  # Append the distance to the expected measurements

    z = np.array(z)  # Convert the expected measurements list to a numpy array

    # Compute the measurement residual\innovation, which is the difference between observed and expected measurements.
    y_residual = ranges - z

    # Computing the Jacobian of the sensor model (H_t), which describes how each landmark measurement relates to the robot's state.
    H = np.zeros((len(ids), 3))
    for i in range(len(ids)):
        landmark = landmarks[ids[i]]  # Get the coordinates of the current landmark
        dx = float(landmark[0] - x)  # Calculate the x difference
        dy = float(landmark[1] - y)  # Calculate the y difference
        dist = np.sqrt(dx**2 + dy**2)  # Computing the Euclidean distance to the landmark

        # Accumulate the  Jacobian matrix with the partial derivatives of the range measurement w.r.t state variables
        H[i, 0] = -dx / dist
        H[i, 1] = -dy / dist
        H[i, 2] = 0  # Orientation does not directly affect the range measurement

    # Define the measurement noise covariance matrix (R), modeling uncertainty in the sensor readings.
    std_range = 0.1  # Standard deviation for range measurements
    R = np.eye(len(ids)) * std_range**2

    # Compute the Kalman gain (K), which determines how much the sensor measurement should influence the state update.
    S = np.dot(np.dot(H, sigma), H.T) + R  # Measurement covariance
    K = np.dot(np.dot(sigma, H.T), np.linalg.inv(S))  # Kalman gain computation

    # Update the state estimate (mu) using the Kalman gain and the measurement residual.
    mu = mu + np.dot(K, y_residual)

    # Update the covariance estimate (sigma) to reflect the reduced uncertainty after incorporating sensor data.
    sigma = np.dot(np.eye(3) - np.dot(K, H), sigma)

    return mu, sigma

def main():
    # implementation of an extended Kalman filter for robot pose estimation

    print ("Reading landmark positions")
    landmarks = read_world("/home/sayeedabegam/utn/sem1/mobrob/Asmt_7/kf_framework-1/kf_framework/data/world.dat")

    print ("Reading sensor data")
    sensor_readings = read_sensor_data("/home/sayeedabegam/utn/sem1/mobrob/Asmt_7/kf_framework-1/kf_framework/data/sensor_data.dat")

    #initialize belief
    mu = [0.0, 0.0, 0.0]
    sigma = np.array([[1.0, 0.0, 0.0],\
                      [0.0, 1.0, 0.0],\
                      [0.0, 0.0, 1.0]])

    map_limits = [-1, 12, -1, 10]

    #run kalman filter
    for timestep in range(len(sensor_readings)//2):

        #plot the current state
        plot_state(mu, sigma, landmarks, map_limits)

        #perform prediction step
        mu, sigma = prediction_step(sensor_readings[timestep,'odometry'], mu, sigma)

        #perform correction step
        mu, sigma = correction_step(sensor_readings[timestep, 'sensor'], mu, sigma, landmarks)

    #plt.show('hold')
    plt.show()

if __name__ == "__main__":
    main()
