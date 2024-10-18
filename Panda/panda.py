import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import e  # If 'e' is used in the code
from hyperparameters import (
    spread, noise, radius, delt, pert, b, sensor_range, n, maxv, attractive_gain, 
    repulsive_gain, collision_distance, clipping_power, seed, priority_type
)


# Global data array initialization
global_data_array = np.zeros((7, 13))

# Set the random seed for reproducibility
np.random.seed(seed)
print("Random seed set to:", seed)


# Function to clear all the output files from previous runs.

def clear_previous_output_files():

    """
    Deletes output files from previous runs if they exist.
    This includes 'time.csv', 'priority.csv', and 'avg.csv'.
    """

    for file_name in ['time.csv', 'priority.csv', 'avg.csv']:
        if os.path.isfile(file_name):
            os.remove(file_name)

# Clear output files at the start of the script
clear_previous_output_files()


# Function to initialize the starting and goal positions of UAVs
def initialize_uav_positions_and_goals(number_of_uavs, radius):

    start_positions = []
    goal_positions = []

    for i in range(1, number_of_uavs // 4 + 1):
        position1 = (np.cos(spread * (i / (number_of_uavs // 4))), np.sin(spread * (i / (number_of_uavs // 4))))
        position2 = (-position1[0], position1[1])
        position3 = (-position1[0], -position1[1])
        position4 = (position1[0], -position1[1])

        start_positions.extend([position1, position2, position3, position4])
        goal_positions.extend([position2, position1, position4, position3])

    start_positions = np.array(start_positions) * radius
    goal_positions = np.array(goal_positions) * radius
    print("start_positions: ", start_positions)
    print("goal_positions: ", goal_positions)
    return start_positions, goal_positions



def plot_initial_positions_and_goals(start_positions, goal_positions, number_of_uavs, radius):

    cmap = plt.get_cmap('hsv')
    colors = [cmap(i) for i in np.linspace(0, 1, len(start_positions))]

    # Create a plot
    fig, ax = plt.subplots(figsize=(20, 10))

    # Scatter plot for starting positions
    plt.scatter(np.array(start_positions)[:, 0], 
                np.array(start_positions)[:, 1], 
                c=colors, s=200)

    # Scatter plot for goal positions
    plt.scatter(1.2 * np.array(goal_positions)[:, 0], 
                1.2 * np.array(goal_positions)[:, 1], 
                c=colors, marker="s", s=200, alpha=0.5)

    # Set plot limits and labels
    plt.xlim(-1.2 * radius, 1.2 * radius)
    plt.ylim(-radius / 2, radius / 2)
    plt.xlabel('x', fontsize=24)
    plt.ylabel('y', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Draw circles to indicate boundaries
    plt.gca().add_patch(plt.Circle((0, 0), radius, fill=False))
    plt.gca().add_patch(plt.Circle((0, 0), 1.2 * radius, fill=False))

    # (Optional) Save the plot if needed
    #plt.savefig('/Users/jaskiratsingh/IISER-Bhopal/Comparison/PANDA/plots')



# Generator function to create initial setup for UAVs
def initialize_uav_properties(number_of_uavs, radius):
    """
    Generates initial properties for UAVs including position, velocity, and other parameters.

    Parameters:
    - number_of_uavs (int): The number of UAVs.

    Returns:
    - A tuple containing arrays for initial velocities, positions, goals, and other UAV attributes.
    """
    # Initialize arrays for velocity, position, and other attributes
    velocity_u = np.zeros((number_of_uavs, 2)) # if n=8 then 8 rows and 2 columns (x,y)
    velocity_v = np.zeros((number_of_uavs, 2))
    start_positions, goal_positions = initialize_uav_positions_and_goals(number_of_uavs, radius)  # Initialize positions
    acceleration = np.zeros((number_of_uavs, 2))
    completed_status = np.zeros(number_of_uavs)
    clipping_status = np.zeros(number_of_uavs)
    adjusted_max_velocity = maxv * np.ones(number_of_uavs) # adjusted_max_velocity = vmax
    alpha = np.ones(number_of_uavs)*0.3

    return velocity_u, velocity_v, start_positions, goal_positions, acceleration, completed_status, clipping_status, adjusted_max_velocity, alpha


def set_uav_priorities(number_of_uavs, priority_type):
    """
    Sets the priorities for each UAV based on the specified priority type.

    Parameters:
    - number_of_uavs (int): The number of UAVs.
    - priority_type (str): The type of priority to set ('Gaussian', 'Uniform', or 'Constant').

    Returns:
    - priority (np.array): Array of priorities for each UAV.
    """
    if priority_type == "Gaussian":
        #priority = np.random.normal(3, 1, number_of_uavs)
        priority = 3 * np.ones(number_of_uavs)
    elif priority_type == "Uniform":
        #priority = np.random.uniform(1, 6, number_of_uavs)
        priority = 3 * np.ones(number_of_uavs)
    else:
        priority = 3 * np.ones(number_of_uavs)

    # Save priorities to a file
    with open("priority.csv", 'a') as priority_file:
        priority_file.write(", ".join([str(x) for x in priority]) + '\n')

    return priority



# Function to check if a UAV has reached its goal
def reached_goal(uav_index, start_positions, goal_positions, radius, priority):
    """
    Checks if a UAV has reached its goal and records the time if it has.
    """
    global mission_completion

    if np.linalg.norm(start_positions[uav_index] - goal_positions[uav_index]) <= 5:
        if mission_completion[uav_index] == 0:  # UAV has not yet reached its goal
            # Record the time taken for the UAV to reach its goal
            mission_completion[uav_index] = time.time() - uav_start_times[uav_index]
        return True
    else:
        return False


# Function to check for potential collisions between UAVs
def collision(uav_i, uav_j, start_positions, velocity_v, velocity_u, collision_distance, sensor_range):
    """
    Checks if there is an imminent collision between two UAVs.

    Parameters:
    - uav_i, uav_j (int): Indices of the two UAVs being checked.
    - positions, velocities (np.array): Arrays of positions and velocities of UAVs.
    - collision_distance (float): The distance threshold for a collision.
    - sensor_range (float): The sensor range of the UAVs.

    Returns:
    - bool: True if a collision is imminent, False otherwise.
    """
    relative_velocity = velocity_v[uav_i] - velocity_v[uav_j]
    t = -np.dot(np.array(start_positions[uav_i]) - np.array(start_positions[uav_j]), relative_velocity) / (np.dot(relative_velocity, relative_velocity) + 0.000001)

    if t < 0:
        return False
    else:
        # projected_distance -> s_quared and dist -> actual distance
        projected_distance = np.dot(start_positions[uav_i] + velocity_v[uav_i] * t - start_positions[uav_j] - velocity_v[uav_j] * t,
                                    start_positions[uav_i] + velocity_v[uav_i] * t - start_positions[uav_j] - velocity_v[uav_j] * t)
        actual_distance_btw_2_uavs = np.linalg.norm(np.array(start_positions[uav_i]) - np.array(start_positions[uav_j]))

        if np.sqrt(projected_distance) < collision_distance and actual_distance_btw_2_uavs < sensor_range:
            return True
        else:
            return False


# Function to clip the velocity of UAVs
def clip_velocity(number_of_uavs):
    global velocity_v, clipping_status
    """
    Adjusts the velocities of UAVs to prevent high-speed collisions and ensure safe operation.

    This function scales down the velocity of each UAV if it exceeds a dynamically adjusted maximum velocity.
    This maximum velocity is determined based on the UAV's priority and the collision risk with other UAVs.

    Parameters:
    - number_of_uavs (int): The total number of UAVs in the simulation.

    Modifies:
    - velocity_v (global numpy.ndarray): The velocities of each UAV.
    - clipping_status (global numpy.ndarray): Indicates whether a UAV's velocity needs to be clipped.
    - adjusted_max_velocity (global numpy.ndarray): The adjusted maximum velocities for each UAV.
    """
    #print("----------------")
    for i in range(number_of_uavs):
        
        if clipping_status[i]:
            adjusted_max_velocity[i] = maxv*(priority[i]/max(in_collision[i]))**clipping_power
        else:
            adjusted_max_velocity[i] = maxv
        #vmax[i]=maxv
        if np.linalg.norm(velocity_v[i])>adjusted_max_velocity[i]:
            velocity_v[i] = adjusted_max_velocity[i]*velocity_v[i]/np.linalg.norm(velocity_v[i])



def perturb_velocity(velocity_v):
    """
    Perturbs the velocities of UAVs by a slight random amount.

    Parameters:
    - velocities (np.array): Array of current velocities of UAVs.

    Modifies the velocities array in place.
    """
    for i, vel in enumerate(velocity_v):
        x = vel[0]
        y = vel[0]
        delta_theta = np.random.normal(0, np.pi / 2 ** 10.5)
        theta = np.arctan2(vel[1], vel[0])
        theta_perturbed = theta + delta_theta

        # Calculate the perturbed vector components using the perturbed angle
        x_perturbed = np.cos(theta_perturbed) * np.sqrt(x**2 + y**2)
        y_perturbed = np.sin(theta_perturbed) * np.sqrt(x**2 + y**2)

        #magnitude = np.sqrt(vel[0] ** 2 + vel[1] ** 2)

        velocity_v[i] = np.array([x_perturbed, y_perturbed])



def rotate_vector(vector, angle):
    """
    Rotates a 2D vector by a given angle.

    Parameters:
    - vector (np.array): The vector to be rotated.
    - angle (float): The angle to rotate the vector, in radians.

    Returns:
    - np.array: The rotated vector.
    """
    x, y = vector[0], vector[1]
    cos_theta, sin_theta = np.cos(angle), np.sin(angle)
    return np.array([x * cos_theta - y * sin_theta, x * sin_theta + y * cos_theta])



if __name__ == "__main__":

    
    if 'number_of_uavs' not in globals():
        number_of_uavs = 20  # Default number of UAVs
    #if 'radius' not in globals():
        #radius = 50  # Default radius
    #if 'delt' not in globals():
        #delt = 0.01  # Default time delta
    #if 'maxv' not in globals():
        #maxv = 3.0  # Default maximum velocity

    min_dist = np.inf*np.ones((number_of_uavs, number_of_uavs))
    avg_dist = np.zeros(number_of_uavs)

    mission_completion = [0]*number_of_uavs
    check = np.ones(number_of_uavs)

    # Initialize UAV positions and goals
    start_positions, goal_positions = initialize_uav_positions_and_goals(number_of_uavs, radius)

    # Plot initial positions and goals (optional)
    plot_initial_positions_and_goals(start_positions, goal_positions, number_of_uavs, radius)

    velocity_u, velocity_v, start_positions, goal_positions, acceleration, completed_status, clipping_status, adjusted_max_velocity, alpha = initialize_uav_properties(number_of_uavs, radius)

    # Initialize start times for each UAV at the beginning of the simulation
    uav_start_times = {uav_id: time.time() for uav_id in range(number_of_uavs)}

    priority_type = "Gaussian"
    priority = set_uav_priorities(number_of_uavs, priority_type)  # or 'Uniform', 'Constant'
    collision_priorities = priority
    in_collision = [[i] for i in priority]

    # File initialization for output
    file = open('out.csv', 'w')
    velocity_file = open('vel.csv', 'w')
    time_file = open('time.csv', 'a')
    acc_file = open('acc.csv', 'w')
    path_file = open('path_lengths.csv', 'w')

    # Write headers for the files
    header = ','.join([f"{i}x,{i}y" for i in range(number_of_uavs)])
    file.write(header + "\n")
    velocity_file.write(header + "\n")
    acc_file.write(header + "\n")
    path_file.write("UAV ID,Path Length\n")

    # Initialize an array to store the path length for each UAV
    path_lengths = np.zeros(number_of_uavs)

    # Initialize an array to store the previous positions of UAVs for distance calculation
    previous_positions = np.copy(start_positions)


    r = 0
    #Main Simulation loop
    while not np.array_equal(completed_status, check):

        r += 1
        clipping_status = np.zeros(number_of_uavs)
        acceleration = np.zeros((number_of_uavs,2))
        for i in range(number_of_uavs):
            if reached_goal(i, start_positions, goal_positions, radius, priority):
                if completed_status[i] != 1:
                    completed_status[i] = 1
                    acceleration[i] = 0
                    velocity_v[i] = 0
                    mission_completion[i]=r
                    path_file.write(f"{i},{path_lengths[i]}\n")
                    print(f"UAV {i} has reached its goal.")

            else:
                #alpha[i] = 1
                distance_traveled = np.linalg.norm(start_positions[i] - previous_positions[i])
                path_lengths[i] += distance_traveled
                # calculates the attractive force
                dist_of_uav_to_goal = np.linalg.norm(np.array(goal_positions[i]) - np.array(start_positions[i]))
                #print(f"UAV {i}: Distance to Goal: {dist_of_uav_to_goal}, Collision Avoidance: {colliding}")
                attractive_force = 2*(1-e**(-2*dist_of_uav_to_goal**2))*(np.array(goal_positions[i])-np.array(start_positions[i]))/dist_of_uav_to_goal
                #attractive_force = attractive_gain*np.array(goal[i]-pos[i])/dist_to_goal
                acceleration[i] = np.array(attractive_force)
                #print("a",i,acceleration[i])

                # Calculate Repulsive Force
                colliding=False
                for j in range(number_of_uavs):
                    dist_of_uav1_from_uav2 = np.linalg.norm(np.array(start_positions[j]) - np.array(start_positions[i])) # Calculating distance of UAV j from UAV i
                    if dist_of_uav1_from_uav2 < min_dist[i][j]:
                        min_dist[i][j] = dist_of_uav1_from_uav2
                    if j != i: # checking if UAV is not checking collision with itself
                        if collision(i, j, start_positions, velocity_v, velocity_u, collision_distance, sensor_range):
                            colliding=True
                            #print("collision",i,j)
                            #dist = np.linalg.norm(pos[j]-pos[i])
                            repulsive_force = (priority[j]/priority[i])*repulsive_gain*(e**(-b*(dist_of_uav1_from_uav2-collision_distance)**2))*(np.array(start_positions[i]) - np.array(start_positions[j]))/dist_of_uav1_from_uav2
                            repulsive_force = rotate_vector(repulsive_force, np.pi/2) # rotating the force vector
                            #print(i,j,dist,rep,a[i])
                            acceleration[i] += repulsive_force


                if colliding: #The clip array is used to keep track of which UAVs need to adjust their movement to avoid collisions. By setting clip[i] = 1, the code is marking UAV i for such an adjustment.
                    clipping_status[i]=1
                    #print(clipping_status)

                #print(f"UAV {i}: Position {start_positions[i]}, Velocity {velocity_v[i]}, Completed {completed_status[i]}")

            previous_positions = np.copy(start_positions)  # Update for the next iteration

        if pert:
            perturb_velocity(velocity_v)
        velocity_v = velocity_v + acceleration*delt
        clip_velocity(number_of_uavs)
        start_positions = start_positions + velocity_v*delt

        file.write(",".join([str(x) for x in start_positions.flatten()])+"\n")
        velocity_file.write(", ".join([str(x) for x in velocity_v.flatten()])+"\n")
        acc_file.write(", ".join([str(x) for x in acceleration.flatten()])+"\n")



        #plt.clf()
        """
        cmap = plt.get_cmap('hsv')
        colors = [cmap(i) for i in np.linspace(0, 1, len(start_positions))]
        #colors = colors.reshape(-1, 1)

    # Create a plot
        #fig, ax = plt.subplots(figsize=(20, 10))

    # Scatter plot for starting positions
        # plt.scatter(np.array(start_positions)[:, 0], 
        #         np.array(start_positions)[:, 1], 
        #         c=colors, s=20, alpha=alpha)
    #make scatter point for each uav individually
        for i in range(number_of_uavs):
            plt.scatter(start_positions[i][0], start_positions[i][1], c=colors[i], s=20)


        #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
        

    # Scatter plot for goal positions
        plt.scatter(1.2 * np.array(goal_positions)[:, 0], 
                1.2 *  np.array(goal_positions)[:, 1], 
                c=colors, marker="s", s=200)

    # Set plot limits and labels
        plt.xlim(-1.2 * radius, 1.2 * radius)
        plt.ylim(-radius/2, radius/2)
        plt.xlabel('x', fontsize=24)
        plt.ylabel('y', fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        #plt.legend()

    # Draw circles to indicate boundaries
        plt.gca().add_patch(plt.Circle((0, 0), radius, fill=False))
        plt.gca().add_patch(plt.Circle((0, 0), 1.2 * radius, fill=False))
        #plt.pause(0.01)
        plt.savefig('plots18/{}.png'.format(r))"""
    print(completed_status)
    print(check)
    time_file.write(", ".join([str(x) for x in mission_completion])+'\n')
    time_file.close()

    results_df = pd.DataFrame({
        'UAV_ID': range(number_of_uavs),
        'Priority': priority,
        'Time_To_Goal': mission_completion
    })

    # Replace 0s with None for UAVs that did not reach their goal
    results_df['Time_To_Goal'].replace(0, None, inplace=True)

    # Save the DataFrame to a CSV file
    results_df.to_csv('20_uav_priority_times.csv', index=False)

    file.close()
    velocity_file.close()
    path_file.close()


    acc_file.close()

    for i in range(n):
        avg_dist[i] = np.mean(min_dist[i])
    
    avg_file = open('avg.csv','a')
    avg_file.write(",".join([str(x) for x in avg_dist])+"\n")
    # end = time.time()
    # excec_time = end-start
    # exec_file.write(f'{i},{excec_time}')
    print("n:",number_of_uavs, "priority: ",priority)
    avg_file.close()



"""
number_of_uavs: Number of UAVs needs to be in the multiple of with minimum of 4 UAVs
adjusted_max_velocity = vmax
"""




