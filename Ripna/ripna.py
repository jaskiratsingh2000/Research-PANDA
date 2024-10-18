import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import e  # If 'e' is used in the code
from hyperparameters import (
    spread, noise, radius, delt, pert, b, sensor_range, n, maxv, attractive_gain, 
    repulsive_gain, collision_distance, clipping_power, seed, priority_type, epsilon, k, min_radius
)
import dubins


# Global data array initialization
global_data_array = np.zeros((7, 13))

# Set the random seed for reproducibility
np.random.seed(seed)
print("Random seed set to:", seed)

def clear_previous_output_files():

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

    # Initialize arrays for velocity, position, and other attributes
    velocity_u = np.zeros((number_of_uavs, 2)) # if n=8 then 8 rows and 2 columns (x,y)
    velocity_v = np.ones((number_of_uavs, 2))
    start_positions, goal_positions = initialize_uav_positions_and_goals(number_of_uavs, radius)  # Initialize positions
    acceleration = np.zeros((number_of_uavs, 2))
    completed_status = np.zeros(number_of_uavs)
    clipping_status = np.zeros(number_of_uavs)
    adjusted_max_velocity = maxv * np.ones(number_of_uavs) # adjusted_max_velocity = vmax
    alpha = np.ones(number_of_uavs)*0.3

    # Initialize velocity_v with non-zero values
    for i in range(number_of_uavs):
        # Example vector pointing from start to goal, normalized and scaled by maxv
        direction = np.array(goal_positions[i]) - np.array(start_positions[i])
        if np.linalg.norm(direction) > 0:  # Avoid division by zero
            normalized_direction = direction / np.linalg.norm(direction)
            velocity_v[i] = normalized_direction * maxv  # Scale by maxv or any other desired value


    return velocity_u, velocity_v, start_positions, goal_positions, acceleration, completed_status, clipping_status, adjusted_max_velocity, alpha


# Function to check if a UAV has reached its goal
def reached_goal(uav_index, start_positions, goal_positions, radius):

    start_positions = np.array(start_positions[uav_index])
    goal_positions = np.array(goal_positions[uav_index])

    global mission_completion

    if np.linalg.norm(start_positions - goal_positions) <= 5:
        if mission_completion[uav_index] == 0:  # UAV has not yet reached its goal
            # Record the time taken for the UAV to reach its goal
            mission_completion[uav_index] = time.time() - uav_start_times[uav_index]
        return True
    else:
        return False



def calculate_time_to_go(uav_i, uav_j, start_positions, velocity_v):

    relative_velocity = velocity_v[uav_i] - velocity_v[uav_j]
    time_to_go = -np.dot(np.array(start_positions[uav_i]) - np.array(start_positions[uav_j]), relative_velocity) / (np.dot(relative_velocity, relative_velocity) + 0.000001)

    return time_to_go


def calculate_zero_effort_miss(uav_i, uav_j, start_positions, velocity_v):

    time_to_go = calculate_time_to_go(uav_i, uav_j, start_positions, velocity_v)
    zem_squared = np.dot(start_positions[uav_i] + velocity_v[uav_i] * time_to_go - start_positions[uav_j] - velocity_v[uav_j] * time_to_go,
                                    start_positions[uav_i] + velocity_v[uav_i] * time_to_go - start_positions[uav_j] - velocity_v[uav_j] * time_to_go)
    return np.sqrt(zem_squared)


def collision(uav_i, uav_j, start_positions, velocity_v, collision_distance, sensor_range): 

    time_to_go = calculate_time_to_go(uav_i, uav_j, start_positions, velocity_v)

    if time_to_go < 0:
        return False
    else:

        zem_squared = calculate_zero_effort_miss(uav_i, uav_j, start_positions, velocity_v)

        actual_distance_btw_2_uavs = np.linalg.norm(np.array(start_positions[uav_i]) - np.array(start_positions[uav_j]))

        if np.sqrt(zem_squared) < collision_distance and actual_distance_btw_2_uavs < sensor_range:
            return True
        else:
            return False


def calculate_turn_radius(ZEM, Rmin, lambda_param, Rdes):

    R = Rmin * np.exp(lambda_param * ZEM / Rdes)
    return R


def rotate_vector_clockwise_90(vector):

    x, y = vector[0], vector[1]
    return np.array([-y, x])  # 90 degrees clockwise rotation


def calculate_dubins_path(start, goal, turning_radius):

    # Dubins path configuration
    dubins_path = dubins.shortest_path(start, goal, turning_radius)
    # Sample the path at intervals of 'delt'
    configurations, _ = dubins_path.sample_many(maxv*delt)
    
    return configurations

def radians_to_degrees(radians):
    return radians * 180 / np.pi

if __name__ == "__main__":

    
    if 'number_of_uavs' not in globals():
        number_of_uavs = 20  # Default number of UAVs
    # if 'radius' not in globals():
    #     radius = 50  # Default radius
    # if 'delt' not in globals():
    #     delt = 0.01  # Default time delta
    # if 'maxv' not in globals():
    #     maxv = 1.0  # Default maximum velocity

    min_dist = np.inf*np.ones((number_of_uavs, number_of_uavs))
    avg_dist = np.zeros(number_of_uavs)

    mission_completion = [0]*number_of_uavs

    # Initialize UAV positions and goals
    start_positions, goal_positions = initialize_uav_positions_and_goals(number_of_uavs, radius)

    # Plot initial positions and goals (optional)
    plot_initial_positions_and_goals(start_positions, goal_positions, number_of_uavs, radius)

    velocity_u, velocity_v, start_positions, goal_positions, acceleration, completed_status, clipping_status, adjusted_max_velocity, alpha = initialize_uav_properties(number_of_uavs, radius)
     # Initialize start times for each UAV at the beginning of the simulation
    uav_start_times = {uav_id: time.time() for uav_id in range(number_of_uavs)}
    path_indices = [0] * number_of_uavs
    #print("initial velocity_v: ", velocity_v)

    # File initialization for output
    file = open('out_ripna.csv', 'w')
    velocity_file = open('vel_ripna.csv', 'w')
    time_file = open('time_ripna.csv', 'a')
    acc_file = open('acc_ripna.csv', 'w')
    path_file = open('path_lengths.csv', 'w')

    # Write headers for the files
    #header = ','.join([f"{i}x,{i}y" for i in range(number_of_uavs)])
    header = ','.join(["{}x,{}y".format(i, i) for i in range(number_of_uavs)])
    file.write(header + "\n")
    velocity_file.write(header + "\n")
    acc_file.write(header + "\n")
    path_file.write("UAV ID,Path Length\n")

    # Initialize position history storage
    position_history = [[] for _ in range(number_of_uavs)]

    # Initialize an array to store the path length for each UAV
    path_lengths = np.zeros(number_of_uavs)

    # Initialize an array to store the previous positions of UAVs for distance calculation
    previous_positions = np.copy(start_positions)


    r = 0
    #Main Simulation loop
    while not np.array_equal(completed_status, np.ones(number_of_uavs)):

        r += 1
        clipping_status = np.zeros(number_of_uavs)
        acceleration = np.zeros((number_of_uavs, 2))

        for i in range(number_of_uavs):
            #print(i, "distance to goal: ", np.linalg.norm(np.array(start_positions[i]) - np.array(goal_positions[i])))
            if reached_goal(i, start_positions, goal_positions, radius):
                print(i, "reached goal")
                if completed_status[i] != 1:
                    completed_status[i] = 1
                    acceleration[i] = 0
                    velocity_v[i] = 0
                    mission_completion[i] = r
                    path_file.write("{},{}\n".format(i, path_lengths[i]))
            else:
                #print(i, "not reached goal | distance to goal: ", np.linalg.norm(np.array(start_positions[i]) - np.array(goal_positions[i])))
                distance_traveled = np.linalg.norm(start_positions[i] - previous_positions[i])
                path_lengths[i] += distance_traveled
                # Identify the UAV with the smallest TGO
                smallest_tgo = float('inf')  # Initialize with a large number
                closest_uav = None  # Initialize with None

                colliding=False
                #print(i, colliding)

                for j in range(number_of_uavs):
                    if j != i:  # Ensure we're not calculating TGO with itself
                        tgo = calculate_time_to_go(i, j, start_positions, velocity_v)
                        if 0 < tgo < smallest_tgo:  # Check if this TGO is the smallest and positive
                            #print(i, "tgo", tgo, "smallest_tgo", smallest_tgo)
                            smallest_tgo = tgo
                            closest_uav = j
                #print(i,"closest uav: ", closest_uav)
                #print(start_positions[i], start_positions[closest_uav], start_positions[j])

                # If a closest UAV is found, apply lateral acceleration to avoid it
                if closest_uav is not None:
                    colliding = collision(i, closest_uav, start_positions, velocity_v, collision_distance, sensor_range)
                    print(i, colliding, "collision of ",i,closest_uav)
                    if colliding:
                        #print(i, "collision avoidance mode")
                        alpha[i] = 1
                        #colliding=True
                        # Calculate lateral acceleration direction by rotating velocity vector by 90 degrees clockwise
                        direction_for_acceleration = rotate_vector_clockwise_90(velocity_v[i])
                        # Calculate the magnitude of acceleration based on velocity magnitude and turn radius
                        V_magnitude = np.linalg.norm(velocity_v[i])
                        ZEM = calculate_zero_effort_miss(i, closest_uav, start_positions, velocity_v)
                        R = calculate_turn_radius(ZEM, Rmin=30, lambda_param=0.1, Rdes=sensor_range)  # Use sensor_range for Rdes
                        acceleration_magnitude = V_magnitude ** 2 / R
                        # Apply acceleration in the direction obtained from the rotated velocity vector
                        norm = np.linalg.norm(direction_for_acceleration)
                        if norm > 0:
                            acceleration[i] = direction_for_acceleration / norm * acceleration_magnitude
                        else:
                            acceleration[i] = np.zeros_like(direction_for_acceleration)

                        clipping_status[i] = 1  # Mark UAV for velocity clipping if needed

                    # Update velocity and position based on acceleration
                    #velocity_v[i] += acceleration[i] * delt
                    #start_positions[i] += velocity_v[i] * delt
                    print(i, colliding)
                    if colliding: #The clip array is used to keep track of which UAVs need to adjust their movement to avoid collisions. By setting clip[i] = 1, the code is marking UAV i for such an adjustment.
                        clipping_status[i]=1
                    current_heading = np.arctan2(velocity_v[i][1], velocity_v[i][0])
                    #print(i,"current heading:",radians_to_degrees(current_heading))
                    if not colliding:
                        alpha[i]=0.5
                        #print(i, "going towards goal")
                        # Calculate current and desired headings
                        
                        goal_heading = np.arctan2(goal_positions[i][1] - start_positions[i][1], goal_positions[i][0] - start_positions[i][0])

                        # Check if within the threshold for proportional control
                        if abs(goal_heading - current_heading) < epsilon:
                            # Adjust heading using proportional control
                            heading_error = current_heading - goal_heading
                            #print("heading error: ", heading_error)
                            
                            #calculate omega = heading error/delt
                            omega = heading_error/delt
                            # new theta = current_heading + omega*delt
                            current_heading = current_heading + omega*delt
                            # new velocity = vmax*[cos(new theta), sin(new theta)]

                            velocity_v[i] = maxv * np.array([np.cos(current_heading), np.sin(current_heading)])
                        else:
                            # Follow Dubins path
                            start_config = (start_positions[i][0], start_positions[i][1], current_heading)
                            goal_config = (goal_positions[i][0], goal_positions[i][1], goal_heading)  # No specific arrival angle
                            dubins_path = calculate_dubins_path(start_config, goal_config, min_radius)
                            if dubins_path:
                                next_config = dubins_path[1]  # Take the next configuration from Dubins path
                                direction = np.array([next_config[0] - start_positions[i][0], next_config[1] - start_positions[i][1]])
                                velocity_v[i] = direction / np.linalg.norm(direction) * maxv

                previous_positions = np.copy(start_positions)  # Update for the next iteration


        #if pert:
            #perturb_velocity(velocity_v)
                        
        velocity_v = velocity_v + acceleration*delt
        print("acceleration",[np.linalg.norm(acceleration[uav]) for uav in range(number_of_uavs)])
        print("velocity",[np.linalg.norm(velocity_v[uav]) for uav in range(number_of_uavs)])
        #print(velocity_v)
        #clip_velocity(number_of_uavs)
        start_positions = start_positions + velocity_v*delt

        file.write(",".join([str(x) for x in start_positions.flatten()])+"\n")
        velocity_file.write(", ".join([str(x) for x in velocity_v.flatten()])+"\n")
        acc_file.write(", ".join([str(x) for x in acceleration.flatten()])+"\n")

        #plt.clf()
        #cmap = plt.get_cmap('hsv')
        #colors = [cmap(i) for i in np.linspace(0, 1, len(start_positions))]

    # Create a plot
        #fig, ax = plt.subplots(figsize=(20, 10))

    # Scatter plot for starting positions
        # plt.scatter(np.array(start_positions)[:, 0], 
        #         np.array(start_positions)[:, 1], 
        #         c=colors, s=20, alpha=alpha)
    #make scatter point for each uav individually
        #for i in range(number_of_uavs):
            #plt.scatter(start_positions[i][0], start_positions[i][1], c=colors[i], s=20)


        #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
        

    # Scatter plot for goal positions
        #plt.scatter(1.2 * np.array(goal_positions)[:, 0], 
                #1.2 *  np.array(goal_positions)[:, 1], 
                #c=colors, marker="s", s=200, alpha=0.5)

    # Set plot limits and labels
        #plt.xlim(-1.2 * radius, 1.2 * radius)
        #plt.ylim(-radius/2, radius/2)
        #plt.xlabel('x', fontsize=24)
        #plt.ylabel('y', fontsize=24)
        #plt.xticks(fontsize=20)
        #plt.yticks(fontsize=20)
        #plt.legend()

    # Draw circles to indicate boundaries
        #plt.gca().add_patch(plt.Circle((0, 0), radius, fill=False))
        #plt.gca().add_patch(plt.Circle((0, 0), 1.2 * radius, fill=False))
        #plt.pause(0.01)
        #plt.savefig('plots17/{}.png'.format(r))
    print(completed_status)
    #print(check)
    time_file.write(", ".join([str(x) for x in mission_completion])+'\n')
    time_file.close()

    results_df = pd.DataFrame({
        'UAV_ID': range(number_of_uavs),
        'Time_To_Goal': mission_completion
    })

    # Replace 0s with None for UAVs that did not reach their goal
    results_df['Time_To_Goal'].replace(0, None, inplace=True)

    # Save the DataFrame to a CSV file
    results_df.to_csv('20_uav_ripna_times.csv', index=False)

    file.close()
    velocity_file.close() 


    acc_file.close()

    for i in range(number_of_uavs):
        avg_dist[i] = np.mean(min_dist[i])
    
    avg_file = open('avg.csv','a')
    avg_file.write(",".join([str(x) for x in avg_dist])+"\n")
    # end = time.time()
    # excec_time = end-start
    # exec_file.write(f'{i},{excec_time}')
    #print("n:",number_of_uavs, "priority: ",priority)
    avg_file.close()



"""
number_of_uavs: Number of UAVs needs to be in the multiple of with minimum of 4 UAVs
adjusted_max_velocity = vmax
"""




