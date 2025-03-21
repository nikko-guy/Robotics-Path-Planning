"""lab5 controller."""

from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import (
    convolve2d,
)  # Uncomment if you want to use something else for finding the configuration space

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633  # [m/s]
AXLE_LENGTH = 0.4044  # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75  # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


##### vvv [Begin] Do Not Modify vvv #####

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = (
    "head_2_joint",
    "head_1_joint",
    "torso_lift_joint",
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
    "arm_7_joint",
    "wheel_left_joint",
    "wheel_right_joint",
)

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, "inf", "inf")
robot_parts = []

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

range_finder = robot.getDevice("range-finder")
range_finder.enable(timestep)
camera = robot.getDevice("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)
lidar = robot.getDevice("Hokuyo URG-04LX-UG01")
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 360x360 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")

# Odometry
pose_x = 0
pose_y = 0
pose_theta = 0

furthest_point_so_far = 0

goal_reached = False

object_positions = [(-2.28, -9.85), (-6.96, -6.14)]
start_ws = (0, 0)
end_ws = [(-1.4, -9.77), (-6.92, -5.23)]

object_of_interest = 0

wait_timer = 0

vL = 0
vR = 0

lidar_sensor_readings = []  # List to hold sensor readings
lidar_offsets = np.linspace(
    -LIDAR_ANGLE_RANGE / 2.0, +LIDAR_ANGLE_RANGE / 2.0, LIDAR_ANGLE_BINS
)
lidar_offsets = lidar_offsets[
    83 : len(lidar_offsets) - 83
]  # Only keep lidar readings not blocked by robot chassis

# map = None
##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
# mode = "manual"  # Part 1.1: manual mode
# mode = "planner"
# mode = "autonomous"
mode = "picknplace"

probability_step = 5e-3

######################
#
# Map Initialization
#
######################


# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
map = np.zeros(shape=[360, 360])
waypoints = []


def load_map():
    global map
    map = np.load("map.npy").astype(np.float32)
    map = np.transpose(map)

    for x in range(0, 360):
        for y in range(0, 360):
            if map[x, y] == 1:
                display.setColor(0xFFFFFF)
                display.drawPixel(x, y)


###################
#
# Planner
#
###################
def path_planner(map, start, end):
    """
    :param map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
    :param start: A tuple of indices representing the start cell in the map
    :param end: A tuple of indices representing the end cell in the map
    :return: A list of tuples as a path from the given start to the given end in the given maze
    """

    # Check if start or end coordinates are out of bounds
    height, width = map.shape

    if not (0 <= start[0] < width and 0 <= start[1] < height):
        print(
            f"Start position {start} is out of bounds for map of size {width}x{height}"
        )
        return []
    if not (0 <= end[0] < width and 0 <= end[1] < height):
        print(f"End position {end} is out of bounds for map of size {width}x{height}")
        return []

    # Check if start or end is in an obstacle
    if map[start[1], start[0]] > 0:
        print(f"Start position {start} is in an obstacle")
        return []
    if map[end[1], end[0]] > 0:
        print(f"End position {end} is in an obstacle")
        return []

    # A* algorithm implementation
    # Based on pseudocode from: https://en.wikipedia.org/wiki/A*_search_algorithm

    # Define heuristic function (Euclidean distance)
    def heuristic(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # Define possible movements (8-connected grid)
    # (x, y, cost) - diagonal moves cost more
    movements = [
        (0, 1, 1),
        (1, 0, 1),
        (0, -1, 1),
        (-1, 0, 1),  # 4-connected
        (1, 1, 1.414),
        (1, -1, 1.414),
        (-1, 1, 1.414),
        (-1, -1, 1.414),  # diagonals
    ]

    # Initialize open and closed sets
    open_set = {start}  # set of nodes to be evaluated
    closed_set = set()  # set of nodes already evaluated

    # Dictionary to store the most efficient previous step
    came_from = {}

    # Dictionary with current best cost from start to each node
    g_score = {start: 0}

    # Dictionary with current best estimated total cost from start to goal through node
    f_score = {start: heuristic(start, end)}

    while open_set:
        # Find node in open_set with lowest f_score
        current = min(open_set, key=lambda x: f_score.get(x, float("inf")))

        # If we reached the end, reconstruct and return the path
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        # Move current from open_set to closed_set
        open_set.remove(current)
        closed_set.add(current)

        # Check all neighbors
        for dx, dy, cost in movements:
            neighbor = (current[0] + dx, current[1] + dy)

            # Skip if out of bounds
            if (
                neighbor[0] < 0
                or neighbor[0] >= map.shape[1]
                or neighbor[1] < 0
                or neighbor[1] >= map.shape[0]
            ):
                continue

            # Skip if in closed set
            if neighbor in closed_set:
                continue

            # Skip if it's an obstacle
            if map[neighbor[1], neighbor[0]] > 0:
                continue

            # Calculate tentative g_score
            tentative_g_score = g_score[current] + cost

            # If neighbor not in open_set, add it
            if neighbor not in open_set:
                open_set.add(neighbor)
            # If this path to neighbor is worse than previous one, skip
            elif tentative_g_score >= g_score.get(neighbor, float("inf")):
                continue

            # This path is the best so far, record it
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)

    # If we get here, no path was found
    print("No path found from start to end")
    return []


def waypoints_to_world(waypoints):
    waypoints_w = []
    for point in waypoints:
        world_x = (point[0] / 30) - 12  # x increases from left to right in both systems
        world_y = -(point[1] / 30)  # y increases downward in map but upward in world
        waypoints_w.append((world_x, world_y))
    waypoints_w = np.array(waypoints_w)
    return waypoints_w


if mode == "planner":
    load_map()
    pose_x = np.float64(gps.getValues()[0])
    pose_y = np.float64(gps.getValues()[1])
    # print(type(pose_x), pose_x)
    # print(np.isnan(pose_x))
    while np.isnan(pose_x) or np.isnan(pose_y):
        print("Waiting for GPS data...")
        robot.step(timestep)
        pose_x = gps.getValues()[0]
        pose_y = gps.getValues()[1]

    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    start_w = pose_x, pose_y
    end_w = None  # (Pose_X, Pose_Y) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    start_pixel_x = 360 - abs(int(pose_x * 30))
    start_pixel_y = abs(int(pose_y * 30))
    start = (start_pixel_x, start_pixel_y)

    # Part 2.2: Compute an approximation of the "configuration space"
    convolved_map = convolve2d(
        map, np.ones((18, 18)), mode="same", boundary="fill", fillvalue=0
    )
    convolved_map = convolved_map > 0.5
    convolved_map = np.transpose(convolved_map)
    # display the convolved map
    plt.figure(figsize=(8, 8))
    plt.imshow(convolved_map, origin="upper")
    # Mark the start position with a red dot
    plt.plot(
        start[0], start[1], "ro", markersize=10
    )  # Use same coordinates as in Webots
    # plt.show()

    # randomly sample a point in the map that is not an obstacle
    while True:
        end = (
            np.random.randint(0, 360),
            np.random.randint(0, 360),
        )
        if convolved_map[end[1], end[0]] == 0:  # Note: array is indexed as [y, x]
            break

    end_w = (end[0] / 30, end[1] / 30)

    print(f"Start pixel: {start}, End pixel: {end}")
    print(f"End world: {end_w}")

    # draw the start and end on the map
    display.setColor(0xFF0000)
    display.drawPixel(start[0], start[1])
    display.drawPixel(end[0], end[1])

    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path

    # Part 2.1: Load map (map.npy) from disk and visualize it
    # map = np.load("map.npy")
    # # switch x and y axes
    # map = np.transpose(map)
    # # for x in range(0,360):
    # #     for y in range(0,360):
    # #         if map[x, y] == 1:
    # #             display.setColor(0x0000FF)
    # #             display.drawPixel(x, y)

    # # Part 2.2: Compute an approximation of the "configuration space"

    # # convolve with a 5x5 kernel to smooth out the map
    # kernel = np.ones((15, 15))
    # map = convolve2d(map, kernel, mode="full", boundary="fill", fillvalue=0)
    # map = map > 0.5
    # plt.imshow(map)
    # plt.show()

    # Part 2.3 continuation: Call path_planner
    path = path_planner(convolved_map, start, end)
    print("Path found: ", path)
    # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
    waypoints = np.array(path)
    np.save("path.npy", waypoints)
    # plot the path
    plt.title("Path")
    plt.plot(waypoints[:, 0], waypoints[:, 1], "ro", markersize=1)
    plt.show()

    while robot.step(timestep) != -1:
        # display goal on map
        display.setColor(0x00FF00)
        display.drawPixel(int(end[0]), int(end[1]))
        # display waypoints on map
        for point in waypoints:
            display.setColor(0x00A000)
            # Convert world coordinates back to pixel coordinates
            map_x = int(point[0])
            map_y = int(point[1])
            display.drawPixel(map_x, map_y)


if mode == "autonomous":
    # Part 3.1: Load path from disk and visualize it
    waypoints = np.load("path.npy")
    # Properly convert map coordinates to world coordinates
    waypoints_w = waypoints_to_world(waypoints)

    # draw the path on the map
    initial_color = 0xA00000
    for i, point in enumerate(waypoints):
        display.setColor(initial_color + i * 2)
        display.drawPixel(int(point[0]), int(point[1]))

    # display.setColor(0x00FF00)
    # display.drawPixel(10, 10)


def get_closest_valid_point(map, point):
    """
    Find the closest valid (non-obstacle) point to the given point on the map.

    Args:
        map: A 2D numpy array where values > 0 represent obstacles
        point: A tuple (x, y) representing the target point

    Returns:
        A tuple (x, y) representing the closest valid point
    """
    x, y = int(point[0]), int(point[1])

    # If the point is already valid, return it
    if 0 <= x < map.shape[1] and 0 <= y < map.shape[0] and map[y, x] == 0:
        return (x, y)

    # Search in expanding circles around the point
    max_radius = 50  # Limit search radius to avoid excessive computation
    for radius in range(1, max_radius):
        # Check points in a square with sides of length 2*radius
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                # Only check points on the perimeter of the square
                if abs(i) == radius or abs(j) == radius:
                    new_x, new_y = x + i, y + j

                    # Check if the point is within map bounds and is not an obstacle
                    if (
                        0 <= new_x < map.shape[1]
                        and 0 <= new_y < map.shape[0]
                        and map[new_y, new_x] == 0
                    ):
                        return (new_x, new_y)

    # If no valid point found within the search radius, return None
    print("No valid point found within search radius")
    return None


def initialise_path(map, start_ws, end_ws):
    start = (360 - abs(int(start_ws[0] * 30)), abs(int(start_ws[1] * 30)))
    end = (360 - abs(int(end_ws[0] * 30)), abs(int(end_ws[1] * 30)))

    end_point = get_closest_valid_point(convolved_map, end)

    if end_point is None:
        print("No valid end point found")
        exit()

    path = path_planner(convolved_map, start, end_point)
    if len(path) == 0:
        print("No path found")
        exit()

    waypoints = np.array(path)
    waypoints_w = waypoints_to_world(waypoints)

    # draw the path on the map
    initial_color = 0xA00000
    for i, point in enumerate(waypoints):
        display.setColor(initial_color + i * 2)
        display.drawPixel(int(point[0]), int(point[1]))

    return waypoints, waypoints_w


if mode == "picknplace":
    # Part 4: Use the function calls from lab5_joints using the comments provided there
    ## use path_planning to generate paths
    ## do not change start_ws and end_ws below
    # load map
    load_map()

    convolved_map = convolve2d(
        map, np.ones((19, 19)), mode="same", boundary="fill", fillvalue=0
    )
    convolved_map = convolved_map > 0.5
    convolved_map = np.transpose(convolved_map)

    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]

    while np.isnan(pose_x) or np.isnan(pose_y):
        print("Waiting for GPS data...")
        robot.step(timestep)
        pose_x = gps.getValues()[0]
        pose_y = gps.getValues()[1]

    start_ws = (pose_x, pose_y)

    end_point_ws = end_ws[object_of_interest]

    # # generate paths
    waypoints, waypoints_w = initialise_path(convolved_map, start_ws, end_point_ws)


def turn_to_direction(pose_theta, target_theta, speed):
    """
    :param direction: A float representing the direction to turn to in radians
    """
    # Calculate the angle difference between current pose and target direction
    angle_diff = target_theta - pose_theta

    # Normalize the angle difference to be between -pi and pi
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi

    # Determine if we need to turn left or right
    if angle_diff > 0:  # Need to turn left
        vL = -speed
        vR = speed
    else:  # Need to turn right
        vL = speed
        vR = -speed

    # Scale speed smoothly based on how close we are to the target angle
    # The closer to the target angle, the slower the robot will turn
    scale_factor = min(1.0, abs(angle_diff) / 0.1)  # Linear scaling between 0 and 1
    vL *= scale_factor
    vR *= scale_factor

    return vL, vR


def speed_control(alpha, speed, distance, alpha_threshold=math.pi / 4):
    """
    Move to a target point with smooth speed control based on orientation error

    :param alpha: Angle difference between current orientation and desired orientation
    :param speed: Maximum speed
    :param distance: Distance to target point
    :param alpha_threshold: Maximum angle difference allowed before robot prioritizes turning over moving
    :return: Left and right wheel velocities
    """
    # Normalize the angle difference to be between 0 and pi
    alpha = abs(alpha)
    while alpha > math.pi:
        alpha -= 2 * math.pi
    alpha = abs(alpha)

    # If alpha is greater than threshold, prioritize turning by setting forward speed to 0
    if alpha > alpha_threshold:
        return 0, 0

    # Scale speed based on orientation error (alpha)
    # When alpha is close to 0, we want to move forward at full speed
    # When alpha is close to threshold, we should reduce speed
    orientation_factor = max(0, 1 - (alpha / alpha_threshold))

    # Also scale speed based on distance to target (slow down when close)
    distance_factor = min(1.0, distance / 0.5)  # Full speed when >= 0.5m away

    # Combine factors - we move slower when either misaligned or close to target
    speed_factor = orientation_factor * distance_factor

    # Set base velocities - move forward
    vL = speed * speed_factor
    vR = speed * speed_factor

    return vL, vR


def find_closest_point_in_path(path, pose_x, pose_y):
    min_distance = float("inf")
    index = 0

    for i, point in enumerate(path):
        distance = math.sqrt((point[0] - pose_x) ** 2 + (point[1] - pose_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            index = i
    return index


def clip_angle(angle):
    # clip between 0 and 2pi
    if angle > 2 * math.pi:
        angle -= 2 * math.pi
    elif angle < 0:
        angle += 2 * math.pi
    return angle


# I had to use claude to figure out how to deal with the caster wheels because my shit kept
# turning left and right due to the damn wheels.
def follow_path_controller(
    pose_x, pose_y, pose_theta, waypoints_w, furthest_point_so_far
):
    """
    Path following controller that uses a waypoint list to navigate

    Args:
        pose_x (float): Current x position of the robot
        pose_y (float): Current y position of the robot
        pose_theta (float): Current orientation of the robot
        waypoints_w (list): List of waypoints in world coordinates
        furthest_point_so_far (int): Index of furthest waypoint reached

    Returns:
        tuple: (vL, vR) wheel velocities, updated furthest_point_so_far
    """
    vL, vR = 0, 0  # Initialize velocities
    max_turn_speed = MAX_SPEED / 4
    max_speed = MAX_SPEED  # Reduced max speed for better control

    # Look ahead further on the path to smooth trajectory
    lookahead = 4  # Increased from 2 to 4

    # Find next waypoint to follow
    index = np.clip(
        find_closest_point_in_path(waypoints_w, pose_x, pose_y) + lookahead,
        0,
        len(waypoints_w) - 1,
    )
    if index > furthest_point_so_far:
        furthest_point_so_far = index
    closest_point = waypoints_w[furthest_point_so_far]
    # print(f"Closest point: {closest_point}, index: {furthest_point_so_far}")

    # Calculate the error
    rho = np.linalg.norm(np.array(closest_point) - np.array([pose_x, pose_y]))

    dx = closest_point[0] - pose_x
    dy = closest_point[1] - pose_y
    desired_theta = clip_angle(math.atan2(dy, dx) - np.pi / 2)

    # Calculate angle difference and normalize between -pi and pi
    angle_diff = desired_theta - pose_theta
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi

    alpha = abs(angle_diff)

    print(f"dx: {dx}, dy: {dy}")
    print(f"Desired theta: {np.degrees(desired_theta)} degrees")
    print(f"Current theta: {np.degrees(pose_theta)} degrees")
    print(f"Alpha: {np.degrees(alpha)} degrees")
    print(f"Distance to closest point: {rho}")

    # Controller logic with smooth blending between turning and forward motion
    if alpha > 0.5:  # Large error - pure rotation
        # Pure rotation to align
        vL, vR = turn_to_direction(pose_theta, desired_theta, max_turn_speed)

        # Add a small correction factor based on the sign of the angle difference
        # This helps compensate for caster wheel drift
        correction = 0.2 * np.sign(angle_diff)
        vL += correction
        vR -= correction
    elif alpha > 0.1:  # Moderate error - blend turning and forward motion
        # Calculate a blend factor (0 at alpha=0.5, 1 at alpha=0.1)
        blend_factor = 1.0 - (alpha - 0.1) / 0.4

        # Get turning velocities
        turn_vL, turn_vR = turn_to_direction(pose_theta, desired_theta, max_turn_speed)

        # Calculate forward motion component (reduced speed)
        forward_speed = max_speed * 0.5 * blend_factor

        # Blend turning and forward motion
        vL = (
            turn_vL * (1 - blend_factor)
            + (forward_speed - 0.3 * angle_diff) * blend_factor
        )
        vR = (
            turn_vR * (1 - blend_factor)
            + (forward_speed + 0.3 * angle_diff) * blend_factor
        )
    else:
        # Well-aligned - prioritize forward motion with minor steering corrections
        if rho > 0.05:
            # Proportional control for forward motion
            base_speed = min(
                max_speed, max_speed * (rho / 1.0) * 2
            )  # Scale with distance

            # Add slight turning component for continuous course correction
            # This compensates for the caster wheels' tendency to drift
            steering = 0.3 * angle_diff  # Proportional to angle error

            vL = base_speed - steering
            vR = base_speed + steering

    return vL, vR, furthest_point_so_far


def picknplace_sequence(waypoints_w, object_pos, pose_theta):
    """
    Pick and place sequence that uses a waypoint list to navigate

    Args:
        waypoints_w (list): List of waypoints in world coordinates
        object_pos (tuple): Position of the object in world coordinates
        pose_theta (float): Current orientation of the robot

    Returns:
        tuple: (vL, vR, sequence_finished) wheel velocities and sequence finished
    """
    global wait_timer, backup_distance, backup_phase

    # Initialize backup variables if they don't exist
    if "backup_distance" not in globals():
        global backup_distance
        backup_distance = 0

    if "backup_phase" not in globals():
        global backup_phase
        backup_phase = False

    dx = object_pos[0] - waypoints_w[-1][0]
    dy = object_pos[1] - waypoints_w[-1][1]
    angle_to_goal = math.atan2(dy, dx) - np.pi / 2
    angle_to_goal = clip_angle(angle_to_goal)

    alpha = abs(angle_to_goal - pose_theta)

    # If we're in the backup phase
    if backup_phase:
        # Back up at half speed
        backup_speed = -MAX_SPEED / 2

        # Increment backup distance (approximation based on timestep)
        backup_distance += (
            abs(backup_speed) / MAX_SPEED * MAX_SPEED_MS * timestep / 1000.0
        )

        # Check if we've backed up 0.5m
        if backup_distance >= 0.5:
            # Reset backup variables for next time
            backup_phase = False
            backup_distance = 0
            return 0, 0, True  # Sequence complete

        # Keep backing up
        return backup_speed, backup_speed, False

    # Normal sequence (aligning and grasping)
    if alpha < 0.1:
        print("Facing orange")
        # wait for 200 timesteps to simulate grasping orange
        wait_timer += 1
        if wait_timer > 200:
            wait_timer = 0
            # Start backing up
            backup_phase = True
            return -MAX_SPEED / 2, -MAX_SPEED / 2, False  # Start backing up

    new_vL, new_vR = turn_to_direction(pose_theta, angle_to_goal, MAX_SPEED / 4)

    return new_vL, new_vR, False


while robot.step(timestep) != -1 and mode != "planner":

    ###################
    #
    # Mapping
    #
    ###################

    ################ v [Begin] Do not modify v ##################
    # Ground truth pose
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]

    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2])) - 1.5708)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83 : len(lidar_sensor_readings) - 83]

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha) * rho
        ry = -math.sin(alpha) * rho

        t = pose_theta + np.pi / 2.0
        # Convert detection from robot coordinates into world coordinates
        wx = math.cos(t) * rx - math.sin(t) * ry + pose_x
        wy = math.sin(t) * rx + math.cos(t) * ry + pose_y

        ################ ^ [End] Do not modify ^ ##################

        # print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))
        if wx >= 12:
            wx = 11.999
        if wy >= 12:
            wy = 11.999
        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.

            # You will eventually REPLACE the following lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.
            pixel = (
                max(0, min(359, 360 - abs(int(wx * 30)))),
                max(0, min(359, abs(int(wy * 30)))),
            )
            pixel_value = map[pixel[0], pixel[1]]
            if pixel_value < 1:
                pixel_value += probability_step
            pixel_value = min(1, pixel_value)
            map[pixel[0], pixel[1]] = pixel_value

            color = int((pixel_value * 256**2 + pixel_value * 256 + pixel_value) * 255)

            display.setColor(color)
            display.drawPixel(pixel[0], pixel[1])

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    display.drawPixel(360 - abs(int(pose_x * 30)), abs(int(pose_y * 30)))

    ###################
    #
    # Controller
    #
    ###################
    if mode == "manual":
        key = keyboard.getKey()
        while keyboard.getKey() != -1:
            pass
        if key == keyboard.LEFT:
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(" "):
            vL = 0
            vR = 0
        elif key == ord("S"):
            # Part 1.4: Filter map and save to filesystem
            filtered_map = map > 0.8
            np.save("map.npy", filtered_map)
            print("Map file saved")
        elif key == ord("L"):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            load_map()
            print("Map loaded")
        else:  # slow down
            vL *= 0.75
            vR *= 0.75
    else:  # not manual mode

        distance_to_goal = np.linalg.norm(
            np.array(waypoints_w[-1]) - np.array([pose_x, pose_y])
        )

        if goal_reached:
            distance_to_goal = 0

        if distance_to_goal < 0.05 and mode != "picknplace":
            vL = 0
            vR = 0
            print("Reached goal")
            break
        elif distance_to_goal < 0.05 and mode == "picknplace":
            # print("Reached goal")
            goal_reached = True

            new_vL, new_vR, sequence_finished = picknplace_sequence(
                waypoints_w, object_positions[object_of_interest], pose_theta
            )

            if sequence_finished:
                if object_of_interest >= len(object_positions) - 1:
                    print("Reached all objects")
                    exit()
                goal_reached = False
                object_of_interest += 1
                furthest_point_so_far = 0

                # reload map
                display.setColor(int(0x000000))
                for i in range(360):
                    for j in range(360):
                        display.drawPixel(i, j)

                load_map()

                waypoints, waypoints_w = initialise_path(
                    convolved_map,
                    (pose_x, pose_y),
                    end_ws[object_of_interest],
                )

        else:
            new_vL, new_vR, furthest_point_so_far = follow_path_controller(
                pose_x, pose_y, pose_theta, waypoints_w, furthest_point_so_far
            )

        vL = np.clip(new_vL, -MAX_SPEED, MAX_SPEED)
        vR = np.clip(new_vR, -MAX_SPEED, MAX_SPEED)
    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    pose_x += (
        (vL + vR)
        / 2
        / MAX_SPEED
        * MAX_SPEED_MS
        * timestep
        / 1000.0
        * math.cos(pose_theta)
    )
    pose_y -= (
        (vL + vR)
        / 2
        / MAX_SPEED
        * MAX_SPEED_MS
        * timestep
        / 1000.0
        * math.sin(pose_theta)
    )
    pose_theta += (vR - vL) / AXLE_LENGTH / MAX_SPEED * MAX_SPEED_MS * timestep / 1000.0

    # Normalize pose_theta to [-π, π]
    while pose_theta > math.pi:
        pose_theta -= 2 * math.pi
    while pose_theta < -math.pi:
        pose_theta += 2 * math.pi

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)

while robot.step(timestep) != -1:
    # there is a bug where webots have to be restarted if the controller exits on Windows
    # this is to keep the controller running
    pass
