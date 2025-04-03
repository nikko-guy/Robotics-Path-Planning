"""lab5 controller."""

from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633  # [m/s]
AXLE_LENGTH = 0.4044  # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75  # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# Create the Robot instance
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = (
    "head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
    "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", 
    "arm_6_joint", "arm_7_joint", "wheel_left_joint", "wheel_right_joint"
)

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, "inf", "inf")
robot_parts = []

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# Set up the sensors
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

# The display is used to display the map
display = robot.getDevice("display")

# Set up robot state variables
pose_x = 0
pose_y = 0
pose_theta = 0
vL = 0
vR = 0
furthest_point_so_far = 0
goal_reached = False
object_positions = [(-2.28, -9.85), (-6.96, -6.14)]
start_ws = (0, 0)
end_ws = [(-1.4, -9.77), (-6.92, -5.23)]
object_of_interest = 0
wait_timer = 0
backup_distance = 0
backup_phase = False

# Set up LIDAR
lidar_sensor_readings = []
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE / 2.0, +LIDAR_ANGLE_RANGE / 2.0, LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets) - 83]

# Set mode
mode = "picknplace"  # Options: "manual", "planner", "autonomous", "picknplace"

probability_step = 5e-3

# Initialize map
map = np.zeros(shape=[360, 360])
waypoints = []
convolved_map = None

class RobotController:
    @staticmethod
    def load_map():
        """Load a saved map from disk."""
        global map
        map = np.load("map.npy").astype(np.float32)
        map = np.transpose(map)
        
        for x in range(0, 360):
            for y in range(0, 360):
                if map[x, y] == 1:
                    display.setColor(0xFFFFFF)
                    display.drawPixel(x, y)
    
    @staticmethod
    def save_map():
        """Save the current map to disk."""
        global map
        filtered_map = map > 0.8
        np.save("map.npy", filtered_map)
        print("Map file saved")
    
    @staticmethod
    def world_to_map(point):
        """Convert world coordinates (meters) to map coordinates (pixels)."""
        x = 360 - abs(int(point[0] * 30))
        y = abs(int(point[1] * 30))
        return x, y
    
    @staticmethod
    def map_to_world(point):
        """Convert map coordinates (pixels) to world coordinates (meters)."""
        x = (point[0] / 30) - 12
        y = -(point[1] / 30)
        return x, y
    
    @staticmethod
    def waypoints_to_world(waypoints):
        waypoints_w = []
        for point in waypoints:
            world_x = (point[0] / 30) - 12  # x increases from left to right in both systems
            world_y = -(point[1] / 30)  # y increases downward in map but upward in world
            waypoints_w.append((world_x, world_y))
        return np.array(waypoints_w)
    
    @staticmethod
    def create_configuration_space():
        """Create the configuration space by dilating obstacles."""
        global convolved_map
        convolved_map = convolve2d(map, np.ones((19, 19)), mode="same", boundary="fill", fillvalue=0)
        convolved_map = convolved_map > 0.5
        convolved_map = np.transpose(convolved_map)
        return convolved_map
    
    @staticmethod
    def get_closest_valid_point(map, point):
        """Find the closest valid (non-obstacle) point to the given point on the map."""
        x, y = int(point[0]), int(point[1])
        
        # If the point is already valid, return it
        if 0 <= x < map.shape[1] and 0 <= y < map.shape[0] and map[y, x] == 0:
            return (x, y)
        
        # Search in expanding circles
        max_radius = 50
        for radius in range(1, max_radius):
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if abs(i) == radius or abs(j) == radius:
                        nx, ny = x + i, y + j
                        if (0 <= nx < map.shape[1] and 0 <= ny < map.shape[0] and 
                            map[ny, nx] == 0):
                            return (nx, ny)
        
        return None
    
    @staticmethod
    def path_planner(map, start, end):
        """Plan a path using A* algorithm."""
        # Check if start or end coordinates are out of bounds
        height, width = map.shape
        
        if not (0 <= start[0] < width and 0 <= start[1] < height):
            print(f"Start position {start} is out of bounds for map of size {width}x{height}")
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
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        
        # Define possible movements (8-connected grid)
        movements = [
            (0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1),  # 4-connected
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)  # diagonals
        ]
        
        open_set = {start}
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float("inf")))
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            open_set.remove(current)
            closed_set.add(current)
            
            for dx, dy, cost in movements:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip if out of bounds
                if (neighbor[0] < 0 or neighbor[0] >= width or 
                    neighbor[1] < 0 or neighbor[1] >= height):
                    continue
                
                # Skip if in closed set
                if neighbor in closed_set:
                    continue
                
                # Skip if it's an obstacle
                if map[neighbor[1], neighbor[0]] > 0:
                    continue
                
                tentative_g_score = g_score[current] + cost
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float("inf")):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
        
        print("No path found")
        return []
    
    @staticmethod
    def initialise_path(map, start_ws, end_ws):
        """Initialize a path from start to end in world coordinates."""
        start = RobotController.world_to_map(start_ws)
        end = RobotController.world_to_map(end_ws)
        
        end_point = RobotController.get_closest_valid_point(map, end)
        
        if end_point is None:
            print("No valid end point found")
            return [], []
        
        path = RobotController.path_planner(map, start, end_point)
        if len(path) == 0:
            print("No path found")
            return [], []
        
        waypoints = np.array(path)
        waypoints_w = RobotController.waypoints_to_world(waypoints)
        
        # Draw the path on the map
        initial_color = 0xA00000
        for i, point in enumerate(waypoints):
            display.setColor(initial_color + i * 2)
            display.drawPixel(int(point[0]), int(point[1]))
        
        return waypoints, waypoints_w
    
    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    @staticmethod
    def clip_angle(angle):
        """Clip angle to [0, 2π]"""
        if angle > 2 * math.pi:
            angle -= 2 * math.pi
        elif angle < 0:
            angle += 2 * math.pi
        return angle
    
    @staticmethod
    def find_closest_point_in_path(path, pose_x, pose_y):
        """Find the index of the closest waypoint."""
        min_distance = float("inf")
        index = 0
        
        for i, point in enumerate(path):
            distance = math.sqrt((point[0] - pose_x) ** 2 + (point[1] - pose_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                index = i
        
        return index
    
    @staticmethod
    def turn_to_direction(pose_theta, target_theta, speed):
        """Turn to face a specific direction."""
        # Calculate angle difference
        angle_diff = target_theta - pose_theta
        
        # Normalize the angle difference
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Determine turn direction
        if angle_diff > 0:  # Need to turn left
            vL = -speed
            vR = speed
        else:  # Need to turn right
            vL = speed
            vR = -speed
        
        # Scale speed based on how close to target angle
        scale_factor = min(1.0, abs(angle_diff) / 0.1)
        vL *= scale_factor
        vR *= scale_factor
        
        return vL, vR
    
    @staticmethod
    def follow_path_controller(pose_x, pose_y, pose_theta, waypoints_w, furthest_point_so_far):
        """Path following controller."""
        vL, vR = 0, 0
        max_turn_speed = MAX_SPEED / 4
        max_speed = MAX_SPEED
        
        # Look ahead for smoother trajectory
        lookahead = 4
        
        # Find next waypoint
        index = np.clip(
            RobotController.find_closest_point_in_path(waypoints_w, pose_x, pose_y) + lookahead,
            0,
            len(waypoints_w) - 1
        )
        
        if index > furthest_point_so_far:
            furthest_point_so_far = index
            
        closest_point = waypoints_w[furthest_point_so_far]
        
        # Calculate error
        rho = np.linalg.norm(np.array(closest_point) - np.array([pose_x, pose_y]))
        
        dx = closest_point[0] - pose_x
        dy = closest_point[1] - pose_y
        desired_theta = RobotController.clip_angle(math.atan2(dy, dx) - np.pi / 2)
        
        # Calculate angle difference
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
        
        # Controller logic
        if alpha > 0.5:  # Large error - pure rotation
            vL, vR = RobotController.turn_to_direction(pose_theta, desired_theta, max_turn_speed)
            
            # Add correction for caster wheel drift
            correction = 0.2 * np.sign(angle_diff)
            vL += correction
            vR -= correction
        elif alpha > 0.1:  # Moderate error - blend turning and forward motion
            blend_factor = 1.0 - (alpha - 0.1) / 0.4
            
            turn_vL, turn_vR = RobotController.turn_to_direction(pose_theta, desired_theta, max_turn_speed)
            
            forward_speed = max_speed * 0.5 * blend_factor
            
            vL = turn_vL * (1 - blend_factor) + (forward_speed - 0.3 * angle_diff) * blend_factor
            vR = turn_vR * (1 - blend_factor) + (forward_speed + 0.3 * angle_diff) * blend_factor
        else:  # Well-aligned - forward motion with steering
            if rho > 0.05:
                base_speed = min(max_speed, max_speed * (rho / 1.0) * 2)
                steering = 0.3 * angle_diff
                
                vL = base_speed - steering
                vR = base_speed + steering
        
        return vL, vR, furthest_point_so_far
    
    @staticmethod
    def picknplace_sequence(waypoints_w, object_pos, pose_theta):
        """Pick and place sequence controller."""
        global wait_timer, backup_distance, backup_phase
        
        dx = object_pos[0] - waypoints_w[-1][0]
        dy = object_pos[1] - waypoints_w[-1][1]
        angle_to_goal = math.atan2(dy, dx) - np.pi / 2
        angle_to_goal = RobotController.clip_angle(angle_to_goal)
        
        alpha = abs(angle_to_goal - pose_theta)
        
        # If we're in backup phase
        if backup_phase:
            backup_speed = -MAX_SPEED / 2
            
            # Increment backup distance
            backup_distance += abs(backup_speed) / MAX_SPEED * MAX_SPEED_MS * timestep / 1000.0
            
            # Check if we've backed up enough
            if backup_distance >= 0.5:
                backup_phase = False
                backup_distance = 0
                return 0, 0, True  # Sequence complete
            
            return backup_speed, backup_speed, False
        
        # Normal sequence (aligning and grasping)
        if alpha < 0.1:
            print("Facing object")
            wait_timer += 1
            if wait_timer > 200:
                wait_timer = 0
                backup_phase = True
                return -MAX_SPEED / 2, -MAX_SPEED / 2, False
        
        # Turn to face object
        new_vL, new_vR = RobotController.turn_to_direction(pose_theta, angle_to_goal, MAX_SPEED / 4)
        
        return new_vL, new_vR, False


# Initialize for chosen mode
if mode in ["autonomous", "picknplace"]:
    RobotController.load_map()
    
    # Wait for valid GPS data
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    while np.isnan(pose_x) or np.isnan(pose_y):
        print("Waiting for GPS data...")
        robot.step(timestep)
        pose_x = gps.getValues()[0]
        pose_y = gps.getValues()[1]
    
    # Create configuration space
    convolved_map = RobotController.create_configuration_space()
    
    if mode == "picknplace":
        start_ws = (pose_x, pose_y)
        end_point_ws = end_ws[object_of_interest]
        waypoints, waypoints_w = RobotController.initialise_path(convolved_map, start_ws, end_point_ws)
    elif mode == "autonomous":
        # Find a random valid goal
        while True:
            end = (np.random.randint(0, 360), np.random.randint(0, 360))
            if convolved_map[end[1], end[0]] == 0:
                break
        
        end_w = RobotController.map_to_world(end)
        start_w = (pose_x, pose_y)
        waypoints, waypoints_w = RobotController.initialise_path(convolved_map, start_w, end_w)
elif mode == "planner":
    RobotController.load_map()
    
    # Wait for valid GPS data
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    while np.isnan(pose_x) or np.isnan(pose_y):
        print("Waiting for GPS data...")
        robot.step(timestep)
        pose_x = gps.getValues()[0]
        pose_y = gps.getValues()[1]
    
    # Create configuration space
    convolved_map = RobotController.create_configuration_space()
    
    # Set start position
    start_w = (pose_x, pose_y)
    start = RobotController.world_to_map(start_w)
    
    # Randomly sample a valid end point
    while True:
        end = (np.random.randint(0, 360), np.random.randint(0, 360))
        if convolved_map[end[1], end[0]] == 0:
            break
    
    # Plan path
    display.setColor(0xFF0000)
    display.drawPixel(start[0], start[1])
    display.drawPixel(end[0], end[1])
    
    path = RobotController.path_planner(convolved_map, start, end)
    waypoints = np.array(path)
    np.save("path.npy", waypoints)
    
    # Display path
    for point in waypoints:
        display.setColor(0x00A000)
        display.drawPixel(int(point[0]), int(point[1]))
    
    # Run simulation
    while robot.step(timestep) != -1:
        display.setColor(0x00FF00)
        display.drawPixel(int(end[0]), int(end[1]))

# Main control loop
while robot.step(timestep) != -1 and mode != "planner":
    # Update robot pose
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    
    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2])) - 1.5708)
    pose_theta = rad
    
    # Update map with LIDAR readings
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings) - 83]
    
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]
        
        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue
        
        # Convert to robot-centric coordinates
        rx = math.cos(alpha) * rho
        ry = -math.sin(alpha) * rho
        
        # Convert to world coordinates
        t = pose_theta + np.pi / 2.0
        wx = math.cos(t) * rx - math.sin(t) * ry + pose_x
        wy = math.sin(t) * rx + math.cos(t) * ry + pose_y
        
        # Handle boundary conditions
        if wx >= 12:
            wx = 11.999
        if wy >= 12:
            wy = 11.999
        
        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Update map
            pixel = (
                max(0, min(359, 360 - abs(int(wx * 30)))),
                max(0, min(359, abs(int(wy * 30))))
            )
            
            pixel_value = map[pixel[0], pixel[1]]
            if pixel_value < 1:
                pixel_value += probability_step
            pixel_value = min(1, pixel_value)
            map[pixel[0], pixel[1]] = pixel_value
            
            # Calculate color value properly
            color = int((pixel_value * 256**2 + pixel_value * 256 + pixel_value) * 255)
            display.setColor(color)
            display.drawPixel(pixel[0], pixel[1])
    
    # Draw robot position on map
    display.setColor(0xFF0000)
    display.drawPixel(360 - abs(int(pose_x * 30)), abs(int(pose_y * 30)))
    
    # Controller logic based on mode
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
            RobotController.save_map()
        elif key == ord("L"):
            RobotController.load_map()
            print("Map loaded")
        else:  # slow down
            vL *= 0.75
            vR *= 0.75
    else:
        # For autonomous and picknplace modes
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
            goal_reached = True
            
            new_vL, new_vR, sequence_finished = RobotController.picknplace_sequence(
                waypoints_w, object_positions[object_of_interest], pose_theta
            )
            
            if sequence_finished:
                object_of_interest += 1
                if object_of_interest >= len(object_positions):
                    print("Reached all objects")
                    vL = vR = 0
                    break
                
                goal_reached = False
                furthest_point_so_far = 0
                
                # Clear display and reload map
                display.setColor(0x000000)
                for i in range(360):
                    for j in range(360):
                        display.drawPixel(i, j)
                
                RobotController.load_map()
                
                # Plan new path to next object
                waypoints, waypoints_w = RobotController.initialise_path(
                    convolved_map,
                    (pose_x, pose_y),
                    end_ws[object_of_interest]
                )
            
            vL, vR = new_vL, new_vR
        else:
            vL, vR, furthest_point_so_far = RobotController.follow_path_controller(
                pose_x, pose_y, pose_theta, waypoints_w, furthest_point_so_far
            )
    
    # Apply velocity limits
    vL = np.clip(vL, -MAX_SPEED, MAX_SPEED)
    vR = np.clip(vR, -MAX_SPEED, MAX_SPEED)
    
    # Odometry update (even though we use GPS, this is for future use)
    pose_x += (vL + vR) / 2 / MAX_SPEED * MAX_SPEED_MS * timestep / 1000.0 * math.cos(pose_theta)
    pose_y -= (vL + vR) / 2 / MAX_SPEED * MAX_SPEED_MS * timestep / 1000.0 * math.sin(pose_theta)
    pose_theta += (vR - vL) / AXLE_LENGTH / MAX_SPEED * MAX_SPEED_MS * timestep / 1000.0
    
    # Normalize pose_theta
    pose_theta = RobotController.normalize_angle(pose_theta)
    
    # Send commands to motors
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)

# Keep controller running to avoid Webots bug on Windows
while robot.step(timestep) != -1:
    pass