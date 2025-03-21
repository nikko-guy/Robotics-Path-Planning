import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Load the original map
original_map = np.load("./controllers/lab5_controller2/map.npy")
# original_map = np.transpose(original_map)

# Create a copy for editing
edited_map = original_map.copy()

# Create a figure and axis
fig, ax = plt.subplots()
im = ax.imshow(edited_map)  # Show the edited map directly instead of convolved map

# Track the current mode (add or remove obstacles)
current_mode = "add"  # Default mode is adding obstacles


# Function to handle rectangle selection for adding obstacles
def add_callback(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    # Ensure coordinates are within map bounds
    x1, x2 = max(0, min(x1, edited_map.shape[1] - 1)), max(
        0, min(x2, edited_map.shape[1] - 1)
    )
    y1, y2 = max(0, min(y1, edited_map.shape[0] - 1)), max(
        0, min(y2, edited_map.shape[0] - 1)
    )

    # Sort coordinates
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Add obstacles (set to 1)
    edited_map[y1 : y2 + 1, x1 : x2 + 1] = 1

    # Update the display
    im.set_data(edited_map)
    fig.canvas.draw_idle()


# Function to handle rectangle selection for removing obstacles
def remove_callback(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    # Ensure coordinates are within map bounds
    x1, x2 = max(0, min(x1, edited_map.shape[1] - 1)), max(
        0, min(x2, edited_map.shape[1] - 1)
    )
    y1, y2 = max(0, min(y1, edited_map.shape[0] - 1)), max(
        0, min(y2, edited_map.shape[0] - 1)
    )

    # Sort coordinates
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Remove obstacles (set to 0)
    edited_map[y1 : y2 + 1, x1 : x2 + 1] = 0

    # Update the display
    im.set_data(edited_map)
    fig.canvas.draw_idle()


# Instructions
ax.set_title(
    "Left-click to add obstacles, Right-click to remove\nPress 'a' to switch to add mode, 'r' to remove mode\nPress 's' to save, 'q' to quit"
)

# Set up the RectangleSelector for adding obstacles (left-click)
rs_add = RectangleSelector(
    ax,
    add_callback,
    useblit=True,
    button=[1],  # Left mouse button
    minspanx=5,
    minspany=5,
    spancoords="pixels",
    interactive=True,
)

# Set up the RectangleSelector for removing obstacles (right-click)
rs_remove = RectangleSelector(
    ax,
    remove_callback,
    useblit=True,
    button=[3],  # Right mouse button
    minspanx=5,
    minspany=5,
    spancoords="pixels",
    interactive=True,
)


# Function to handle keyboard events
def on_key_press(event):
    global current_mode, rs_add, rs_remove
    if event.key == "s":
        # Save only the edited map
        np.save("./controllers/lab5_controller2/map_edited.npy", edited_map)
        print("Map saved as map_edited.npy")
    elif event.key == "q":
        plt.close()
    elif event.key == "a":
        current_mode = "add"
        print("Mode: Adding obstacles")
        # Ensure only the add selector is active
        rs_add.set_active(True)
        rs_remove.set_active(False)
    elif event.key == "r":
        current_mode = "remove"
        print("Mode: Removing obstacles")
        # Ensure only the remove selector is active
        rs_add.set_active(False)
        rs_remove.set_active(True)


# Connect the key press event
fig.canvas.mpl_connect("key_press_event", on_key_press)


# Function to handle mouse button press to set the mode
def on_mouse_press(event):
    if event.inaxes != ax:  # Ignore clicks outside the axes
        return

    global current_mode, rs_add, rs_remove
    if event.button == 1:  # Left click
        current_mode = "add"
        # Ensure only the add selector is active
        rs_add.set_active(True)
        rs_remove.set_active(False)
    elif event.button == 3:  # Right click
        current_mode = "remove"
        # Ensure only the remove selector is active
        rs_add.set_active(False)
        rs_remove.set_active(True)


# Connect the mouse press event
fig.canvas.mpl_connect("button_press_event", on_mouse_press)

plt.tight_layout()
plt.show()
