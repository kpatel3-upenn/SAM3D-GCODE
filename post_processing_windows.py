import open3d as o3d
import tkinter as tk
from queue import Queue
from threading import Thread
import numpy as np
import time
import copy

# Thread-safe queue for communication
command_queue = Queue()
undo_stack = []  # Stack to keep track of point cloud changes for undo functionality

def generate_spherical_point_cloud(radius=1, points=1000):
    """Generate a spherical point cloud."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    pcd = mesh.sample_points_uniformly(number_of_points=points)
    return pcd

def process_point_cloud_commands(vis, pcd=None):
    """Handle incoming commands to manipulate the point cloud."""
    if pcd is None:
        pcd = generate_spherical_point_cloud()  # Generate initial point cloud
    vis.add_geometry(pcd)
    while True:
        time.sleep(0.05)
        if not command_queue.empty():
            command, value = command_queue.get()
            if command in ['downsample', 'remove_outliers']:
                undo_stack.append(copy.deepcopy(pcd))
            if command == 'downsample':
                pcd = pcd.uniform_down_sample(every_k_points=int(value))
            elif command == 'remove_outliers':
                n_neighbors, radius = value
                _, ind = pcd.remove_radius_outlier(nb_points=int(n_neighbors), radius=float(radius))
                pcd = pcd.select_by_index(ind)
            elif command == 'background':
                vis.get_render_option().background_color = np.array(value)
            elif command == 'undo':
                if undo_stack:
                    pcd = undo_stack.pop()
            elif command == 'save_and_close':
                pcd_path = "saved_point_cloud.ply"
                o3d.io.write_point_cloud(pcd_path, pcd)
                print(f"Point cloud saved to {pcd_path}")
                vis.destroy_window()
                break
            elif command == 'close':
                vis.destroy_window()
                break
            vis.clear_geometries()
            vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

def create_open3d_window(pcd=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    process_point_cloud_commands(vis, pcd)

def update_slider_from_entry(slider, entry, min_val, max_val):
    """ Update slider value from entry, ensuring it is within valid range. """
    try:
        value = int(entry.get())
        if min_val <= value <= max_val:
            slider.set(value)  # Synchronize slider position with entry
        else:
            entry.delete(0, tk.END)
            entry.insert(0, str(slider.get()))
    except ValueError:
        entry.delete(0, tk.END)
        entry.insert(0, str(slider.get()))

def on_enter(e):
    e.widget['background'] = 'gray'

def on_leave(e):
    e.widget['background'] = 'SystemButtonFace'

def setup_tkinter():
    root = tk.Tk()
    root.title("Point Cloud Control Panel")
    root.geometry("1300x300")  # Set a larger default size for the Tkinter window

    frame_top = tk.Frame(root)
    frame_top.pack(fill=tk.X, padx=10, pady=10)
    frame_middle = tk.Frame(root)
    frame_middle.pack(fill=tk.X, padx=10, pady=10)
    frame_bottom = tk.Frame(root)
    frame_bottom.pack(fill=tk.X, padx=10, pady=10)

    # Downsampling control with Entry
    tk.Label(frame_top, text="Downsample every k points:").pack(side=tk.LEFT)
    downsample_var = tk.IntVar(value=2)
    downsample_entry = tk.Entry(frame_top, width=5, textvariable=downsample_var)
    downsample_entry.pack(side=tk.LEFT, padx=5)
    downsample_slider = tk.Scale(frame_top, from_=1, to=10, orient='horizontal', variable=downsample_var)
    downsample_slider.pack(side=tk.LEFT, padx=5)
    downsample_entry.bind('<Return>', lambda e: update_slider_from_entry(downsample_slider, downsample_entry, 1, 10))
    b = tk.Button(frame_top, text="Apply Downsampling", command=lambda: command_queue.put(('downsample', downsample_var.get())))
    b.pack(side=tk.LEFT)
    b.bind("<Enter>", on_enter)
    b.bind("<Leave>", on_leave)

    # Outlier removal controls with Entry
    tk.Label(frame_middle, text="Outlier removal n neighbors:").pack(side=tk.LEFT)
    n_neighbors_var = tk.IntVar(value=20)
    n_neighbors_entry = tk.Entry(frame_middle, width=5, textvariable=n_neighbors_var)
    n_neighbors_entry.pack(side=tk.LEFT, padx=5)
    n_neighbors_slider = tk.Scale(frame_middle, from_=1, to=50, orient='horizontal', variable=n_neighbors_var)
    n_neighbors_slider.pack(side=tk.LEFT, padx=5)
    n_neighbors_entry.bind('<Return>', lambda e: update_slider_from_entry(n_neighbors_slider, n_neighbors_entry, 1, 50))
    b = tk.Button(frame_middle, text="Apply n Neighbors", command=lambda: command_queue.put(('remove_outliers', (n_neighbors_var.get(), 0.05))))
    b.pack(side=tk.LEFT)
    b.bind("<Enter>", on_enter)
    b.bind("<Leave>", on_leave)

    # Background color and other controls
    b = tk.Button(frame_bottom, text="White Background", command=lambda: command_queue.put(('background', [1, 1, 1])))
    b.pack(side=tk.LEFT)
    b.bind("<Enter>", on_enter)
    b.bind("<Leave>", on_leave)

    b = tk.Button(frame_bottom, text="Black Background", command=lambda: command_queue.put(('background', [0, 0, 0])))
    b.pack(side=tk.LEFT)
    b.bind("<Enter>", on_enter)
    b.bind("<Leave>", on_leave)

    b = tk.Button(frame_bottom, text="Undo", command=lambda: command_queue.put(('undo', None)))
    b.pack(side=tk.LEFT)
    b.bind("<Enter>", on_enter)
    b.bind("<Leave>", on_leave)

    b = tk.Button(frame_bottom, text="Save and Close Visualization", command=lambda: command_queue.put(('save_and_close', None)))
    b.pack(side=tk.LEFT)
    b.bind("<Enter>", on_enter)
    b.bind("<Leave>", on_leave)

    b = tk.Button(frame_bottom, text="Close Visualization", command=lambda: command_queue.put(('close', None)))
    b.pack(side=tk.LEFT)
    b.bind("<Enter>", on_enter)
    b.bind("<Leave>", on_leave)

    b = tk.Button(frame_bottom, text="Close Window", command=root.destroy)
    b.pack(side=tk.LEFT)
    b.bind("<Enter>", on_enter)
    b.bind("<Leave>", on_leave)

    root.mainloop()

if __name__ == "__main__":
    # Start Open3D in a separate thread
    open3d_thread = Thread(target=create_open3d_window)
    open3d_thread.start()

    # Start Tkinter in the main thread
    setup_tkinter()
