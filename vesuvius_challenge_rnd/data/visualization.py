import numpy as np
import plotly.graph_objects as go
import trimesh
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import zoom


def create_gif_from_2d_single_channel_images(
    frames: np.ndarray,
    frame_dim: int = 0,
    out_path: str = "output.gif",
    cmap: str = "viridis",
    duration: float = 100,
    loop: int = 0,
    scale_factor: float = 1.0,
    downsample_factor: float = 1.0,
    downsample_order: int = 3,
):
    """
    Create an animated GIF of single-channel 2D images.

    Parameters:
        frames (np.ndarray): NumPy array representing the frames.
        frame_dim (int, optional): Dimension along which frames are stacked in the input array. Defaults to 0.
        cmap (str): Name of the colormap to use.
        out_path (str): Path to save the animated GIF.
        duration (float, optional): Duration of each frame in milliseconds. Defaults to 100.
        loop (int, optional): Number of loops. 0 for infinite. Defaults to 0.
        scale_factor (float, optional): Factor by which to scale the pixel values. Defaults to 1.0.
        downsample_factor (float, optional): Factor by which to downsample the H and W dimensions. Should be between 0 and 1.
        downsample_order (int, optional): The spline interpolation order for downsampling. Defaults to 3.
    """
    if len(frames.shape) != 3:
        raise ValueError("3D frames array expected to be in `THW` format.")
    elif not 0 < downsample_factor <= 1:
        raise ValueError("downsample_factor should be between 0 and 1.")
    elif not 0 < scale_factor <= 255:
        raise ValueError("scale_factor should be between 0 and 255.")

    frames = np.moveaxis(frames, frame_dim, 0)

    if downsample_factor < 1.0:
        zoom_factors = [1, downsample_factor, downsample_factor]
        frames = zoom(frames, zoom_factors, order=downsample_order)

    frames = (frames * scale_factor).astype(np.uint8, copy=False)

    colormap = colormaps[cmap]
    frames_colored = colormap(frames / 255.0)
    frames_colored = (frames_colored[:, :, :, :3] * 255).astype(np.uint8, copy=False)
    frames_pil = [Image.fromarray(frame, "RGB") for frame in frames_colored]

    frames_pil[0].save(
        out_path, save_all=True, append_images=frames_pil[1:], loop=loop, duration=duration
    )


def plot_rectangles(rectangles, stroke_width=2, edgecolor="r", xlim=None, ylim=None, ax=None):
    """
    Plots a list of rectangles with customizable x and y axis limits.

    Parameters:
    rectangles (list of tuples): List of rectangles defined by lower-left and upper-right corners.
    stroke_width (int): The width of the stroke for the rectangles.
    edgecolor (str): The color of the edges.
    xlim (tuple): A tuple (min, max) to set the limit for the x-axis.
    ylim (tuple): A tuple (min, max) to set the limit for the y-axis.
    ax (plt.Axes): The current axes.
    """
    if ax is None:
        ax = plt.gca()

    for rect in rectangles:
        (x1, y1), (x2, y2) = rect
        width, height = x2 - x1, y2 - y1
        rect_patch = Rectangle(
            (x1, y1), width, height, linewidth=stroke_width, edgecolor=edgecolor, facecolor="none"
        )
        ax.add_patch(rect_patch)

    # Set custom or calculated axis limits
    if xlim:
        ax.set_xlim(*xlim)
    else:
        all_x = [x for rect in rectangles for x in [rect[0][0], rect[1][0]]]
        ax.set_xlim(min(all_x), max(all_x))

    if ylim:
        ax.set_ylim(*ylim)
    else:
        all_y = [y for rect in rectangles for y in [rect[0][1], rect[1][1]]]
        ax.set_ylim(min(all_y), max(all_y))

    return ax


def show_mesh_without_texture(
    mesh: trimesh.Trimesh, title: str = "Mesh", intensity=None, colorscale=None, **mesh_kwargs
):
    vertices = mesh.vertices
    triangles = mesh.faces

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                opacity=0.5,
                intensity=intensity,
                colorscale=colorscale,
                **mesh_kwargs,
            )
        ],
    )

    fig.update_layout(title=dict(text=title))

    fig.show()
