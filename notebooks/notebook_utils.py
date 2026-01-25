"""
Visualization utilities for BSL data exploration notebooks.

Provides common plotting functions for hand landmarks, comparisons,
and tolerance visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Literal, Union
import sys

# Add parent to path for imports
sys.path.insert(0, '..')

from src.landmarks import (
    LANDMARK_CONNECTIONS,
    LANDMARK_NAMES,
    FINGERTIP_INDICES,
    WRIST,
    THUMB_TIP,
    INDEX_FINGER_TIP,
    MIDDLE_FINGER_TIP,
    RING_FINGER_TIP,
    PINKY_TIP,
)
from src.types import Point3D


# =============================================================================
# Color Schemes
# =============================================================================

FINGER_COLORS = {
    'thumb': '#FF6B6B',      # Red
    'index': '#4ECDC4',      # Teal
    'middle': '#45B7D1',     # Blue
    'ring': '#96CEB4',       # Green
    'pinky': '#FFEAA7',      # Yellow
    'palm': '#DDA0DD',       # Plum
}

LANDMARK_COLORS = {
    0: FINGER_COLORS['palm'],     # Wrist
    1: FINGER_COLORS['thumb'],    # Thumb CMC
    2: FINGER_COLORS['thumb'],    # Thumb MCP
    3: FINGER_COLORS['thumb'],    # Thumb IP
    4: FINGER_COLORS['thumb'],    # Thumb TIP
    5: FINGER_COLORS['index'],    # Index MCP
    6: FINGER_COLORS['index'],    # Index PIP
    7: FINGER_COLORS['index'],    # Index DIP
    8: FINGER_COLORS['index'],    # Index TIP
    9: FINGER_COLORS['middle'],   # Middle MCP
    10: FINGER_COLORS['middle'],  # Middle PIP
    11: FINGER_COLORS['middle'],  # Middle DIP
    12: FINGER_COLORS['middle'],  # Middle TIP
    13: FINGER_COLORS['ring'],    # Ring MCP
    14: FINGER_COLORS['ring'],    # Ring PIP
    15: FINGER_COLORS['ring'],    # Ring DIP
    16: FINGER_COLORS['ring'],    # Ring TIP
    17: FINGER_COLORS['pinky'],   # Pinky MCP
    18: FINGER_COLORS['pinky'],   # Pinky PIP
    19: FINGER_COLORS['pinky'],   # Pinky DIP
    20: FINGER_COLORS['pinky'],   # Pinky TIP
}


# =============================================================================
# Data Extraction Helpers
# =============================================================================

def landmarks_to_arrays(landmarks: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert landmarks list to numpy arrays.

    Args:
        landmarks: List of Point3D or dicts with x, y, z

    Returns:
        Tuple of (x, y, z) arrays
    """
    if not landmarks:
        return np.array([]), np.array([]), np.array([])

    if hasattr(landmarks[0], 'x'):
        # Point3D objects
        x = np.array([p.x for p in landmarks])
        y = np.array([p.y for p in landmarks])
        z = np.array([p.z for p in landmarks])
    else:
        # Dict objects
        x = np.array([p['x'] for p in landmarks])
        y = np.array([p['y'] for p in landmarks])
        z = np.array([p['z'] for p in landmarks])

    return x, y, z


def get_connection_segments(landmarks: list) -> list[tuple]:
    """
    Get line segments for connections between landmarks.

    Args:
        landmarks: List of landmarks

    Returns:
        List of ((x1, y1, z1), (x2, y2, z2)) tuples
    """
    x, y, z = landmarks_to_arrays(landmarks)
    segments = []

    for i, j in LANDMARK_CONNECTIONS:
        if i < len(x) and j < len(x):
            segments.append((
                (x[i], y[i], z[i]),
                (x[j], y[j], z[j])
            ))

    return segments


# =============================================================================
# 3D Plotting
# =============================================================================

def plot_hand_3d(
    landmarks: list,
    ax: Optional[plt.Axes] = None,
    title: str = "Hand Landmarks",
    show_labels: bool = False,
    show_connections: bool = True,
    point_size: int = 50,
    figsize: tuple = (10, 8),
    elev: float = 20,
    azim: float = 45,
) -> plt.Axes:
    """
    Plot hand landmarks in 3D.

    Args:
        landmarks: List of Point3D or dicts
        ax: Existing axis to plot on
        title: Plot title
        show_labels: Show landmark index labels
        show_connections: Draw connections between landmarks
        point_size: Size of landmark points
        figsize: Figure size if creating new figure
        elev: Elevation angle
        azim: Azimuth angle

    Returns:
        Matplotlib 3D axis
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    x, y, z = landmarks_to_arrays(landmarks)

    # Plot connections first (so points are on top)
    if show_connections:
        for (p1, p2) in get_connection_segments(landmarks):
            ax.plot3D(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                'gray', alpha=0.5, linewidth=1.5
            )

    # Plot points with colors
    for i in range(len(x)):
        color = LANDMARK_COLORS.get(i, 'gray')
        ax.scatter(x[i], y[i], z[i], c=color, s=point_size, edgecolors='black', linewidth=0.5)

        if show_labels:
            ax.text(x[i], y[i], z[i], f' {i}', fontsize=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio
    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) / 2
    mid_x = (x.max() + x.min()) / 2
    mid_y = (y.max() + y.min()) / 2
    mid_z = (z.max() + z.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return ax


def plot_hand_2d(
    landmarks: list,
    view: Literal['front', 'side', 'top'] = 'front',
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_labels: bool = False,
    show_connections: bool = True,
    point_size: int = 50,
    figsize: tuple = (8, 8),
) -> plt.Axes:
    """
    Plot hand landmarks in 2D from different views.

    Args:
        landmarks: List of Point3D or dicts
        view: 'front' (XY), 'side' (ZY), or 'top' (XZ)
        ax: Existing axis to plot on
        title: Plot title (auto-generated if None)
        show_labels: Show landmark index labels
        show_connections: Draw connections between landmarks
        point_size: Size of landmark points
        figsize: Figure size if creating new figure

    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x, y, z = landmarks_to_arrays(landmarks)

    # Select coordinates based on view
    if view == 'front':
        coords_x, coords_y = x, y
        xlabel, ylabel = 'X', 'Y'
    elif view == 'side':
        coords_x, coords_y = z, y
        xlabel, ylabel = 'Z', 'Y'
    else:  # top
        coords_x, coords_y = x, z
        xlabel, ylabel = 'X', 'Z'

    # Plot connections
    if show_connections:
        for i, j in LANDMARK_CONNECTIONS:
            if i < len(coords_x) and j < len(coords_x):
                ax.plot(
                    [coords_x[i], coords_x[j]],
                    [coords_y[i], coords_y[j]],
                    'gray', alpha=0.5, linewidth=1.5
                )

    # Plot points
    for i in range(len(coords_x)):
        color = LANDMARK_COLORS.get(i, 'gray')
        ax.scatter(coords_x[i], coords_y[i], c=color, s=point_size,
                   edgecolors='black', linewidth=0.5, zorder=10)

        if show_labels:
            ax.annotate(str(i), (coords_x[i], coords_y[i]), fontsize=8,
                       xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'Hand Landmarks ({view.title()} View)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax


def plot_hand_views(
    landmarks: list,
    title: str = "Hand Landmarks - All Views",
    figsize: tuple = (16, 5),
) -> plt.Figure:
    """
    Plot hand from all three 2D views side by side.

    Args:
        landmarks: List of Point3D or dicts
        title: Overall figure title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_hand_2d(landmarks, view='front', ax=axes[0], title='Front (XY)')
    plot_hand_2d(landmarks, view='side', ax=axes[1], title='Side (ZY)')
    plot_hand_2d(landmarks, view='top', ax=axes[2], title='Top (XZ)')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


# =============================================================================
# Comparison Plotting
# =============================================================================

def plot_comparison(
    original: list,
    normalized: list,
    title: str = "Original vs Normalized",
    figsize: tuple = (16, 6),
) -> plt.Figure:
    """
    Plot original and normalized landmarks side by side.

    Args:
        original: Original landmarks
        normalized: Normalized landmarks
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(121, projection='3d')
    plot_hand_3d(original, ax=ax1, title='Original')

    ax2 = fig.add_subplot(122, projection='3d')
    plot_hand_3d(normalized, ax=ax2, title='Normalized')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


def plot_comparison_overlay(
    landmarks1: list,
    landmarks2: list,
    labels: tuple = ('Set 1', 'Set 2'),
    title: str = "Landmark Comparison",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Plot two sets of landmarks overlaid.

    Args:
        landmarks1: First set of landmarks
        landmarks2: Second set of landmarks
        labels: Labels for each set
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    x1, y1, z1 = landmarks_to_arrays(landmarks1)
    x2, y2, z2 = landmarks_to_arrays(landmarks2)

    # Plot first set
    ax.scatter(x1, y1, z1, c='blue', s=50, alpha=0.7, label=labels[0])

    # Plot second set
    ax.scatter(x2, y2, z2, c='red', s=50, alpha=0.7, label=labels[1])

    # Draw lines connecting corresponding points
    for i in range(min(len(x1), len(x2))):
        ax.plot3D([x1[i], x2[i]], [y1[i], y2[i]], [z1[i], z2[i]],
                  'gray', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    return fig


def plot_multiple_hands(
    landmarks_list: list,
    titles: Optional[list] = None,
    ncols: int = 3,
    figsize: tuple = (15, 5),
) -> plt.Figure:
    """
    Plot multiple hand poses in a grid.

    Args:
        landmarks_list: List of landmark sets
        titles: List of titles for each subplot
        ncols: Number of columns
        figsize: Figure size per row

    Returns:
        Matplotlib figure
    """
    n = len(landmarks_list)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(figsize[0], figsize[1] * nrows))

    for i, landmarks in enumerate(landmarks_list):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        title = titles[i] if titles and i < len(titles) else f'Pose {i + 1}'
        plot_hand_3d(landmarks, ax=ax, title=title)

    plt.tight_layout()
    return fig


# =============================================================================
# Tolerance Visualization
# =============================================================================

def plot_tolerance_region(
    canonical: list,
    tolerances: dict,
    ax: Optional[plt.Axes] = None,
    title: str = "Canonical Pose with Tolerances",
    view: Literal['front', 'side', 'top'] = 'front',
    figsize: tuple = (10, 10),
) -> plt.Axes:
    """
    Plot canonical pose with tolerance regions.

    Args:
        canonical: Canonical landmarks
        tolerances: Dict with 'position' tolerance
        ax: Existing axis to plot on
        title: Plot title
        view: View direction
        figsize: Figure size

    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x, y, z = landmarks_to_arrays(canonical)

    # Select coordinates based on view
    if view == 'front':
        coords_x, coords_y = x, y
    elif view == 'side':
        coords_x, coords_y = z, y
    else:  # top
        coords_x, coords_y = x, z

    # Get tolerance radius
    tol = tolerances.get('position', 0.02)

    # Plot tolerance circles
    for i in range(len(coords_x)):
        circle = plt.Circle(
            (coords_x[i], coords_y[i]),
            tol,
            fill=True,
            alpha=0.2,
            color=LANDMARK_COLORS.get(i, 'gray')
        )
        ax.add_patch(circle)

    # Plot canonical points on top
    plot_hand_2d(canonical, view=view, ax=ax, title=title, point_size=30)

    return ax


def plot_variance_heatmap(
    samples: list[list],
    title: str = "Landmark Variance Heatmap",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot variance of landmarks across samples as heatmap.

    Args:
        samples: List of landmark sets
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if not samples:
        raise ValueError("No samples provided")

    # Stack all samples
    all_coords = []
    for landmarks in samples:
        x, y, z = landmarks_to_arrays(landmarks)
        all_coords.append(np.stack([x, y, z], axis=1))

    stacked = np.stack(all_coords, axis=0)  # (n_samples, n_landmarks, 3)

    # Calculate variance
    variance = np.var(stacked, axis=0)  # (n_landmarks, 3)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Variance per landmark per coordinate
    im1 = axes[0].imshow(variance.T, aspect='auto', cmap='YlOrRd')
    axes[0].set_xlabel('Landmark Index')
    axes[0].set_ylabel('Coordinate')
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(['X', 'Y', 'Z'])
    axes[0].set_title('Variance by Coordinate')
    plt.colorbar(im1, ax=axes[0])

    # Total variance per landmark
    total_var = np.sum(variance, axis=1)
    colors = [LANDMARK_COLORS.get(i, 'gray') for i in range(len(total_var))]
    axes[1].bar(range(len(total_var)), total_var, color=colors)
    axes[1].set_xlabel('Landmark Index')
    axes[1].set_ylabel('Total Variance')
    axes[1].set_title('Total Variance per Landmark')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


# =============================================================================
# Distribution Plots
# =============================================================================

def plot_landmark_distributions(
    samples: list[list],
    landmark_indices: Optional[list] = None,
    figsize: tuple = (15, 10),
) -> plt.Figure:
    """
    Plot distributions of landmark coordinates.

    Args:
        samples: List of landmark sets
        landmark_indices: Specific landmarks to plot (None = fingertips)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if landmark_indices is None:
        landmark_indices = list(FINGERTIP_INDICES)

    n_landmarks = len(landmark_indices)
    fig, axes = plt.subplots(n_landmarks, 3, figsize=figsize)

    if n_landmarks == 1:
        axes = axes.reshape(1, -1)

    for row, lm_idx in enumerate(landmark_indices):
        # Collect coordinates for this landmark
        x_vals, y_vals, z_vals = [], [], []

        for landmarks in samples:
            if lm_idx < len(landmarks):
                x, y, z = landmarks_to_arrays([landmarks[lm_idx]])
                x_vals.extend(x)
                y_vals.extend(y)
                z_vals.extend(z)

        color = LANDMARK_COLORS.get(lm_idx, 'gray')
        name = LANDMARK_NAMES.get(lm_idx, f'Landmark {lm_idx}')

        axes[row, 0].hist(x_vals, bins=30, color=color, alpha=0.7)
        axes[row, 0].set_ylabel(name)
        axes[row, 0].set_xlabel('X')

        axes[row, 1].hist(y_vals, bins=30, color=color, alpha=0.7)
        axes[row, 1].set_xlabel('Y')

        axes[row, 2].hist(z_vals, bins=30, color=color, alpha=0.7)
        axes[row, 2].set_xlabel('Z')

    fig.suptitle('Landmark Coordinate Distributions', fontsize=14)
    plt.tight_layout()

    return fig


def plot_distance_distribution(
    samples: list[list],
    reference: Optional[list] = None,
    title: str = "Distance from Reference",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot distribution of distances from reference pose.

    Args:
        samples: List of landmark sets
        reference: Reference landmarks (None = use mean)
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Calculate reference if not provided
    if reference is None:
        all_coords = []
        for landmarks in samples:
            x, y, z = landmarks_to_arrays(landmarks)
            all_coords.append(np.stack([x, y, z], axis=1))
        stacked = np.stack(all_coords, axis=0)
        ref_coords = np.mean(stacked, axis=0)
    else:
        x, y, z = landmarks_to_arrays(reference)
        ref_coords = np.stack([x, y, z], axis=1)

    # Calculate distances for each sample
    distances = []
    for landmarks in samples:
        x, y, z = landmarks_to_arrays(landmarks)
        coords = np.stack([x, y, z], axis=1)
        dist = np.sqrt(np.sum((coords - ref_coords) ** 2, axis=1))
        distances.append(dist)

    distances = np.array(distances)  # (n_samples, n_landmarks)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Per-landmark box plot
    axes[0].boxplot(distances, vert=True)
    axes[0].set_xlabel('Landmark Index')
    axes[0].set_ylabel('Distance')
    axes[0].set_title('Distance Distribution per Landmark')

    # Overall histogram
    all_distances = distances.flatten()
    axes[1].hist(all_distances, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(all_distances), color='red', linestyle='--',
                    label=f'Mean: {np.mean(all_distances):.4f}')
    axes[1].axvline(np.median(all_distances), color='green', linestyle='--',
                    label=f'Median: {np.median(all_distances):.4f}')
    axes[1].set_xlabel('Distance')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Overall Distance Distribution')
    axes[1].legend()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig


# =============================================================================
# Angle Visualization
# =============================================================================

def plot_joint_angles(
    angles: dict,
    title: str = "Joint Angles",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot joint angles as bar chart.

    Args:
        angles: Dict with finger -> {mcp, pip, dip} structure
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    joints = ['mcp', 'pip', 'dip']

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(fingers))
    width = 0.25

    for i, joint in enumerate(joints):
        values = []
        for finger in fingers:
            if finger in angles and joint in angles[finger]:
                values.append(angles[finger][joint])
            else:
                values.append(0)

        ax.bar(x + i * width, values, width, label=joint.upper())

    ax.set_xlabel('Finger')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f.title() for f in fingers])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add reference lines for anatomical limits
    ax.axhline(90, color='red', linestyle=':', alpha=0.5, label='90°')
    ax.axhline(180, color='red', linestyle=':', alpha=0.5)

    plt.tight_layout()
    return fig


def plot_angle_comparison(
    angles_list: list[dict],
    labels: list[str],
    title: str = "Angle Comparison",
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Compare joint angles across multiple samples.

    Args:
        angles_list: List of angle dicts
        labels: Labels for each sample
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    joints = ['mcp', 'pip', 'dip']

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for j, joint in enumerate(joints):
        x = np.arange(len(fingers))
        width = 0.8 / len(angles_list)

        for i, (angles, label) in enumerate(zip(angles_list, labels)):
            values = []
            for finger in fingers:
                if finger in angles and joint in angles[finger]:
                    values.append(angles[finger][joint])
                else:
                    values.append(0)

            axes[j].bar(x + i * width, values, width, label=label, alpha=0.7)

        axes[j].set_xlabel('Finger')
        axes[j].set_ylabel('Angle (degrees)')
        axes[j].set_title(f'{joint.upper()} Joint')
        axes[j].set_xticks(x + width * len(angles_list) / 2)
        axes[j].set_xticklabels([f.title() for f in fingers])
        axes[j].legend()
        axes[j].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


# =============================================================================
# Quality Visualization
# =============================================================================

def plot_quality_scores(
    scores: list[float],
    labels: Optional[list[str]] = None,
    title: str = "Quality Scores",
    threshold: float = 0.5,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot quality scores with threshold line.

    Args:
        scores: List of quality scores
        labels: Labels for each score
        title: Figure title
        threshold: Quality threshold line
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar chart
    x = range(len(scores))
    colors = ['green' if s >= threshold else 'red' for s in scores]
    axes[0].bar(x, scores, color=colors, alpha=0.7)
    axes[0].axhline(threshold, color='orange', linestyle='--', label=f'Threshold: {threshold}')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Quality Score')
    axes[0].set_title('Quality Scores by Sample')
    axes[0].legend()

    if labels:
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha='right')

    # Histogram
    axes[1].hist(scores, bins=20, alpha=0.7, edgecolor='black')
    axes[1].axvline(threshold, color='orange', linestyle='--', label=f'Threshold: {threshold}')
    axes[1].axvline(np.mean(scores), color='blue', linestyle='-', label=f'Mean: {np.mean(scores):.2f}')
    axes[1].set_xlabel('Quality Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Quality Score Distribution')
    axes[1].legend()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


# =============================================================================
# Animation Helpers
# =============================================================================

def create_rotation_frames(
    landmarks: list,
    n_frames: int = 36,
) -> list[tuple]:
    """
    Create frames for rotating 3D view.

    Args:
        landmarks: Landmarks to visualize
        n_frames: Number of rotation frames

    Returns:
        List of (elevation, azimuth) tuples
    """
    frames = []
    for i in range(n_frames):
        azim = i * 360 / n_frames
        frames.append((20, azim))
    return frames


# =============================================================================
# Utility Functions
# =============================================================================

def set_notebook_style():
    """Set matplotlib style for notebooks."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


def create_sample_landmarks() -> list[Point3D]:
    """Create sample landmarks for testing visualization."""
    # Simple flat hand for testing
    landmarks = []

    # Wrist
    landmarks.append(Point3D(x=0.5, y=0.8, z=0.0))

    # Thumb (indices 1-4)
    landmarks.append(Point3D(x=0.35, y=0.75, z=0.02))
    landmarks.append(Point3D(x=0.28, y=0.68, z=0.03))
    landmarks.append(Point3D(x=0.22, y=0.60, z=0.03))
    landmarks.append(Point3D(x=0.18, y=0.52, z=0.03))

    # Index finger (indices 5-8)
    landmarks.append(Point3D(x=0.42, y=0.55, z=0.0))
    landmarks.append(Point3D(x=0.40, y=0.42, z=0.0))
    landmarks.append(Point3D(x=0.39, y=0.32, z=0.0))
    landmarks.append(Point3D(x=0.38, y=0.22, z=0.0))

    # Middle finger (indices 9-12)
    landmarks.append(Point3D(x=0.50, y=0.53, z=0.0))
    landmarks.append(Point3D(x=0.50, y=0.38, z=0.0))
    landmarks.append(Point3D(x=0.50, y=0.26, z=0.0))
    landmarks.append(Point3D(x=0.50, y=0.15, z=0.0))

    # Ring finger (indices 13-16)
    landmarks.append(Point3D(x=0.58, y=0.55, z=0.0))
    landmarks.append(Point3D(x=0.60, y=0.42, z=0.0))
    landmarks.append(Point3D(x=0.61, y=0.32, z=0.0))
    landmarks.append(Point3D(x=0.62, y=0.23, z=0.0))

    # Pinky (indices 17-20)
    landmarks.append(Point3D(x=0.65, y=0.60, z=0.0))
    landmarks.append(Point3D(x=0.70, y=0.52, z=0.0))
    landmarks.append(Point3D(x=0.74, y=0.45, z=0.0))
    landmarks.append(Point3D(x=0.78, y=0.40, z=0.0))

    return landmarks
