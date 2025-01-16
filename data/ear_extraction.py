import numpy as np
import open3d as o3d
import os
from sklearn.neighbors import NearestNeighbors


def find_ear_centers_basic(points):
    """Find ear centers based on comparing y and z ranges"""
    # Filter for head region
    head_mask = points[:, 0] > -50
    head_points = points[head_mask]
    
    # Find min and max for y and z
    y_values = head_points[:, 1]
    z_values = head_points[:, 2]
    
    y_min_idx = np.argmin(y_values)
    y_max_idx = np.argmax(y_values)
    z_min_idx = np.argmin(z_values)
    z_max_idx = np.argmax(z_values)
    
    # Compare ranges
    y_range = abs(y_values[y_max_idx] - y_values[y_min_idx])
    z_range = abs(z_values[z_max_idx] - z_values[z_min_idx])

    # Choose centers based on smaller range
    if y_range > z_range:
        left_center = head_points[z_min_idx]
        right_center = head_points[z_max_idx]
    else:
        left_center = head_points[y_min_idx]
        right_center = head_points[y_max_idx]
    
    return left_center, right_center

def find_ear_centers(points):
    """Find ear centers using head symmetry"""
    # 1. Filter head region (assuming head is above neck)
    head_mask = points[:, 0] > -50  # Adjust threshold as needed
    head_points = points[head_mask]
    
    # 2. Find symmetry plane using PCA
    mean = np.mean(head_points, axis=0)
    centered = head_points - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort eigenvectors by eigenvalues in descending order
    sort_idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_idx]
    
    # First two eigenvectors define the symmetry plane
    # Third eigenvector (smallest variation) is normal to symmetry plane
    symmetry_normal = eigenvectors[:, 2]
    
    # 3. Project points onto symmetry plane normal
    projected = np.dot(centered, symmetry_normal)
    
    # 4. Split into left/right halves
    left_mask = projected < 0
    right_mask = projected >= 0
    
    left_half = head_points[left_mask]
    right_half = head_points[right_mask]
    
    # 5. Find ear candidates using local density and curvature
    def find_ear_in_half(half_points):
        # Use KNN to estimate local density
        nbrs = NearestNeighbors(n_neighbors=30).fit(half_points)
        distances, _ = nbrs.kneighbors(half_points)
        density = 1 / np.mean(distances, axis=1)
        
        # Find points with lower density (ears typically protrude)
        density_threshold = np.percentile(density, 20)  # Bottom 20%
        candidate_mask = density < density_threshold
        candidates = half_points[candidate_mask]
        
        # Find most lateral point among candidates
        lateral_scores = np.dot(candidates - mean, symmetry_normal)
        ear_idx = np.argmax(np.abs(lateral_scores))
        return candidates[ear_idx]
    
    left_ear = find_ear_in_half(left_half)
    right_ear = find_ear_in_half(right_half)
    
    return left_ear, right_ear

def create_ear_ranges(center, r=60):
    """Create bounding box range around ear center"""
    return {
        'x_min': center[0] - r,
        'x_max': center[0] + r,
        'y_min': center[1] - r,
        'y_max': center[1] + r,
        'z_min': center[2] - r,
        'z_max': center[2] + r
    }


# Create the output directory if it doesn't exist
output_dir = "/autofs/thau04b/hghallab/comp/Huawei/TechArena20241016/ears"
os.makedirs(output_dir, exist_ok=True)

def create_mask(points, coord_range):
    # Create boolean masks for each coordinate axis
    x_mask = (points[:, 0] >= coord_range['x_min']) & (points[:, 0] <= coord_range['x_max'])
    y_mask = (points[:, 1] >= coord_range['y_min']) & (points[:, 1] <= coord_range['y_max'])
    z_mask = (points[:, 2] >= coord_range['z_min']) & (points[:, 2] <= coord_range['z_max'])
    
    # Combine masks to get the final mask
    mask = x_mask & y_mask & z_mask
    return mask

# Iterate over all patients
# patient_ids = [
#     4, 38, 68, 84, 99, 102, 114, 122, 134, 135, 136, 143, 145, 146, 150, 151, 164, 
#     173, 175, 183, 187, 189, 190, 191, 202, 216, 217, 220, 228, 230, 231]

# for patient_id in patient_ids:
#     patient_str = f"P{patient_id:04d}"
#     folder_range_start = (patient_id - 1) // 10 * 10 + 1
#     folder_range_end = folder_range_start + 9
#     folder_str = f"P{folder_range_start:04d}-P{folder_range_end:04d}"
#     point_cloud_file = f"/autofs/thau04b/hghallab/comp/Huawei/TechArena20241016/data/{folder_str}/{patient_str}/3DSCAN/{patient_str}_Project1.asc"
#     if not os.path.exists(point_cloud_file):
#         print(f"File not found: {point_cloud_file}")
#         continue
    
#     try:
#         # Load point cloud
#         point_data = np.loadtxt(point_cloud_file)
#         points = point_data[:, :3]
#         normals = point_data[:, 3:]
        
#         # Find ear centers and create ranges
# #         left_center, right_center = find_ear_centers(points)
# #         left_ear_range = create_ear_ranges(left_center)
#         right_ear_range = create_ear_ranges(right_center)
        
#         # Create masks and filter points
#         left_mask = create_mask(points, left_ear_range)
#         right_mask = create_mask(points, right_ear_range)
        
#         left_points = points[left_mask]
#         left_normals = normals[left_mask]
#         right_points = points[right_mask]
#         right_normals = normals[right_mask]
        
#         # Create and save point clouds
#         left_point_cloud = o3d.geometry.PointCloud()
#         left_point_cloud.points = o3d.utility.Vector3dVector(left_points)
#         left_point_cloud.normals = o3d.utility.Vector3dVector(left_normals)
        
#         right_point_cloud = o3d.geometry.PointCloud()
#         right_point_cloud.points = o3d.utility.Vector3dVector(right_points)
#         right_point_cloud.normals = o3d.utility.Vector3dVector(right_normals)
        
#         # Save point clouds
#         left_output_path = os.path.join(output_dir, f"{patient_str}_left.ply")
#         right_output_path = os.path.join(output_dir, f"{patient_str}_right.ply")
        
#         o3d.io.write_point_cloud(left_output_path, left_point_cloud)
#         o3d.io.write_point_cloud(right_output_path, right_point_cloud)
        
#         print(f"Processed {patient_str}")
        
#     except Exception as e:
#         print(f"Error processing {patient_str}: {str(e)}")
#         continue

