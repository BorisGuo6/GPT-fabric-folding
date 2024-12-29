import numpy as np
import open3d as o3d
import numpy as np
from shapely.geometry import Point, Polygon, MultiPoint, LineString,MultiPolygon
from shapely.ops import triangulate, unary_union
import alphashape
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import os
import torch
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
import cv2
def dbscan_pcd(points):
    # 使用DBSCAN进行聚类
    db = DBSCAN(eps=0.15, min_samples=10).fit(points)#[:,[0,2]]
    
    # 获取标签 (-1表示噪点)
    labels = db.labels_
    
    # 过滤掉标签为 -1 的点
    return points[labels != -1]

import trimesh
def vertex_normal_continuity(faces, face_normals, vertices):
    """
    Analyze vertex normal continuity.
    
    Parameters:
    - faces: (n_faces, 3) array of vertex indices per face
    - face_normals: (n_faces, 3) array of face normals
    - vertices: (n_vertices, 3) array of vertex positions
    
    Returns:
    - continuity_scores: Per-vertex score representing normal continuity
    """
    # Create an empty list to store continuity per vertex
    continuity_scores = np.zeros(len(vertices))
    counts = np.zeros(len(vertices))  # Count for averaging

    # Iterate through each face and update vertices
    for i, face in enumerate(faces):
        for j, vertex in enumerate(face):
            # Compute dot products between this face normal and its neighbors
            neighbors = faces[np.any(faces == vertex, axis=1)]  # All faces sharing this vertex
            neighbor_normals = face_normals[np.any(faces == vertex, axis=1)]
            
            # Compute angle continuity between normals
            dot_products = np.clip(np.sum(face_normals[i] * neighbor_normals, axis=1), -1.0, 1.0)
            continuity_scores[vertex] += np.arccos(dot_products).sum()
            counts[vertex] += len(dot_products)

    # Average continuity for each vertex
    continuity_scores /= np.maximum(counts, 1)  # Avoid division by zero
    return continuity_scores
def analyze_mesh_smoothness(mesh):
    """
    Analyze the smoothness of a mesh.
    
    Parameters:
    - mesh: trimesh object
    
    Returns:
    - smoothness_report: A dictionary summarizing smoothness metrics
    """
    # Vertex normal continuity
    face_normals = mesh.face_normals
    faces = mesh.faces
    vertices = mesh.vertices
    # continuity_scores = vertex_normal_continuity(faces, face_normals, vertices)
    
    # Surface curvature
    vertex_curvature = mesh.vertex_defects
    # mesh.visual.vertex_colors = trimesh.visual.interpolate(vertex_curvature, color_map='jet')
    # mesh.show()


    # Compute metrics
    curvature = vertex_curvature
    mean_curvature = np.mean(curvature)
    mean_absolute_curvature = np.mean(np.abs(curvature))
    curvature_std = np.std(curvature)
    curvature_range = np.max(curvature) - np.min(curvature)
    low_curvature_percentage = np.sum(np.abs(curvature) < 0.1) / len(curvature) * 100

    # Create a detailed report
    curvature_report = {
        "Mean Curvature": mean_curvature,
        "Mean Absolute Curvature (MAC)": mean_absolute_curvature,
        "Curvature Standard Deviation": curvature_std,
        "Curvature Range": curvature_range,
        "Low Curvature Percentage": low_curvature_percentage
    }
    # Edge length uniformity
    # edge_lengths = mesh.edges_unique_length
    # edge_uniformity = edge_lengths.std() / edge_lengths.mean()  # Coefficient of Variation
    
    # Compile results
    smoothness_report = {
        # "Vertex Normal Continuity": {
        #     "min": continuity_scores.min(),
        #     "max": continuity_scores.max(),
        #     "mean": continuity_scores.mean(),
        # },
        "Surface Curvature": {
            "min": vertex_curvature.min(),
            "max": vertex_curvature.max(),
            "mean": vertex_curvature.mean(),
            "Mean Curvature": mean_curvature,
            "Mean Absolute Curvature (MAC)": mean_absolute_curvature,
            "Curvature Standard Deviation": curvature_std,
            "Curvature Range": curvature_range,
            "Low Curvature Percentage": low_curvature_percentage
        },
        # "Edge Length Uniformity (CoV)": edge_uniformity,
    }

    for key, value in smoothness_report.items():
        print(f"{key}: {value}")
    mesh.visual.vertex_colors = trimesh.visual.interpolate(vertex_curvature, color_map='jet')
    mesh.show()

    return smoothness_report
def check_planeness(vertices,faces):
    
    mesh = trimesh.Trimesh(vertices, faces, process=True)

    # Define ray origins (above each vertex, high on Z-axis)
    ray_origins = mesh.triangles_center + np.array([0, 1.0, 0])  # Offset above the triangles

    # Define ray directions (straight down)
    ray_directions = np.tile(np.array([0, -1, 0]), (len(ray_origins), 1))

    # Initialize the ray intersector
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    # Perform ray intersection
    hit_faces = intersector.intersects_id(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        return_locations=True,
    )

    # Calculate hit distances for all intersections
    hit_distances = np.linalg.norm(hit_faces[2] - ray_origins[hit_faces[1]], axis=1)

    # Create an array to store the first face hit per ray
    first_hit_faces = np.full(len(ray_origins), -1, dtype=np.int32)

    # Loop through the rays and find the closest hit for each
    for ray_idx in range(len(ray_origins)):
        # Find all hits for the current ray
        ray_hits = np.where(hit_faces[1] == ray_idx)[0]
        
        if len(ray_hits) > 0:
            # Get distances for the current ray's hits
            ray_hit_distances = hit_distances[ray_hits]
            
            # Find the closest hit
            closest_hit_idx = ray_hits[np.argmin(ray_hit_distances)]
            
            # Store the closest hit face
            first_hit_faces[ray_idx] = hit_faces[0][closest_hit_idx]
    mask=np.unique(first_hit_faces)
    print(len(mask),len(faces))
    print("nomal:",calculate_face_normals_similarity(vertices,faces[mask]))   
    # save_surface_visualization(vertices,faces,faces[mask])
    # return calculate_face_normals_similarity(vertices,faces[mask])
    surface_faces = faces[mask].copy()
    unique_vertex_indices, inverse_indices = np.unique(surface_faces.flatten(), return_inverse=True)

    # Map faces to the new vertex indices
    new_faces = inverse_indices.reshape(surface_faces.shape)

    # Extract the corresponding vertices
    new_vertices = vertices[unique_vertex_indices]
    surface_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    # Optional: Verify the integrity of the mesh
    # surface_mesh.fix_normals() 

    smoothness_report = analyze_mesh_smoothness(surface_mesh)

    # Print the smoothness report
    # for key, value in smoothness_report.items():
    #     print(f"{key}: {value}")
    return smoothness_report["Surface Curvature"]["mean"]

def load_cloth_mesh(path):
    """Load .obj of cloth mesh. Only triangular mesh is acceptable!
    Return:
        - vertices: ndarray, (N, 3)
        - triangle_faces: ndarray, (S, 3)
        - stretch_edges: ndarray, (M1, 2)
        - bend_edges: ndarray, (M2, 2)
        - shear_edges: ndarray, (M3, 2)
    This function was modified to handle triangular meshes.
    """

    vertices, faces = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # 3D vertex
        if line.startswith('v '):
            vertices.append([float(n)*0.1 for n in line.replace('v ', '').split(' ')])
        # Face
        elif line.startswith('f '):
            idx = [n.split('/') for n in line.replace('f ', '').split(' ')]
            face = [int(n[0]) - 1 for n in idx]
            # print(len(face))
            assert(len(face) == 3)  # Ensure the face is triangular
            faces.append(face)
    return np.array(faces,dtype=int)


def density_based_interpolation(points, num_new_points=4096):
    """
    根据局部密度在点云内生成更多点。
    :param points: 原始点集 (n, 2)
    :param num_new_points: 需要生成的新点数量
    :return: 扩充后的点集
    """
    # 使用DBSCAN进行聚类
    db = DBSCAN(eps=0.3, min_samples=10).fit(points)
    
    #获取标签 (-1表示噪点)
    labels = db.labels_
    
    # 过滤掉标签为 -1 的点
    points = points[labels != -1]

    # 生成凹壳边界
    alpha = 0.05
    boundary_polygon = alphashape.alphashape(points, alpha)

    # 计算点的局部密度
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    density_scores = np.mean(distances, axis=1)
    # 根据密度分数计算每个点需要生成的新点数
    total_density = np.sum(density_scores ** 2)
    # 初始化新点列表
    new_points = []

    for i, density in enumerate(density_scores):
        p = points[i]
        # 根据密度分数计算当前点需要生成的新点数量
        num_points_to_add = int((density**2 / total_density) * num_new_points)
        
        # 在当前点的邻域内生成新点，使用均匀分布
        for _ in range(num_points_to_add):
            # 使用均匀噪声在局部范围内生成新点
            jitter = np.random.uniform(low=-density, high=density, size=p.shape)
            new_point = p + jitter
            # 判断新生成的点是否在凹壳边界内
            if boundary_polygon.contains(Point(new_point[0], new_point[1])):
                new_points.append(new_point)
    
    # 提取新点坐标
    new_points_array = np.array(new_points)
    if new_points_array.shape[0]>0:
        expanded_points = np.vstack([points, new_points_array])
    else:
        expanded_points = points
    return expanded_points
def get_current_covered_area(pos, cloth_particle_radius: float = 0.00625):
    """
    Calculate the covered area by taking max x,y cood and min x,y 
    coord, create a discritized grid between the points
    :param pos: Current positions of the particle states
    """
    min_x = np.min(pos[:, 0])
    min_y = np.min(pos[:, 1])
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 1])
    # print(min_x," ",max_x," ",min_y," ",max_y," ")
    const = 30
    init = np.array([min_x, min_y])
    span = np.array([max_x - min_x, max_y - min_y]) / (const*1.0)
    pos2d = pos[:, [0, 1]]

    offset = pos2d - init
    slotted_x_low = np.maximum(np.round((offset[:, 0] - cloth_particle_radius) / span[0]).astype(int), 0)
    slotted_x_high = np.minimum(np.round((offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), const)
    slotted_y_low = np.maximum(np.round((offset[:, 1] - cloth_particle_radius) / span[1]).astype(int), 0)
    slotted_y_high = np.minimum(np.round((offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), const)
    # Method 1
    grid = np.zeros(const*const)  # Discretization
    listx = vectorized_range1(slotted_x_low, slotted_x_high)
    listy = vectorized_range1(slotted_y_low, slotted_y_high)
    listxx, listyy = vectorized_meshgrid1(listx, listy)
    idx = listxx * const + listyy
    idx = np.clip(idx.flatten(), 0, const*const-1)
    grid[idx] = 1
    return np.sum(grid) * span[0] * span[1]
    

def extend_point_cloud(points,num):    
        # 提取 XY 平面坐标
    points_xy = points[:, [0,2]]

    # 在 Alpha 形状内部生成新点
    num_new_points = int(num - len(points))
    if num_new_points <= 0:          
        return points

    expanded_points = density_based_interpolation(points_xy, num_new_points)
    # 将z轴补全
    new_dimension = np.zeros((expanded_points.shape[0], 1))
    
        
    return np.hstack((expanded_points, new_dimension))

def add_number(self, new_number):
    self.count += 1  # 更新计数器
    self.current_average += (new_number - self.current_average) / self.count  # 计算新的平均值
    return self.current_average  # 返回当前平均值

def vectorized_range1(start, end):
    """  Return an array of NxD, iterating from the start to the end"""
    N = int(np.max(end - start)) + 1
    idxes = np.floor(np.arange(N) * (end - start)
                    [:, None] / N + start[:, None]).astype('int')
    return idxes

def vectorized_meshgrid1(vec_x, vec_y):
    """vec_x in NxK, vec_y in NxD. Return xx in Nx(KxD) and yy in Nx(DxK)"""
    N, K, D = vec_x.shape[0], vec_x.shape[1], vec_y.shape[1]
    vec_x = np.tile(vec_x[:, None, :], [1, D, 1]).reshape(N, -1)
    vec_y = np.tile(vec_y[:, :, None], [1, 1, K]).reshape(N, -1)
    return vec_x, vec_y

def get_current_covered_area_loU(pos0, pos1 ,cloth_particle_radius: float = 0.00625):
    """
    Calculate the covered area by taking max x,y cood and min x,y 
    coord, create a discritized grid between the points
    :param pos: Current positions of the particle states
    """
    pos = np.concatenate((pos0,pos1))
    min_x = np.min(pos[:, 0])
    min_y = np.min(pos[:, 1])
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 1])
    # print(min_x," ",max_x," ",min_y," ",max_y," ")
    const = 30
    init = np.array([min_x, min_y])
    span = np.array([max_x - min_x, max_y - min_y]) / (const*1.0)
    pos2d = pos0[:, [0, 1]]

    offset = pos2d - init
    slotted_x_low = np.maximum(np.round((offset[:, 0] - cloth_particle_radius) / span[0]).astype(int), 0)
    slotted_x_high = np.minimum(np.round((offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), const)
    slotted_y_low = np.maximum(np.round((offset[:, 1] - cloth_particle_radius) / span[1]).astype(int), 0)
    slotted_y_high = np.minimum(np.round((offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), const)
    # Method 1
    grid0 = np.zeros(const*const)  # Discretization
    listx = vectorized_range1(slotted_x_low, slotted_x_high)
    listy = vectorized_range1(slotted_y_low, slotted_y_high)
    listxx, listyy = vectorized_meshgrid1(listx, listy)
    idx0 = listxx * const + listyy
    idx0 = np.clip(idx0.flatten(), 0, const*const-1)

    pos2d = pos1[:, [0, 1]]

    offset = pos2d - init
    slotted_x_low = np.maximum(np.round((offset[:, 0] - cloth_particle_radius) / span[0]).astype(int), 0)
    slotted_x_high = np.minimum(np.round((offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), const)
    slotted_y_low = np.maximum(np.round((offset[:, 1] - cloth_particle_radius) / span[1]).astype(int), 0)
    slotted_y_high = np.minimum(np.round((offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), const)
    # Method 1
    grid0 = np.zeros(const*const)  # Discretization
    listx = vectorized_range1(slotted_x_low, slotted_x_high)
    listy = vectorized_range1(slotted_y_low, slotted_y_high)
    listxx, listyy = vectorized_meshgrid1(listx, listy)
    idx1 = listxx * const + listyy
    idx1 = np.clip(idx1.flatten(), 0, const*const-1)
    grid0[idx0] = 1
    # path = CONST_PATH+str(self.cloth_id-1)+"/mesh/output.txt"
    # if self.cloth_id == 0:
    #     path = CONST_PATH+str(self.cloth_id)+"/mesh/output00.txt"
    # np.savetxt(path, np.array((grid0)).reshape(const,const),fmt="%d",delimiter='')
    # print("idx",np.sum(grid0))
    grid0[idx1] = 1
    # print("idx",np.sum(grid0))
    grid1 = np.zeros(const*const)
    # grid1[idx1] = 1
    # path = CONST_PATH+str(self.cloth_id-1)+"/mesh/output1.txt"
    # if self.cloth_id == 0:
    #     path = CONST_PATH+str(self.cloth_id)+"/mesh/output01.txt"
    # np.savetxt(path, np.array((grid1)).reshape(const,const),fmt="%d",delimiter='')
    grid1[idx1] = -1
    # print("idx",np.sum(grid1))
    for id in idx0:
        if int(grid1[id]) == -1:
            grid1[id] = 1
        else :
            if int(grid1[id]) == 0:
                grid1[id] = -2   

    grid1 = np.maximum(grid1,0)
    # print("loU:",np.sum(grid1),np.sum(grid0))
    return np.sum(grid1)/np.sum(grid0)

CALCULATE_MEAN_FIRST = True

def metrics_lou_mean(self, pos0, pos1):
    if CALCULATE_MEAN_FIRST == True:
        self.count = 0
        self.current_average = 0
        CALCULATE_MEAN_FIRST = False
    if(len(pos0)<6144):
        pos0 = extend_point_cloud(pos0)
    if(len(pos1)<6144):
        pos1 = extend_point_cloud(pos1)
    loU = get_current_covered_area_loU(pos0,pos1)
    add_number(loU)
    return self.current_average

def load_pcd_from_txt(file_path):
    """从txt文件中加载点云数据"""
    return np.loadtxt(file_path)


def process_pcd_files(root_path):
    """遍历所有子文件夹，寻找匹配的txt文件，并计算IoU"""
    for dirpath, _, filenames in os.walk(root_path):
        pcd1_files = [f for f in filenames if f.endswith('1output.txt')]
        pcd2_files = [f for f in filenames if f.endswith('4output.txt')]

        for pcd1_file in pcd1_files:
            pcd1_path = os.path.join(dirpath, pcd1_file)
            pcd2_file = pcd1_file.replace('1output.txt', '4output.txt')
            if pcd2_file in pcd2_files:
                pcd2_path = os.path.join(dirpath, pcd2_file)
                pos0 = load_pcd_from_txt(pcd1_path)
                pos1 = load_pcd_from_txt(pcd2_path)

                if(len(pos0)<6144):
                    pos0 = extend_point_cloud(pos0)
                if(len(pos1)<6144):
                    pos1 = extend_point_cloud(pos1)

                loU = get_current_covered_area_loU(pos0,pos1)
                coverage_ratio = get_current_covered_area(pos0) / get_current_covered_area(pos1)
                output_path = os.path.join(dirpath, pcd2_file.replace('4output.txt', '4_loU'+str(loU)+'.txt'))

                with open(output_path, 'w') as f:
                    f.write(f"\nIoU: {loU:.4f}\n")

                print(f"Processed {pcd1_path} and {pcd2_path}, IoU saved to {output_path}")



def load_point_cloud_from_txt(file_path):
    # 从 .txt 文件加载点云数据
    points = np.loadtxt(file_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# def align_point_clouds(source, target, max_iterations=50, threshold=0.02):
#     # 使用 ICP 算法对齐点云
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#         source, target, threshold,
#         np.identity(4),  # 初始变换矩阵
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
#     )
#     return reg_p2p.transformation

# def calculate_metrics(pcd1, pcd2):
#     # 计算欧氏距离的均值作为度量指标
#     distances = np.linalg.norm(np.asarray(pcd1.points) - np.asarray(pcd2.points), axis=1)
#     mean_distance = np.mean(distances)
#     return mean_distance

def normalize_point_cloud_trajectory(trajectory):
    """
    Normalize the first frame of a batch of point cloud trajectories, and scale the entire trajectory
    accordingly, keeping the center and scale consistent with the first frame.
    
    :param trajectory: Tensor of shape [batch_size, num_points, traj_steps, dim]
    :return: normalized_trajectory: Tensor of the same shape, normalized based on the first frame
    """
    # Step 1: Extract the first frame for normalization
    first_frame = trajectory[:, :, 0, :]  # [batch_size, num_points, dim]

    # Step 2: Compute the centroid of the first frame for each batch
    first_frame_centroid = first_frame.mean(dim=1, keepdim=True)  # [batch_size, 1, dim]
    
    # Step 3: Center the first frame
    first_frame_centered = first_frame - first_frame_centroid  # [batch_size, num_points, dim]
    
    # Step 4: Compute the max distance from the origin in the first frame for each batch
    distances = torch.sqrt((first_frame_centered ** 2).sum(dim=-1))  # [batch_size, num_points]
    max_distance, _ = distances.max(dim=1, keepdim=True)  # [batch_size, 1]
    
    # Step 5: Normalize the first frame by its max distance
    scaling_factor = max_distance.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1, 1]
    normalized_trajectory = (trajectory - first_frame_centroid.unsqueeze(2)) / scaling_factor  # Apply same scaling across all frames
    
    # # Step 6: Re-center the trajectory back to the first frame's centroid         # Remain in origin point
    # normalized_trajectory += first_frame_centroid.unsqueeze(2)

    return normalized_trajectory, scaling_factor, first_frame_centroid

def metrics(pos0,pos1):
    input = torch.stack((torch.from_numpy(pos0), torch.from_numpy(pos1)), dim=1).unsqueeze(0)
    input,_,_ = normalize_point_cloud_trajectory(input)
    pcd1 = input[0,:,0,:]
    pcd2 = input[0,:,1,:]
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(pcd2)
    
    distances, _ = neigh.kneighbors(pcd1)
    RUnf = np.mean(distances)

    if(len(pos0)<6144):
        pos0 = extend_point_cloud(pos0)
    if(len(pos1)<6144):
        pos1 = extend_point_cloud(pos1)

    return get_current_covered_area_loU(pos0,pos1), get_current_covered_area(pos0), get_current_covered_area(pos1) ,RUnf


# # 加载点云数据
# sim1_pcd1 = load_point_cloud_from_txt("path/to/sim1_pcd1.txt")
# sim2_pcd1 = load_point_cloud_from_txt("path/to/sim2_pcd1.txt")
# sim1_pcd2 = load_point_cloud_from_txt("path/to/sim1_pcd2.txt")
# sim2_pcd2 = load_point_cloud_from_txt("path/to/sim2_pcd2.txt")

# # 对齐 sim2_pcd1 到 sim1_pcd1
# transformation = align_point_clouds(sim2_pcd1, sim1_pcd1)

# # 使用变换矩阵对 sim2_pcd2 进行变换
# sim2_pcd2.transform(transformation)

# # 计算 sim1_pcd2 和 变换后的 sim2_pcd2 之间的度量
# metric_result = calculate_metrics(sim1_pcd2, sim2_pcd2)
# print(f"Metric result (mean Euclidean distance): {metric_result:.4f}")

import os
import json
import numpy as np
import open3d as o3d
import h5py

def load_point_cloud_from_txt(file_path):
    """Load point cloud data from txt file."""
    return np.loadtxt(file_path)

def load_point_cloud_from_h5(file_path, dataset_name):
    """Load point cloud data from h5 file."""
    with h5py.File(file_path, "r") as f:
        return np.array(f[dataset_name])

def align_point_clouds(initial_pcd_sim1, initial_pcd_sim2):
    # 计算两个点云的质心
    centroid_sim1 = np.mean(initial_pcd_sim1, axis=0)
    centroid_sim2 = np.mean(initial_pcd_sim2, axis=0)

    # 将 initial_pcd_sim1 平移，使其质心与 initial_pcd_sim2 对齐
    initial_pcd_sim1_centered = initial_pcd_sim1 - centroid_sim1
    initial_pcd_sim2_centered = initial_pcd_sim2 - centroid_sim2

    # 使用 ICP 进行精细对齐（你可以使用 Open3D 或其他库实现 ICP）
    # 例如，使用 Open3D 进行 ICP 对齐
    import open3d as o3d

    pcd_sim1 = o3d.geometry.PointCloud()
    pcd_sim2 = o3d.geometry.PointCloud()
    pcd_sim1.points = o3d.utility.Vector3dVector(initial_pcd_sim1_centered)
    pcd_sim2.points = o3d.utility.Vector3dVector(initial_pcd_sim2_centered)

    threshold = 0.02
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_sim1, pcd_sim2, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # 获取精细对齐后的变换矩阵
    transformation = np.copy(reg_p2p.transformation)

    # 将质心平移变换和精细对齐变换合并
    transformation[:3, 3] += centroid_sim2 - centroid_sim1

    return transformation
def calculate_face_normals_similarity(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    normals = np.cross(v1 - v0, v2 - v0)
    # 单位化法向量
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    normals[normals[:, 1] < 0] *= -1
    normal_mean = np.mean(normals, axis=0)
    normal_mean = normal_mean / np.linalg.norm(normal_mean)  # 单位化均值
    # 计算每个法向量与均值的相似度（点积）
    similarities = np.abs(np.dot(normals, normal_mean))
    return np.mean(similarities)
def calculate_metrics(aligned_pcd, target_pcd):
    """Calculate the mean Euclidean distance between two point clouds."""
    input = torch.from_numpy(aligned_pcd).unsqueeze(0).unsqueeze(2)
    input,s,t = normalize_point_cloud_trajectory(input)
    aligned_pcd1 = input[0,:,0,:].numpy()
    s = s[0,0,0,:].numpy()
    t = t[:,0,:].repeat(target_pcd.shape[0],1).numpy()
    
    target_pcd1 = (target_pcd-t)/s

    # print(np.mean(aligned_pcd1,axis=0),np.mean(target_pcd1,axis=0),t[0])
    # 创建 KDTree
    target_tree = KDTree(target_pcd1)
    target_tree2 = KDTree(aligned_pcd1)
    # print(aligned_pcd.shape)
    # print(target_pcd.shape)
    # 批量查找 aligned_pcd 中每个点在 target_pcd 中的最近邻
    distances, _ = target_tree.query(aligned_pcd1, k=1)
    distances2, _ = target_tree2.query(target_pcd1, k=1)

    # 计算平均距离
    mean_distance = np.mean(distances) + np.mean(distances2)
    return mean_distance

def dbscan(points):
    db = DBSCAN(eps=0.3, min_samples=100).fit((points))
    labels = db.labels_
    return points[labels != -1]
def remove_outliers_statistical(points, threshold=2.0):
    # Calculate the centroid of the point cloud
    centroid = np.mean(points, axis=0)
    
    # Compute the Euclidean distance of each point from the centroid
    distances = np.linalg.norm(points - centroid, axis=1)
    
    # Calculate mean and standard deviation of distances
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    # Filter out points that are beyond the threshold
    filtered_points = points[distances < mean_distance + threshold * std_distance]
    return filtered_points
def calculate_rectangle_ratio(point_cloud):
        # 将3D点云投影到XY平面
        # point_cloud = self.dbscan(point_cloud)
        points_2d = point_cloud[:, [0,2]]
        # points_2d = remove_outliers_statistical(points_2d)
        # 计算点云的凸包
        # points_2d = points_2d[np.random.choice(points_2d.shape[0], 20000, replace=False)]
        if points_2d.shape[0]<4000:
            points_2d = dbscan(points_2d)

        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
        hull_points = np.array(hull_points, dtype=np.float32)
        # 使用OpenCV计算最小外接矩形
        rect = cv2.minAreaRect(hull_points)
        box = cv2.boxPoints(rect)
        width = np.linalg.norm(box[0] - box[1])
        height = np.linalg.norm(box[1] - box[2])

        # print(box,(np.max(points_2d[:,0])-np.min(points_2d[:,0]))*(np.max(points_2d[:,1])-np.min(points_2d[:,1])),width * height)
        area_rect = width * height
        
        return float(area_rect)
        # return (np.max(points_2d[:,0])-np.min(points_2d[:,0]))*(np.max(points_2d[:,1])-np.min(points_2d[:,1]))
def process_mesh_folder(mesh_folder, initial_path, fin_path,CONST_P,face_t_path):
    # initial_path = os.path.join(mesh_folder, "initial_pcd.txt")
    # fin_path = os.path.join(mesh_folder, "fin_pcd.txt")

    # Load sim1 point clouds
    initial_pcd_sim1 = load_point_cloud_from_txt(initial_path) 
    # initial_pcd_sim1[:,1],initial_pcd_sim1[:,2] = initial_pcd_sim1[:,2],initial_pcd_sim1[:,1]
    fin_pcd_sim1 = load_point_cloud_from_txt(fin_path)
    # fin_pcd_sim1[:,1],fin_pcd_sim1[:,2] = fin_pcd_sim1[:,2],fin_pcd_sim1[:,1]
    # initial_pcd_sim1 = dbscan(initial_pcd_sim1)
    # fin_pcd_sim1 = dbscan(fin_pcd_sim1)
    planeness_i=1
    planeness_f=0

    if face_t_path is not None:
        faces=load_cloth_mesh(face_t_path)
        planeness_i=check_planeness(initial_pcd_sim1.copy(),faces)
        planeness_f=check_planeness(fin_pcd_sim1.copy(),faces)

    if CONST_P:
        initial_pcd_sim1*=np.array([-1,1,-1])
        fin_pcd_sim1*=np.array([-1,1,-1])

    # Load sim2 point clouds from h5 files

    initial_area_rect = calculate_rectangle_ratio(initial_pcd_sim1.copy())
    fin_area_rect = calculate_rectangle_ratio(fin_pcd_sim1.copy())
    pos0 = initial_pcd_sim1 

    num = 16144
    if(pos0.shape[0]<num):
        pos0 = extend_point_cloud(pos0,num)
    else :
        # indices = np.random.choice(pos0.shape[0], 50000, replace=False)
        # pos0 = pos0[indices]
        swapped_pcd = pos0.copy()
        swapped_pcd[:, [1, 2]] = swapped_pcd[:, [2, 1]]
        pos0 = swapped_pcd

    # print(get_current_covered_area(initial_pcd_sim1*scale_factors),scale_factors)
    c0i = get_current_covered_area(pos0)
       
    pos0 = fin_pcd_sim1

    num = 16144
    if(pos0.shape[0]<num):
        pos0 = extend_point_cloud(pos0,num)
    else :
        # indices = np.random.choice(pos0.shape[0], num, replace=False)
        # pos0 = pos0[indices]
        swapped_pcd = pos0.copy()
        swapped_pcd[:, [1, 2]] = swapped_pcd[:, [2, 1]]
        pos0 = swapped_pcd

    # print(get_current_covered_area(initial_pcd_sim1*scale_factors),scale_factors)
    c0=  get_current_covered_area(pos0)
     
    rate_initial_fin_rect = fin_area_rect / initial_area_rect
    ract_ratio = c0/fin_area_rect
    # Save metrics to JSON file
    # mean_distance = calculate_metrics(pos0, pos1)
    data = {
        "initial_real": c0i,

        "coverage_real" : c0,
        "coverage_rate_real" : c0/c0i,
        "rate_initial_fin_rect" : rate_initial_fin_rect,
        "ract_ratio" : ract_ratio
    }
    json_path = os.path.join(mesh_folder, "metrics.json")
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    return c0/c0i,rate_initial_fin_rect,ract_ratio

def process_eval_folder(root_folder):
    # First level: Iterate through each folder in eval_ep/*
    
    for eval_folder in os.listdir(root_folder):
        if (not eval_folder.endswith("sleeve")) and  (not eval_folder.endswith("Pants")):
            continue
        coverage_rate_r = 0
        rate_initial_fin_rect=0
        ract_ratio=0
        num=0
        error = []
        eval_path = os.path.join(root_folder, eval_folder)
        print(eval_folder)
        for case_folder in os.listdir(eval_path):
            if case_folder.startswith("0009"):
                CONST_P = False
            else:
                # continue
                CONST_P = False
            case_path = os.path.join(eval_path, case_folder)
                
            if not os.path.isdir(case_path):
                continue

            # Second level: Iterate through each mesh folder in eval_ep/*/mesh*
            # loU = 0
            # mean = 10
            
            for mesh_folder in os.listdir(case_path):
                if not mesh_folder.endswith("mesh0"):
                    continue

                mesh_folder_path = os.path.join(case_path, mesh_folder)
                if not os.path.isdir(mesh_folder_path):
                    continue

                # Paths for point clouds
                tmp = os.path.join("/home/transfer/chenhn_data/cloth3d_eval/eval/eval/", eval_folder,case_folder,"initial.obj")
                # tmp = case_path


                # Check if initial and fin txt files exist in mesh folder
                initial_path = os.path.join(mesh_folder_path, "initial_pcd.txt")
                fin_path = os.path.join(mesh_folder_path, "fin_pcd.txt")
                if os.path.exists(initial_path) and os.path.exists(fin_path) :
                    # print(f"Processing {mesh_folder_path}...")
                    r,a,b = process_mesh_folder(mesh_folder_path, initial_path, fin_path,CONST_P,tmp)
                    # loU = max(loU,l)
                    # mean = min(mean,m)
                    coverage_rate_r += r
                    rate_initial_fin_rect += a
                    ract_ratio += b
                    num+=1
                    print(case_folder,"\t",r,"\t",a,"\t",b)
                    # coverage_rate_t = min(coverage_rate_t,t)
                    # break
                else:
                    print(case_folder,"\t1\t1\t0")
            # json_path = os.path.join(case_path, "metrics.json")
            # data = {
            #     # "loU_with_target" : loU,
            #     # "RUnf_with_target" : mean,
            #     "coverage_rate_real" : coverage_rate_r,
            #     # "coverage_rate_target" : coverage_rate_t
            # }
            # with open(json_path, "w") as json_file:
            #     json.dump(data, json_file, indent=4)
        if num == 0 :
            continue
        json_path = os.path.join(eval_path, "metrics.json")
        data = {
            # "loU_with_target" : loU,
            # "RUnf_with_target" : mean,
            "coverage_rate_real" : coverage_rate_r/num,
            "rate_initial_fin_rect" : rate_initial_fin_rect/num,
            "ract_ratio" : ract_ratio/num
        }
        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        for e in error:
            print(e)
        


if __name__ == "__main__":
    # eval_folder = "/home/transfer/chenhn_data/eval_ep"
    eval_folder = "/home/transfer/.local/share/ov/pkg/isaac-sim-4.1.0/extension_examples/Cloth-Flod-in-isaac-sim/isaac_sim/output"
    # eval_folder = "/home/transfer/chenhn_data/cloth3d_eval/uniG/eval/"
    # eval_folder = "/home/transfer/chenhn/GPT-fabric-folding/eval result/AllCornersInward/Tshirt/2024-11-11"
    # eval_folder = "/home/transfer/chenhn/GPT-fabric-folding/eval result/DoubleStraight/Trousers/2024-11-11"
    # eval_folder  = "/home/transfer/chenhn/language_deformable/eval/single"
    # eval_folder = "/home/transfer/chenhn/GPT-fabric-folding/eval result/DoubleTriangle/Dress/2024-11-12"
    # eval_folder = "/home/transfer/chenhn_data/train/eval_cloth_new/"
    # eval_folder = "/home/transfer/chenhn_data/eval_ep/eval/output/output"
    process_eval_folder(eval_folder)
