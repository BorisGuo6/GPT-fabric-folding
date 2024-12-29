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
def dbscan_pcd(points):
    # 使用DBSCAN进行聚类
    db = DBSCAN(eps=0.15, min_samples=10).fit(points)#[:,[0,2]]
    
    # 获取标签 (-1表示噪点)
    labels = db.labels_
    
    # 过滤掉标签为 -1 的点
    return points[labels != -1]



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
    print(min_x," ",max_x," ",min_y," ",max_y," ")
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
    db = DBSCAN(eps=0.3, min_samples=10).fit(torch.from_numpy(points))
    labels = db.labels_
    return points[labels != -1]

def process_mesh_folder(mesh_folder, initial_path, fin_path, pcd_traj_path, pcd_traj2_path,CONST_P):
    # initial_path = os.path.join(mesh_folder, "initial_pcd.txt")
    # fin_path = os.path.join(mesh_folder, "fin_pcd.txt")

    # Load sim1 point clouds
    initial_pcd_sim1 = load_point_cloud_from_txt(initial_path) 
    # initial_pcd_sim1[:,1],initial_pcd_sim1[:,2] = initial_pcd_sim1[:,2],initial_pcd_sim1[:,1]
    fin_pcd_sim1 = load_point_cloud_from_txt(fin_path)
    # fin_pcd_sim1[:,1],fin_pcd_sim1[:,2] = fin_pcd_sim1[:,2],fin_pcd_sim1[:,1]
    # initial_pcd_sim1 = dbscan(initial_pcd_sim1)
    # fin_pcd_sim1 = dbscan(fin_pcd_sim1)

    if CONST_P:
        initial_pcd_sim1*=np.array([-1,1,-1])
        fin_pcd_sim1*=np.array([-1,1,-1])

    # Load sim2 point clouds from h5 files
    if not (os.path.exists(pcd_traj_path) and os.path.exists(pcd_traj2_path)):
        print(f"Skipping {mesh_folder} due to missing sim2 point clouds.")
        return

    with h5py.File(pcd_traj_path, 'r') as h5_file:
        initial_pcd_sim2 = np.array(h5_file['pcd_traj'])[0]
    # initial_pcd_sim2 = load_point_cloud_from_h5(pcd_traj_path, "pcd_traj", 0)  # Assuming first frame is initial
    with h5py.File(pcd_traj2_path, 'r') as h5_file:
        fin_pcd_sim2 = np.array(h5_file['pcd_traj'])[-1]
    # fin_pcd_sim2 = load_point_cloud_from_h5(pcd_traj2_path, "pcd_traj2", -1)   # Assuming last frame is final

    # Align sim1 initial point cloud to sim2 initial point cloud
    transformation = align_point_clouds(initial_pcd_sim1, initial_pcd_sim2)

    # initial_pcd_sim1_aligned = (transformation[:3, :3] @ initial_pcd_sim1.T).T + transformation[:3, 3]

    # 计算两个点云的质心
    centroid_sim1 = np.mean(initial_pcd_sim1, axis=0)
    centroid_sim2 = np.mean(initial_pcd_sim2, axis=0)

    # Transform fin_pcd_sim1 using the obtained transformation
    fin_pcd_sim1_aligned = fin_pcd_sim1- centroid_sim1
    fin_pcd_sim2 = fin_pcd_sim2 - centroid_sim2
    # print( np.mean(fin_pcd_sim1_aligned, axis=0),np.mean(fin_pcd_sim2, axis=0))
    
    bbox_sim1 = np.max(initial_pcd_sim1, axis=0) - np.min(initial_pcd_sim1, axis=0)
    bbox_sim2 = np.max(initial_pcd_sim2, axis=0) - np.min(initial_pcd_sim2, axis=0)
    bbox_sim1[1] = (bbox_sim1[2]+bbox_sim1[0])/2
    bbox_sim2[1] = (bbox_sim2[2]+bbox_sim2[0])/2
    # 分别计算每个维度的缩放因子
    scale_factors = bbox_sim2 / bbox_sim1

    # 对每个维度应用缩放因子
    fin_pcd_sim1_aligned *= scale_factors
    # print( np.max(fin_pcd_sim1_aligned, axis=0),np.max(fin_pcd_sim2, axis=0))
    # print( np.min(fin_pcd_sim1_aligned, axis=0),np.min(fin_pcd_sim2, axis=0))
    # Calculate metrics
    mean_distance = calculate_metrics(fin_pcd_sim2, fin_pcd_sim1_aligned)

    pos0 = (initial_pcd_sim1 - centroid_sim1)* scale_factors
    pos1 = initial_pcd_sim2 - centroid_sim2
    num = 16144
    if(pos0.shape[0]<num):
        pos0 = extend_point_cloud(pos0,num)
    else :
        # indices = np.random.choice(pos0.shape[0], 50000, replace=False)
        # pos0 = pos0[indices]
        swapped_pcd = pos0.copy()
        swapped_pcd[:, [1, 2]] = swapped_pcd[:, [2, 1]]
        pos0 = swapped_pcd
    if(pos1.shape[0]<num):
        pos1 = extend_point_cloud(pos1,num)
    else :
        # indices = np.random.choice(pos1.shape[0], 50000, replace=False)
        # pos1 = pos1[indices]
        swapped_pcd = pos1.copy()
        swapped_pcd[:, [1, 2]] = swapped_pcd[:, [2, 1]]
        pos1 = swapped_pcd
    # print(get_current_covered_area(initial_pcd_sim1*scale_factors),scale_factors)
    loUi,c0i,c1i = get_current_covered_area_loU(pos0,pos1), get_current_covered_area(pos0), get_current_covered_area(pos1)

    pos0 = fin_pcd_sim1_aligned
    pos1 = fin_pcd_sim2
    num = 50000
    if(pos0.shape[0]<num):
        pos0 = extend_point_cloud(pos0,num)
    else :
        # indices = np.random.choice(pos0.shape[0], num, replace=False)
        # pos0 = pos0[indices]
        swapped_pcd = pos0.copy()
        swapped_pcd[:, [1, 2]] = swapped_pcd[:, [2, 1]]
        pos0 = swapped_pcd
    if(pos1.shape[0]<num):
        pos1 = extend_point_cloud(pos1,num)
    else :
        # indices = np.random.choice(pos1.shape[0], num, replace=False)
        # pos1 = pos1[indices]
        swapped_pcd = pos1.copy()
        swapped_pcd[:, [1, 2]] = swapped_pcd[:, [2, 1]]
        pos1 = swapped_pcd
    # print(get_current_covered_area(initial_pcd_sim1*scale_factors),scale_factors)
    loU,c0,c1 = get_current_covered_area_loU(pos0,pos1), get_current_covered_area(pos0), get_current_covered_area(pos1)
    # Save metrics to JSON file
    # mean_distance = calculate_metrics(pos0, pos1)
    data = {
        "initial_real": c0i,
        "initial_target": c1i,
        "initial_loU": loUi,
        "coverage_real" : c0,
        "coverage_target" : c1,
        "loU_with_target" : loU,
        "RUnf_with_target" : mean_distance
    }
    json_path = os.path.join(mesh_folder, "metrics.json")
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    return loU,mean_distance,c0/c0i,c1/c1i

def process_eval_folder(eval_folder):
    # First level: Iterate through each folder in eval_ep/*
    for case_folder in os.listdir(eval_folder):
        if case_folder.startswith("P"):
            CONST_P = False
        else:
            # continue
            CONST_P = False
        case_path = os.path.join(eval_folder, case_folder)
            
        if not os.path.isdir(case_path):
            continue

        # Second level: Iterate through each mesh folder in eval_ep/*/mesh*
        loU = 0
        mean = 10
        coverage_rate_r = 1
        coverage_rate_t = 1
        for mesh_folder in os.listdir(case_path):
            if not mesh_folder.startswith("mesh"):
                continue

            mesh_folder_path = os.path.join(case_path, mesh_folder)
            if not os.path.isdir(mesh_folder_path):
                continue

            # Paths for point clouds
            tmp = os.path.join("/home/transfer/chenhn_data/eval_ep/eval", case_folder)
            # tmp = case_path
            pcd_traj_path = os.path.join(tmp, "pcd_traj.h5")
            pcd_traj2_path = os.path.join(tmp, "pcd_traj2.h5")

            # Check if initial and fin txt files exist in mesh folder
            initial_path = os.path.join(mesh_folder_path, "initial_pcd.txt")
            fin_path = os.path.join(mesh_folder_path, "fin_pcd.txt")
            if os.path.exists(initial_path) and os.path.exists(fin_path) and os.path.exists(pcd_traj_path) and os.path.exists(pcd_traj2_path):
                print(f"Processing {mesh_folder_path}...")
                l,m,r,t = process_mesh_folder(mesh_folder_path, initial_path, fin_path, pcd_traj_path, pcd_traj2_path,CONST_P)
                loU = max(loU,l)
                mean = min(mean,m)
                coverage_rate_r = min(coverage_rate_r,r)
                coverage_rate_t = min(coverage_rate_t,t)
                # break
            # else:
            #     print(f"Skipping {mesh_folder_path} due to missing initial or fin txt files.")
        json_path = os.path.join(case_path, "metrics.json")
        data = {
            "loU_with_target" : loU,
            "RUnf_with_target" : mean,
            "coverage_rate_real" : coverage_rate_r,
            "coverage_rate_target" : coverage_rate_t
        }
        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)



if __name__ == "__main__":
    # eval_folder = "/home/transfer/chenhn_data/eval_ep"
    eval_folder = "/home/transfer/.local/share/ov/pkg/isaac-sim-4.1.0/extension_examples/Cloth-Flod-in-isaac-sim/isaac_sim/output"
    # eval_folder = "/home/transfer/chenhn_data/train/eval_cloth_new/"
    # eval_folder = "/home/transfer/chenhn_data/eval_ep/eval/output/output"
    process_eval_folder(eval_folder)
