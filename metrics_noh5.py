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
def calculate_rectangle_ratio(point_cloud):
        # 将3D点云投影到XY平面
        points_2d = point_cloud[:, [0,2]]

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

        area_rect = width * height
        
        return float(area_rect)

# obj_mapping={'0000': 'TCLC/TCLC_Jacket135_action0', '0001': 'TCLC/TCLC_Top012_0_action0', '0002': 'TCLC/TCLC_Jacket060_action0', '0003': 'TCLC/TCLC_Jacket150_action0', '0004': 'TCLC/TCLC_Jacket165_action0', '0005': 'TCLO/TCLO_model2_035_action0', '0006': 'TCLO/TCLO_025_action0', '0007': 'TNLC/TNLC_Jacket001_0_action0', '0008': 'TNLC/TNLC_Dress425_action0', '0009': 'TNLC/TNLC_Top550_action0', '0010': 'TNLC/TNLC_Jacket075_action0', '0011': 'TNLC/TNLC_Top220_action0', '0012': 'TNLC/TNLC_Top470_action0', '0013': 'TNLC/TNLC_model2_065_action0', '0014': 'TNLC/TNLC_Top400_action0', '0015': 'TNLC/TNLC_Top315_action0', '0016': 'TNLC/TNLC_010_action0', '0017': 'TNLC/TNLC_Top200_action0', '0018': 'DLLS/DLLS_Dress355_action0', '0019': 'DLLS/DLLS_Dress008_0_action0', '0020': 'DSLS/DSLS_Dress405_action0', '0021': 'DSLS/DSLS_Dress360_action0', '0022': 'DSLS/DSLS_Dress370_action0', '0023': 'DSLS/DSLS_Dress415_action0', '0024': 'DSLS/DSLS_Dress280_action0', '0025': 'TNLO/TNLO_070_action0', '0026': 'TNLO/TNLO_035_action0', '0027': 'THLO/THLO_Jacket015_action0', '0028': 'THLC/THLC_Jacket110_action0', '0029': 'THLC/THLC_Normal_Model_045_action0'}
# obj_mapping={'0000': '0038', '0001': '0019', '0002': '0035', '0003': '0027', '0004': '0039', '0005': '0030', '0006': '0033', '0007': '0016', '0008': '0031', '0009': '0005', '0010': '0042', '0011': '0004', '0012': '0037', '0013': '0020', '0014': '0001', '0015': '0010', '0016': '0003', '0017': '0021', '0018': '0029', '0019': '0013', '0020': '0012', '0021': '0017', '0022': '0002', '0023': '0014', '0024': '0028', '0025': '0023', '0026': '0032', '0027': '0018', '0028': '0025', '0029': '0000', '0030': '0041', '0031': '0040', '0032': '0024', '0033': '0011', '0034': '0007', '0035': '0036', '0036': '0008', '0037': '0043', '0038': '0015', '0039': '0044', '0040': '0026', '0041': '0006', '0042': '0034', '0043': '0009', '0044': '0022'}
obj_mapping={'0000': 'PL/PL_M1_100_action0', '0001': 'PL/PL_030_action0', '0002': 'PL/PL_060_action0', '0003': 'PL/PL_longpants005_action0', '0004': 'PL/PL_M2_015_action0', '0005': 'PL/PL_LongPants025_action0', '0006': 'PL/PL_075_action0', '0007': 'PL/PL_short010_action0', '0008': 'PL/PL_M1_065_action0', '0009': 'PL/PL_070_action0', '0010': 'PL/PL_095_action0', '0011': 'PL/PL_M1_080_action0', '0012': 'PL/PL_M1_075_action0', '0013': 'PL/PL_M2_010_action0', '0014': 'PL/PL_M1_045_action0', '0015': 'PL/PL_085_action0', '0016': 'PL/PL_pants030_action0', '0017': 'PL/PL_pants5_action0', '0018': 'PL/PL_080_action0', '0019': 'PL/PL_LongPants015_action0', '0020': 'PL/PL_090_action0', '0021': 'PL/PL_longpants020_action0', '0022': 'PL/PL_020_action0', '0023': 'PS/PS_015_action0', '0024': 'PS/PS_010_action0', '0025': 'PS/PS_M1_070_action0', '0026': 'PS/PS_M1_015_action0', '0027': 'PS/PS_025_action0', '0028': 'PS/PS_055_action0', '0029': 'PS/PS_045_action0', '0030': 'PS/PS_050_action0', '0031': 'PS/PS_M1_060_action0', '0032': 'PS/PS_005_action0', '0033': 'PS/PS_M1_095_action0', '0034': 'PS/PS_M1_040_action0'}
# obj_mapping={'0000': 'TNNC/TNNC_Normal_Model_025_action0', '0001': 'TNNC/TNNC_Top006_0_action0', '0002': 'TNNC/TNNC_Top265_action0', '0003': 'TNNC/TNNC_Top300_action0', '0004': 'TNNC/TNNC_model2_010_action0', '0005': 'TNNC/TNNC_Top510_action0', '0006': 'TNNC/TNNC_model2_005_action0', '0007': 'TNNC/TNNC_Normal_Model_050_action0', '0008': 'TNNC/TNNC_045_action0', '0009': 'TNNC/TNNC_Normal_Model_010_action0', '0010': 'TNNC/TNNC_Top505_action0', '0011': 'TNNC/TNNC_Top075_action0', '0012': 'TNNC/TNNC_Top285_action0', '0013': 'TNNC/TNNC_model2_040_action0', '0014': 'TNNC/TNNC_Top150_action0', '0015': 'TNNC/TNNC_Top270_action0', '0016': 'TNNC/TNNC_Top050_action0', '0017': 'TNNC/TNNC_Normal_Model_015_action0', '0018': 'TNNC/TNNC_Normal_Model_005_action0', '0019': 'TNNC/TNNC_Top595_action0', '0020': 'TNNC/TNNC_090_action0', '0021': 'TNNC/TNNC_top190_action0', '0022': 'TNNC/TNNC_model2_050_action0', '0023': 'DLT/DLT_Dress036_0_action0', '0024': 'DLT/DLT_Dress006_0_action0', '0025': 'DLT/DLT_Dress070_action0', '0026': 'DLT/DLG_Dress032_0_action0', '0027': 'DLT/DLT_Dress390_action0', '0028': 'DLT/DLT_Dress310_action0', '0029': 'DLT/DLT_Dress140_action0', '0030': 'DSNS/DSNS_Dress051_0_action0', '0031': 'DSNS/DSNS_Dress150_action0', '0032': 'DSNS/DSNS_Dress170_action0', '0033': 'DSNS/DSNS_Dress130_action0', '0034': 'DSNS/DSNS_Dress285_action0', '0035': 'DSNS/DSNS_Dress180_action0', '0036': 'DSNS/DSNS_dress145_action0', '0037': 'DSNS/DSNS_Dress029_0_action0', '0038': 'DSNS/DSNS_Dress115_action0', '0039': 'DSNS/DSNS_Dress200_action0', '0040': 'DSNS/DSNS_Dress120_action0', '0041': 'DSNS/DSNS_Dress215_action0', '0042': 'DSNS/DSNS_Dress255_action0', '0043': 'DLNS/DLNS_dress040_action0', '0044': 'DLNS/DLNS_Dress005_action0', '0045': 'DLNS/DLNS_Dress019_0_action0', '0046': 'DLNS/DLNS_Dress020_action0', '0047': 'DLNS/DLNS_Dress060_action0', '0048': 'DLNS/DLNS_Dress046_0_action0', '0049': 'DLNS/DLNS_Dress003_0_action0', '0050': 'DLNS/DLNS_Dress059_0_action0', '0051': 'DLNS/DLNS_Dress044_0_action0', '0052': 'DLNS/DLNS_Dress045_0_action0', '0053': 'DLNS/DLNS_Dress041_0_action0', '0054': 'DLNS/DLNS_Dress038_0_action0', '0055': 'DLNS/DLNS_Dress205_action0', '0056': 'DLNS/DLNS_Dress030_action0', '0057': 'DLNS/DLNS_Dress345_action0', '0058': 'DLNS/DLNS_Dress018_0_action0', '0059': 'DLNS/DLNS_Dress043_0_action0', '0060': 'DLNS/DLNS_Dress240_action0', '0061': 'DLNS/DLNS_Dress375_action0', '0062': 'DLNS/DLNS_Dress190_action0', '0063': 'DLG/DLG_Dress055_0_action0', '0064': 'DLG/DLG_Dress400_action0', '0065': 'DLG/DLG_Dress100_action0', '0066': 'DLG/DLG_Dress054_0_action0', '0067': 'DLG/DLG_Dress035_0_action0', '0068': 'DLG/DLG_Dress165_action0', '0069': 'SS/SS_Skirt045_action0', '0070': 'SS/SS_Skirt215_action0', '0071': 'SS/SS_Skirt055_action0', '0072': 'SS/SS_Skirt150_action0', '0073': 'SS/SS_Skirt225_action0', '0074': 'SS/SL_Skirt300_action0', '0075': 'SS/SS_Skirt052_0_action0', '0076': 'SS/SS_Skirt060_action0', '0077': 'SS/SL_Skirt155_action0', '0078': 'SS/SS_Skirt170_action0', '0079': 'SS/SS_Skirt090_action0', '0080': 'SS/SS_skirt110_action0', '0081': 'SS/SS_Skirt135_action0', '0082': 'SS/SL_Skirt061_0_action0', '0083': 'TCNC/TCNC_Top145_action0', '0084': 'TCNC/TCNC_Top031_0_action0', '0085': 'TCNC/TCNC_Vest006_0_action0', '0086': 'TCNC/TCNC_Jacket160_action0', '0087': 'TCNC/TCNC_Vest023_0_action0', '0088': 'TCNC/TCNC_Shirt020_action0', '0089': 'TCNC/TCNC_Jacket145_action0', '0090': 'TCNC/TCNC_Top615_action0', '0091': 'TCNC/TCNC_Top230_action0', '0092': 'TCNC/TCNC_Jacket115_action0', '0093': 'TCNC/TCNC_Top210_action0', '0094': 'TCNC/TCNC_Top350_action0', '0095': 'TCNC/TCNC_Jacket155_action0', '0096': 'TCNC/TCNC_Shirt035_action0', '0097': 'TCNC/TCNC_Top215_action0', '0098': 'TCNC/TCNC_Top032_0_action0', '0099': 'TCNC/TCNC_Top060_action0', '0100': 'TCNC/TCNC_Top655_action0', '0101': 'TCNC/TCNC_Top033_0_action0', '0102': 'SL/SL_Skirt048_0_action0', '0103': 'SL/SL_Skirt020_action0', '0104': 'SL/SL_Skirt080_action0', '0105': 'SL/SL_Skirt380_action0', '0106': 'SL/SL_Skirt038_0_action0', '0107': 'SL/SL_Skirt049_0_action0', '0108': 'SL/SL_skirt180_action0', '0109': 'SL/SL_Skirt005_action0', '0110': 'SL/SL_Skirt285_action0', '0111': 'SL/SL_Skirt035_action0', '0112': 'SL/SL_Skirt050_0_action0', '0113': 'SL/SL_Skirt140_action0', '0114': 'SL/SL_Skirt320_action0', '0115': 'SL/SL_Skirt100_action0', '0116': 'SL/SL_skirt240_action0', '0117': 'SL/SS_Skirt105_action0', '0118': 'SL/SL_Skirt041_0_action0', '0119': 'SL/SL_Skirt205_action0', '0120': 'SL/SL_Skirt355_action0', '0121': 'SL/SL_Skirt350_action0', '0122': 'SL/SL_Skirt385_action0', '0123': 'SL/SL_Skirt160_action0', '0124': 'SL/SL_Skirt200_action0', '0125': 'SL/SL_Skirt330_action0', '0126': 'SL/SL_Skirt340_action0', '0127': 'SL/SL_Skirt315_action0', '0128': 'SL/SL_Skirt195_action0', '0129': 'SL/SL_Skirt245_action0', '0130': 'SL/SL_Skirt030_action0', '0131': 'THNC/THNC_Normal_Model_030_action0', '0132': 'THNC/THNC_Top202_0_action0', '0133': 'THNC/THNC_Vest015_action0', '0134': 'TCNO/TCNO_Vest035_action0', '0135': 'TCNO/TCNO_050_action0', '0136': 'TCNO/TCNO_Vest020_action0', '0137': 'TCNO/TCNO_030_action0', '0138': 'TCNO/TCNO_Shirt010_action0', '0139': 'TCNO/TCNO_080_action0', '0140': 'TCNO/TCNO_Jacket120_action0', '0141': 'TCNO/TCNO_020_action0'}
# obj_mapping={'0000': 'TCSC/TCSC_Top410_action0', '0001': 'TCSC/TCSC_075_action0', '0002': 'TCSC/TCSC_top115_action0', '0003': 'TCSC/TCSC_Top095_action0', '0004': 'TCSC/TCSC_model2_080_action0', '0005': 'TCSC/TCSC_Top585_action0', '0006': 'TCSC/TCSC_Top034_0_action0', '0007': 'TCSC/TCSC_model2_045_action0', '0008': 'TCSC/TCSC_Top610_action0', '0009': 'TCSC/TCSC_Top010_action0', '0010': 'DLSS/DLSS_Dress037_0_action0', '0011': 'DLSS/DLSS_Dress420_action0', '0012': 'DLSS/DLSS_Dress110_action0', '0013': 'DSSS/DSSS_Dress385_action0', '0014': 'DSSS/DSSS_dress10_action0', '0015': 'TNSC/TNSC_Top195_action0', '0016': 'TNSC/TNSC_Polo010_action0', '0017': 'TNSC/TNSC_Top280_action0', '0018': 'TNSC/TNSC_Top135_action0', '0019': 'TNSC/TNSC_Top560_action0', '0020': 'TNSC/TNSC_top515_action0', '0021': 'TNSC/TNSC_Normal_Model_020_action0', '0022': 'TNSC/TNSC_Top340_action0', '0023': 'TNSC/TNSC_model2_085_action0', '0024': 'TNSC/TNSC_top065_action0', '0025': 'TNSC/TNSC_top175_action0', '0026': 'TNSC/TNSC_Top495_action0', '0027': 'TNSC/TNSC_Normal_Model_040_action0', '0028': 'TNSC/TNSC_Top535_action0'}

def get_original_cloth_name(obj_name):
    """根据新 .obj 文件名查询原始衣服名称"""
    return obj_mapping.get(obj_name, "TCLC/TCLC_Jacket135_action0")

def determine_cloth_type(folder_name):
    # Define a mapping for known cloth types based on folder prefixes
    cloth_type_map = {
        "TNSC": "Short-sleeve",
        "TCSC": "Short-sleeve",
        "TNNC": "No-sleeve",
        "TCLC": "Long-sleeve",
        "DLT": "No-sleeve",
        "DLSS": "Short-sleeve",
        "DSNS": "No-sleeve",
        "DSSS": "Short-sleeve",
        "DLNS": "No-sleeve",
        "PL": "Pants",
        "TCLO": "Long-sleeve",
        "TNLC": "Long-sleeve",
        "DLLS": "Long-sleeve",
        "DSLS": "Long-sleeve",
        "PS": "Pants",
        "DLG": "No-sleeve",
        "TNLO": "Long-sleeve",
        "SS": "No-sleeve",
        "TCNC": "No-sleeve",
        "THLO": "Long-sleeve",
        "THLC": "Long-sleeve",
        "SL": "No-sleeve",
        "THNC": "No-sleeve",
        "TCNO": "No-sleeve"
    }

    # Return the corresponding cloth type or "Unknown" if not in the map
    return cloth_type_map.get(folder_name, folder_name)


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
    surface_mesh.fix_normals() 
    print("nomal:",calculate_face_normals_similarity(vertices,faces[mask]))   
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

def process_mesh_folder(mesh_folder, initial_path, fin_path, pcd_traj_path, pcd_traj2_path,CONST_P,face_path=None,face_t_path=None):
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
    planeness_ti=0
    planeness_tf=0
    if face_t_path is not None:
        faces=load_cloth_mesh(face_t_path)
        planeness_i=check_planeness(initial_pcd_sim1.copy(),faces)
        planeness_f=check_planeness(fin_pcd_sim1.copy(),faces)




    if CONST_P:
        initial_pcd_sim1*=np.array([-1,1,-1])
        fin_pcd_sim1*=np.array([-1,1,-1])
    initial_area_rect = calculate_rectangle_ratio(initial_pcd_sim1.copy())
    fin_area_rect = calculate_rectangle_ratio(fin_pcd_sim1.copy())
    # Load sim2 point clouds from h5 files
    if not (os.path.exists(pcd_traj_path) and os.path.exists(pcd_traj2_path)):
        print(f"Skipping {mesh_folder} due to missing sim2 point clouds.")
        return


    initial_pcd_sim2 = load_point_cloud_from_txt(pcd_traj_path)

    fin_pcd_sim2 = load_point_cloud_from_txt(pcd_traj2_path)

    if face_t_path is not None:
        faces2=load_cloth_mesh(face_t_path)
        planeness_ti=check_planeness(initial_pcd_sim2.copy(),faces2)
        planeness_tf=check_planeness(fin_pcd_sim2.copy(),faces2)
    # Align sim1 initial point cloud to sim2 initial point cloud
    # transformation = align_point_clouds(initial_pcd_sim1, initial_pcd_sim2)

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
    num = 16144
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
    rate_initial_fin_rect = fin_area_rect / initial_area_rect
    ract_ratio = c0/calculate_rectangle_ratio(fin_pcd_sim1_aligned.copy())
    
    data = {
        "initial_real": c0i,
        "initial_target": c1i,
        "initial_loU": loUi,
        "coverage_real" : c0,
        "coverage_target" : c1,
        "loU_with_target" : loU,
        "RUnf_with_target" : mean_distance,
        "rate_initial_fin_rect" : rate_initial_fin_rect,
        "ract_ratio" : ract_ratio,
        "planeness_i":planeness_i,
        "planeness_f":planeness_f,
        "planeness_ti":planeness_ti,
        "planeness_tf":planeness_tf,
    }
    json_path = os.path.join(mesh_folder, "metrics.json")
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    return loU,mean_distance,c0/c0i,rate_initial_fin_rect,ract_ratio,c0,c0i,c1/c1i,planeness_i,planeness_f,planeness_ti,planeness_tf
import re

def find_latest_target(base_folder, case):
    # Extract the base name without the "_action0" suffix
    base_name = case.rsplit("_action", 1)[0]
    pattern = re.compile(f"{re.escape(base_name)}_action\\d+")

    # List all matching folders in the base directory
    matching_folders = [f for f in os.listdir(base_folder) if pattern.fullmatch(f)]
    
    if not matching_folders:
        return None

    # Sort the folders by the numeric suffix in ascending order
    matching_folders.sort(key=lambda x: int(x.rsplit("_action", 1)[1]))

    # Return the folder with the highest numeric suffix
    return matching_folders[-1]
def process_eval_folder(root_folder):
    # First level: Iterate through each folder in eval_ep/*
    file_name = os.path.join(root_folder, f"output.txt")        
    with open(file_name, 'w') as file:
        pass
    for eval_folder in os.listdir(root_folder):
        if eval_folder.endswith("TNSC"):
                CONST_P = False
        else:
                # continue
                CONST_P = False
        coverage_rate_r = 0
        rate_initial_fin_rect=0
        ract_ratio=0
        num=0
        error = []
        cloth_type = determine_cloth_type(eval_folder)
        # if not cloth_type.startswith("No-"):
        #     continue
        eval_path = os.path.join(root_folder, eval_folder)
        if not os.path.isdir(eval_path):
            continue
        print(eval_folder)
        tmp = os.path.join("/home/transfer/chenhn_data/cloth_eval_all/target", eval_folder)
        if not os.path.isdir(tmp):
            continue
        for case_folder in os.listdir(eval_path):
            
            case_path = os.path.join(eval_path, case_folder)
                
            if not os.path.isdir(case_path):
                continue

            # Second level: Iterate through each mesh folder in eval_ep/*/mesh*
            loU = -1
            mean = 10
            success = None
            coverage_rate_r = 1
            coverage_rate_t = 1
            # original_path = get_original_cloth_name(case_folder.zfill(4))
            # eval_folder=original_path.split('/')[-2]
            # case_folder=original_path.split('/')[-1]
            tmp = os.path.join("/home/transfer/chenhn_data/cloth_eval_all/target", eval_folder)
            target = find_latest_target(tmp, case_folder)
            
            print(target)
            tmp1 = os.path.join(tmp,case_folder)
            tmp2 = os.path.join(tmp,target)
            # tmp = case_path
            pcd_traj_path = os.path.join(tmp1, "initial_pcd.txt")
            face_t_path = os.path.join(tmp1, "initial.obj")
            pcd_traj2_path = os.path.join(tmp2, "fin_pcd.txt")

            q=0
            for mesh_folder in os.listdir(case_path):
                if not mesh_folder.endswith("mesh0"):
                    continue

                mesh_folder_path = os.path.join(case_path, mesh_folder)
                if not os.path.isdir(mesh_folder_path):
                    continue

                # Paths for point clouds
                q=1

                # Check if initial and fin txt files exist in mesh folder
                initial_path = os.path.join(mesh_folder_path, "initial_pcd.txt")
                fin_path = os.path.join(mesh_folder_path, "fin_pcd.txt")
                face_path = os.path.join(mesh_folder_path, "faces.txt")
                if not os.path.exists(face_path):
                    face_path = None
                if os.path.exists(initial_path) and os.path.exists(fin_path) and os.path.exists(pcd_traj_path) and os.path.exists(pcd_traj2_path):
                    # print(f"Processing {mesh_folder_path}...")
                    l,m,r,rif,rt,f,i,t,pli,plf,plti,pltf = process_mesh_folder(mesh_folder_path, initial_path, fin_path, pcd_traj_path, pcd_traj2_path,CONST_P,face_path,face_t_path)
                    if l > loU:
                        loU = l
                        success = [l,m,r,rif,rt,f,i,t,pli,plf,plti,pltf]
                    mean = min(mean,m)
                    coverage_rate_r = min(coverage_rate_r,r)
                    coverage_rate_t = min(coverage_rate_t,t)
            if q==1:
                if success is not None:
                    l,m,r,rif,rt,f,i,t,pli,plf,plti,pltf = success[:]
                    with open(file_name, 'a') as file:
                        # print(original_path,"\t",r,"\t",rif,"\t",rt,file=file)
                        print(case_folder,"\t",cloth_type,"\t",l,"\t",m,"\t",r,"\t",rif,"\t",rt,"\t",f,"\t",i,"\t",plf,"\t",pli,"\t",t,"\t",plti,"\t",pltf,file=file)
                # break
                else:
                    initial_pcd_sim2 = load_point_cloud_from_txt(pcd_traj_path)

                    fin_pcd_sim2 = load_point_cloud_from_txt(pcd_traj2_path)

                    if face_t_path is not None:
                        faces2=load_cloth_mesh(face_t_path)
                        planeness_ti=check_planeness(initial_pcd_sim2.copy(),faces2)
                        planeness_tf=check_planeness(fin_pcd_sim2.copy(),faces2)
                        
                    with open(file_name, 'a') as file:
                        print(case_folder,"\t",cloth_type,"\t0\t1\t1\t1\t0\t0\t0\t5\t5\t0\t",planeness_ti,"\t",planeness_tf,file=file)
        json_path = os.path.join(eval_path, "metrics.json")
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
    eval_folder = "/home/transfer/.local/share/ov/pkg/isaac-sim-4.1.0/extension_examples/Cloth-Flod-in-isaac-sim/isaac_sim/output/11-22_1"
    # eval_folder = "/home/transfer/chenhn_data/cloth_eval_all/cloth_eval_data_all_uniG"  
    # eval_folder = "/home/transfer/chenhn/GPT-fabric-folding/eval result_copy/AllCornersInward/Top/2024-11-11"
    # eval_folder = "/home/transfer/chenhn/GPT-fabric-folding/eval result/DoubleStraight/Trousers/2024-11-12"
    # eval_folder  = "/home/transfer/chenhn/language_deformable/eval/single"
    # eval_folder = "/home/transfer/chenhn/GPT-fabric-folding/eval result_copy/DoubleTriangle/Dress/2024-11-11"
    # eval_folder = "/home/transfer/chenhn/GPT-fabric-folding/eval result_copy/AllCornersInward/Tshirt/2024-11-11"
    # eval_folder="/home/transfer/chenhn/language_deformable/eval/single"
    # eval_folder = "/home/transfer/chenhn_data/eval_ep/eval/output/output"
    process_eval_folder(eval_folder)
