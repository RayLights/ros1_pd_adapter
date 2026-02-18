import numpy as np
import math
import os
import pyquaternion


'''
select bbox
'''
def check_point_in_box(pts, box):
    """
    	pts[x,y,z]
    	box[c_x,c_y,c_z,dx,dy,dz,heading]
    """

    shift_x = pts[0] - box[0]
    shift_y = pts[1] - box[1]
    shift_z = pts[2] - box[2]
    cos_a = math.cos(box[6])
    sin_a = math.sin(box[6])
    dx,dy,dz = box[3], box[4], box[5]
    local_x = shift_x * cos_a + shift_y * sin_a
    local_y = shift_y * cos_a - shift_x * sin_a
    if(abs(shift_z)>dz/2.0 or abs(local_x)>dx/2.0 or abs(local_y)>dy/2.0):
        return False
    return True

def img2velodyne(calib_dir, img_id, p):
    """
    :param calib_dir
    :param img_id
    :param velo_box: (n,8,4)
    :return: (n,4)
    """
    calib_txt = os.path.join(calib_dir, img_id) + '.txt'
    calib_lines = [line.rstrip('\n') for line in open(calib_txt, 'r')]
    for calib_line in calib_lines:
        if 'P2' in calib_line:
            P2 = calib_line.split(' ')[1:]
            P2 = np.array(P2, dtype='float').reshape(3, 4)
        elif 'R0_rect' in calib_line:
            R0_rect = np.zeros((4, 4))
            R0 = calib_line.split(' ')[1:]
            R0 = np.array(R0, dtype='float').reshape(3, 3)
            R0_rect[:3, :3] = R0
            R0_rect[-1, -1] = 1
        elif 'velo_to_cam' in calib_line:
            velo_to_cam = np.zeros((4, 4))
            velo2cam = calib_line.split(' ')[1:]
            velo2cam = np.array(velo2cam, dtype='float').reshape(3, 4)
            velo_to_cam[:3, :] = velo2cam
            velo_to_cam[-1, -1] = 1

    pts_rect_hom = p
    pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_rect, velo_to_cam).T))

    return pts_lidar


'''
Corruptions
'''

def density_dec_bbox(pointcloud, severity):
    N, C = pointcloud.shape
    num = int(N*0.1)
    c = [int(0.1 * N), int(0.2 * N), int(0.3 * N), int(0.4 * N), int(0.5 * N)][severity]
    idx = np.random.choice(N, c, replace=False)
    pointcloud = np.delete(pointcloud, idx, axis=0)
    return pointcloud

def cutout_bbox(pointcloud, severity):
    N, C = pointcloud.shape
    #from 30 changed to 3000 to qualify kitti
    c = [(1,int(N*0.3)), (1,int(N*0.4)), (1,int(N*0.5)), (1,int(N*0.6)), (1,int(N*0.7))][severity]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        # pointcloud[idx.squeeze()] = 0
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    # print(pointcloud.shape)
    return pointcloud

def gaussian_noise_bbox(pointcloud, severity):
    N, C = pointcloud.shape # N*3
    c = [0.02, 0.04, 0.06, 0.08, 0.10][severity]
    jitter = np.random.normal(size=(N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc


def uniform_noise_bbox(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.02, 0.04, 0.06, 0.08, 0.10][severity]
    jitter = np.random.uniform(-c, c, (N, C))
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc


def impulse_noise_bbox(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N // 30, N // 25, N // 20, N // 15, N // 10][severity]
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.choice([-1, 1], size=(c, C)) * 0.1
    return pointcloud

'''
bbox_convert
'''

def to_Max2(points, gt_boxes_lidar):
    # shift
    points[:, :3] = points[:, :3] - gt_boxes_lidar[:3]
    # normalize to 2 units
    points[:, :3] = points[:, :3] / np.max(gt_boxes_lidar[3:6]) * 2
    # reversely rotate
    angle = -gt_boxes_lidar[6]
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array(
        [cosa, sina, 0.0,
         -sina, cosa, 0.0,
         0.0, 0.0, 1.0]).reshape(3, 3)
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    
    # FIX: Drop the reshape! Just horizontally stack the extra columns natively.
    points = np.hstack((points_rot, points[:, 3:]))
    return points

def to_Lidar(points, gt_boxes_lidar):
    angle = gt_boxes_lidar[6]
    # along_z
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array(
        [cosa, sina, 0.0,
         -sina, cosa, 0.0,
         0.0, 0.0, 1.0]).reshape(3, 3)
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    
    # FIX: Drop the reshape!
    points = np.hstack((points_rot, points[:, 3:]))
    
    # denormalize to lidar
    points[:, :3] = points[:, :3] * np.max(gt_boxes_lidar[3:6]) / 2
    # shift
    points[:, :3] = points[:, :3] + gt_boxes_lidar[:3]
    return points

# normalize
def normalize_gt(points, gt_box_ratio):
    """
    Args:
        points: N x 3+C
        gt_box_ratio: 3
    Returns:
        limit points to gt: N x 3+C
    """
    if points.shape[0] != 0:
        box_boundary_normalized = gt_box_ratio/np.max(gt_box_ratio)
        for i in range(3):
            indicator = np.max(np.abs(points[:,i])) / box_boundary_normalized[i]
            if indicator > 1:
                points[:,i] /= indicator
    return points

def shear_bbox(pointcloud, severity, gt_boxes):
    N, _ = pointcloud.shape
    c = [0.05, 0.1, 0.15, 0.2, 0.25][severity]

    pts_obj_max2 = to_Max2(pointcloud, gt_boxes)
    b = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    d = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    e = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    f = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    matrix = np.array([1, 0, b,
                       d, 1, e,
                       f, 0, 1]).reshape(3, 3)
                       
    new_pc = np.matmul(pts_obj_max2[:, :3], matrix).astype('float32')

    # FIX: Use [:, 3:] to keep ALL extra columns (Intensity AND Ring) instead of just [:, 3]
    pts_obj_max2_crp = np.hstack((new_pc, pts_obj_max2[:, 3:]))
    
    pts_obj_max2_crp = normalize_gt(pts_obj_max2_crp, gt_boxes[3:6])
    pts_cor = to_Lidar(pts_obj_max2_crp, gt_boxes)
    return pts_cor


def scale_bbox(pointcloud, severity, gt_boxes):
    N, _ = pointcloud.shape
    c = [0.04, 0.08, 0.12, 0.16, 0.20][severity]

    pts_obj_max2 = to_Max2(pointcloud, gt_boxes)
    xs, ys, zs = 1.0, 1.0, 1.0
    r = np.random.randint(0,3)
    t = np.random.choice([-1,1])
    if r == 0:
        xs += c * t
    elif r == 1:
        ys += c * t
    else:
        zs += c * t
        
    # FIX: Only multiply the 3D coordinates (X, Y, Z) by a 3x3 matrix
    matrix = np.array([[xs,0,0],[0,ys,0],[0,0,zs]])
    new_pc = np.matmul(pts_obj_max2[:, :3], matrix)
    
    # FIX: Re-attach Intensity and Ring
    pts_obj_max2_crp = np.hstack((new_pc, pts_obj_max2[:, 3:]))
    
    pts_obj_max2_crp[:,2] += (zs-1) * gt_boxes[5]/np.max(gt_boxes[3:6])
    pts_cor = to_Lidar(pts_obj_max2_crp, gt_boxes)
    return pts_cor


def rotation_bbox(pointcloud, severity, gt_boxes):
    N, _ = pointcloud.shape
    c = [1, 3, 5, 7, 9][severity]
    beta = np.random.uniform(c-1,c+1) * np.random.choice([-1,1]) * np.pi / 180.
    pts_obj_max2 = to_Max2(pointcloud, gt_boxes)
    
    matrix_roration_z = np.array([[np.cos(beta),np.sin(beta),0],[-np.sin(beta),np.cos(beta),0],[0,0,1]])
    pts_rotated = np.matmul(pts_obj_max2[:,:3], matrix_roration_z)
    
    # FIX: Drop the reshape, use [:, 3:]
    pts_obj_max2_crp = np.hstack((pts_rotated, pts_obj_max2[:, 3:]))
    
    pts_cor = to_Lidar(pts_obj_max2_crp, gt_boxes)
    return pts_cor

def moving_noise_bbox(pointcloud, severity):
    # for kitti: the x is forward
    N, C = pointcloud.shape
    c = [0.2, 0.4, 0.6, 0.8, 1.0][severity]
    m1, m2 = float(c/2), c
    x_min, x_max = np.min(pointcloud[:,0]), np.max(pointcloud[:,0])
    x_l = (x_max - x_min) / 3
    
    # --- VECTORIZED REPLACEMENT ---
    # Shift points in the front third of the box
    pointcloud[(pointcloud[:,0] > x_min) & (pointcloud[:,0] <= x_min + x_l), 0] += m1
    
    # Shift points in the back two-thirds of the box
    pointcloud[(pointcloud[:,0] > x_min + x_l) & (pointcloud[:,0] <= x_max), 0] += m2
    # ------------------------------
    
    return pointcloud


MAP = {
    'density_dec_bbox':density_dec_bbox,
    'cutout_bbox':cutout_bbox,
    'gaussian_noise_bbox':gaussian_noise_bbox,
    'uniform_noise_bbox':uniform_noise_bbox,
    'impulse_noise_bbox':impulse_noise_bbox,
    'scale_bbox':scale_bbox,
    'shear_bbox':shear_bbox,
    'rotation_bbox':rotation_bbox,
    'moving_noise_bbox':moving_noise_bbox,
    'move_bbox': moving_noise_bbox,
}



def pick_bbox(cor, slevel, data, pointcloud):
    xyz = pointcloud
    
    # --- BULLETPROOF DATA EXTRACTION ---
    # 1. Unwrap the data if it got packed into a tuple or list by the ROS wrapper
    bboxes = data[0] if isinstance(data, (tuple, list)) else data
    
    # 2. Convert to Numpy and handle empty data
    bboxes = np.array(bboxes)
    if bboxes.size == 0:
        return xyz  # No objects detected, return the clean pointcloud instantly
        
    # 3. Force it into an N x 7 2D Array (This fixes the IndexError!)
    bboxes = bboxes.reshape(-1, 7)
    
    for box in bboxes:
        # Box parameters: [x, y, z, dx, dy, dz, heading]
        cos_a = np.cos(box[6])
        sin_a = np.sin(box[6])
        
        # 1. FAST VECTORIZED MATH: Calculate shifts for all points at once
        shift_x = xyz[:, 0] - box[0]
        shift_y = xyz[:, 1] - box[1]
        shift_z = xyz[:, 2] - box[2]
        
        local_x = shift_x * cos_a + shift_y * sin_a
        local_y = shift_y * cos_a - shift_x * sin_a
        
        # 2. FAST FILTERING: Create a boolean mask of which points are inside the 3D box
        inside_mask = (np.abs(shift_z) <= box[5] / 2.0) & \
                      (np.abs(local_x) <= box[3] / 2.0) & \
                      (np.abs(local_y) <= box[4] / 2.0)
                      
        # Split the pointcloud instantly
        pcd_2 = xyz[inside_mask]   # Points INSIDE the box
        pcd_1 = xyz[~inside_mask]  # Points OUTSIDE the box
        
        # 3. Apply Corruption
        if len(pcd_2) != 0:
            if cor in ['shear_bbox', 'scale_bbox', 'rotation_bbox']:
                pcd_2 = MAP[cor](pcd_2, slevel, box)
            else:
                pcd_2 = MAP[cor](pcd_2, slevel)
                
            # Stitch the corrupted object points back together with the uncorrupted background
            xyz = np.append(pcd_2, pcd_1, axis=0)
            
    return xyz








