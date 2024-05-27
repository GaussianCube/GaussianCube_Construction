import os
import torch
import numpy as np
import argparse
from plyfile import PlyData
from typing import NamedTuple
import concurrent.futures
import ot
import time
from lapjv import lapjv
from multiprocessing import Pool  
import open3d as o3d


num_segments = 4
segment_size = 8192


class GaussianPointCloud(NamedTuple):
    xyz : torch.tensor
    features_dc : torch.tensor
    features_rest : torch.tensor
    opacity : torch.tensor
    scaling : torch.tensor
    rotation : torch.tensor


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def inverse_scaling(x):
    return torch.log(x)


def process_single_wrapper(line, source_root, save_root, max_sh_degree, bound, visuzalize_mapping):  
    # Call the original processing function with the provided arguments  
    process_single(os.path.join(source_root, line), save_root, max_sh_degree, bound, visuzalize_mapping)


def init_volume_grid(bound=0.45, num_pts_each_axis=32):
    # Define the range and number of points  
    start = -bound
    stop = bound
    num_points = num_pts_each_axis  # Adjust the number of points to your preference  
    
    # Create a linear space for each axis  
    x = np.linspace(start, stop, num_points)  
    y = np.linspace(start, stop, num_points)  
    z = np.linspace(start, stop, num_points)  
    
    # Create a 3D grid of points using meshgrid  
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  
    
    # Stack the grid points in a single array of shape (N, 3)  
    xyz = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T  
    
    return xyz


def compute_lap(p1_segment, p2_segment, scaling_factor=1000, index=0):  
    cost_matrix = ot.dist(p1_segment, p2_segment, metric='sqeuclidean')  
    # Scale to integers for faster computation
    scaled_cost_matrix = np.rint(cost_matrix * scaling_factor).astype(int)    
    x, y, cost = lapjv(scaled_cost_matrix)
    return cost, x, y, index


def load_ply(path, max_sh_degree=3):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3))
    features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # (P,F*SH_coeffs)
    features_extra = features_extra

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = torch.tensor(xyz, dtype=torch.float, device="cpu")
    features_dc = torch.tensor(features_dc, dtype=torch.float, device="cpu").contiguous()
    features_rest = torch.tensor(features_extra, dtype=torch.float, device="cpu").contiguous()
    opacity = torch.tensor(opacities, dtype=torch.float, device="cpu")
    scaling = torch.tensor(scales, dtype=torch.float, device="cpu")
    rotation = torch.tensor(rots, dtype=torch.float, device="cpu")
    print("xyz shape: {} \t features_dc shape: {} \t features_rest shape: {} \t opacity shape: {} \t scaling shape: {} \t rotation shape: {}".format(xyz.shape, features_dc.shape, features_rest.shape, opacity.shape, scaling.shape, rotation.shape))
    return GaussianPointCloud(xyz, features_dc, features_rest, opacity, scaling, rotation)


def padding_input(point_cloud, num_pts=32768, bound=0.45):
    xyz = point_cloud.xyz
    if xyz.shape[0] > num_pts:
        raise ValueError("The number of points in the input point cloud is larger than the maximum number of points allowed.")
    elif xyz.shape[0] == num_pts:
        return point_cloud
    else:
        padding_num = num_pts - xyz.shape[0]
        padding_xyz = torch.tensor([bound, bound, bound], dtype=torch.float).unsqueeze(0).repeat(padding_num, 1)
        padding_features_dc = torch.zeros((padding_num, point_cloud.features_dc.shape[1]), dtype=torch.float)
        padding_features_rest = torch.zeros((padding_num, point_cloud.features_rest.shape[1]), dtype=torch.float)
        padding_opacity = inverse_sigmoid(torch.ones((padding_num, point_cloud.opacity.shape[1]), dtype=torch.float) * 1e-6)
        padding_scaling = inverse_scaling(torch.ones((padding_num, point_cloud.scaling.shape[1]), dtype=torch.float) * 1e-6)
        padding_rotation = torch.nn.functional.normalize(torch.ones((padding_num, point_cloud.rotation.shape[1]), dtype=torch.float))
        new_xyz = torch.cat([xyz, padding_xyz], dim=0)
        new_features_dc = torch.cat([point_cloud.features_dc, padding_features_dc], dim=0)
        new_features_rest = torch.cat([point_cloud.features_rest, padding_features_rest], dim=0)
        new_opacity = torch.cat([point_cloud.opacity, padding_opacity], dim=0)
        new_scaling = torch.cat([point_cloud.scaling, padding_scaling], dim=0)
        new_rotation = torch.cat([point_cloud.rotation, padding_rotation], dim=0)
        new_point_cloud = GaussianPointCloud(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        return new_point_cloud


def lag_segment_matching(p1, p2):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  
        futures = {executor.submit(compute_lap,  
                                p1[i*segment_size:(i+1)*segment_size],  
                                p2[i*segment_size:(i+1)*segment_size],  
                                scaling_factor=1000, index=i): i for i in range(num_segments)}  
    
        results = [None] * num_segments  # Prepare a list to hold results in order  
        for future in concurrent.futures.as_completed(futures):  
            cost, x, y, index = future.result()  
            results[index] = (cost, x, y)  # Store the results in the corresponding index 
    corrs_2_to_1 = np.concatenate([y + i*segment_size for i, (cost, x, y) in enumerate(results)], axis=0)
    corrs_1_to_2 = np.concatenate([x + i*segment_size for i, (cost, x, y) in enumerate(results)], axis=0)
    return corrs_2_to_1, corrs_1_to_2


def activate_volume(volume, max_sh_degree=0):
    scaling_activation = torch.exp
    opacity_activation = torch.sigmoid
    rotation_activation = torch.nn.functional.normalize

    sh_dim = 3 * ((max_sh_degree + 1) ** 2 - 1)
    H, W, D, C = volume.shape
    volume = volume.reshape(-1, volume.shape[-1])
    volume[..., 6+sh_dim:7+sh_dim] = opacity_activation(volume[..., 6+sh_dim:7+sh_dim])
    volume[..., 7+sh_dim:10+sh_dim] = scaling_activation(volume[..., 7+sh_dim:10+sh_dim])
    volume[..., 10+sh_dim:] = rotation_activation(volume[..., 10+sh_dim:])
    volume = volume.reshape(H, W, D, -1)
    return volume


def process_single(source_root, save_root, max_sh_degree=0, bound=0.45, visuzalize_mapping=False):
    os.makedirs(os.path.join(save_root, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "volume"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "volume_act"), exist_ok=True)
    source_name = os.path.basename(os.path.normpath(source_root))
    source_file = os.path.join(source_root, "point_cloud/iteration_30000/point_cloud.ply")

    if not os.path.exists(source_file):
        print("File {} does not exist. Skip.".format(source_file))
        return
    if os.path.exists(os.path.join(save_root, "volume", source_name+".pt")):
        print("File {} already exists. Skip.".format(source_name))
        return
    
    generated_gaussian = load_ply(source_file, max_sh_degree)
    std_volume = init_volume_grid(bound=bound, num_pts_each_axis=32)
    generated_gaussian_padded = padding_input(generated_gaussian, bound=bound)
    xyz = generated_gaussian_padded.xyz.cpu().numpy()

    sorted_indices = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))  
    xyz = xyz[sorted_indices]
    
    start = time.time()
    corrs_2_to_1, corrs_1_to_2 = lag_segment_matching(xyz, std_volume)
    print("Time taken: {}".format(time.time() - start))

    ########## Point Cloud Visualization ##########
    if visuzalize_mapping:
        min_old, max_old = -bound, bound  
        min_new, max_new = 0, 1  
        p2 = std_volume.copy()
        p2 = (p2 - min_old) / (max_old - min_old) * (max_new - min_new) + min_new 
        colors = p2[corrs_1_to_2]

        mapping_of_p1 = o3d.geometry.PointCloud()  
        mapping_of_p1.points = o3d.utility.Vector3dVector(xyz)  
        mapping_of_p1.colors = o3d.utility.Vector3dVector(colors)  # Assuming 'colors' is a NumPy array with the same length as p1 
        os.makedirs(os.path.join(save_root, "point_cloud"), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(save_root, "point_cloud", source_name+".ply"), mapping_of_p1) 
    ###############################################
    
    xyz_offset = torch.from_numpy(xyz[corrs_2_to_1] - std_volume)
    new_gaussian = GaussianPointCloud(xyz_offset, generated_gaussian_padded.features_dc[sorted_indices][corrs_2_to_1], generated_gaussian_padded.features_rest[sorted_indices][corrs_2_to_1], generated_gaussian_padded.opacity[sorted_indices][corrs_2_to_1], generated_gaussian_padded.scaling[sorted_indices][corrs_2_to_1], generated_gaussian_padded.rotation[sorted_indices][corrs_2_to_1])

    new_gaussian_dict = {"xyz": new_gaussian.xyz, "features_dc": new_gaussian.features_dc, "features_rest": new_gaussian.features_rest, "opacity": new_gaussian.opacity, "scaling": new_gaussian.scaling, "rotation": new_gaussian.rotation}
    torch.save(new_gaussian_dict, os.path.join(save_root, "raw", source_name+".pt"))
    
    if new_gaussian.features_rest.shape[-1] == 0:
        volume = torch.cat([new_gaussian.xyz, new_gaussian.features_dc, new_gaussian.opacity, new_gaussian.scaling, new_gaussian.rotation], dim=-1).reshape(32, 32, 32, -1)
        act_volume = activate_volume(volume.clone(), max_sh_degree)
    else:
        volume = torch.cat([new_gaussian.xyz, new_gaussian.features_dc, new_gaussian.features_rest, new_gaussian.opacity, new_gaussian.scaling, new_gaussian.rotation], dim=-1).reshape(32, 32, 32, -1)
        act_volume = activate_volume(volume.clone(), max_sh_degree)
    torch.save(volume, os.path.join(save_root, "volume", source_name+".pt"))
    torch.save(act_volume, os.path.join(save_root, "volume_act", source_name+".pt"))

    return new_gaussian


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=str, default="./output_dc_fitting/")
    parser.add_argument("--save_root", type=str, default="./output_gaussiancube/")
    parser.add_argument("--max_sh_degree", type=int, default=0)
    parser.add_argument("--txt_file", type=str, default="./example_data/shapenet_car.txt")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1)
    parser.add_argument("--bound", type=float, default=0.45)
    parser.add_argument("--visuzalize_mapping", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    with open(args.txt_file, "r") as f:
        lines = f.read().splitlines()[args.start_idx:args.end_idx]
  
    # Number of worker processes to use  
    num_workers = args.num_workers
  
    # Create a pool of workers and map the processing function to the data  
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:  
        # Submit all the tasks and wait for them to complete  
        futures = [executor.submit(process_single_wrapper, line, args.source_root, args.save_root, args.max_sh_degree, args.bound, args.visuzalize_mapping) for line in lines]  
        for future in futures:  
            future.result()  # You can add error handling here if needed 
