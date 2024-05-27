import os
import argparse
import subprocess
from multiprocessing import Process

PORT = 8000
def run_on_gpu(obj_chunk, gpu_id, args):
    for obj in obj_chunk:
        obj_name = os.path.basename(obj)
        if os.path.exists(os.path.join(args.output_path, obj_name, "point_cloud/iteration_30000/point_cloud.ply")):
            print("skip", obj_name)
            continue
        command = f"CUDA_VISIBLE_DEVICES={gpu_id} conda run --no-capture-output -n gaussian_splatting python train.py -s {obj} -m {os.path.join(args.output_path, obj_name)} --dataset_type shapenet --white_background --test_iterations 7000 30000 --port {PORT+gpu_id} --densification_interval 50 --sh_degree 0 --no_tqdm"
        try:  
            subprocess.run(command, shell=True, check=True)  
        except Exception as e:  
            print(f"An error occurred while running command: {command}")  
            print(str(e)) 

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", type=str, default="./example_data/")
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--end_idx", type=int, default=1)
parser.add_argument("--output_path", type=str, default="./output_dc_fitting/")
parser.add_argument("--txt_file", type=str, default="./example_data/shapenet_car.txt")
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--N_max_pts", type=int, default=32768)
args = parser.parse_args()

with open(args.txt_file, "r") as f:
    objs = f.read().splitlines()[args.start_idx:args.end_idx]

objs = [os.path.join(args.source_path, obj_path) for obj_path in objs]

obj_chunks = chunkify(objs, args.num_gpus)

processes = []
for gpu_id, obj_chunk in enumerate(obj_chunks):
    p = Process(target=run_on_gpu, args=(obj_chunk, gpu_id, args))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
