import argparse
import numpy as np
import torch
import os
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

def process_chunk(chunk_files):
    partial_sum = None
    partial_sum_sq = None
    count = 0
    for volume_file in chunk_files:
        try:
            volume = torch.load(volume_file).numpy()
        except:
            print("Failed to load: ", volume_file)
            continue
        if partial_sum is None:
            partial_sum = np.zeros_like(volume, dtype=np.float64)
            partial_sum_sq = np.zeros_like(volume, dtype=np.float64)
        partial_sum += volume
        partial_sum_sq += volume ** 2
        count += 1
    return partial_sum, partial_sum_sq, count

def chunked_file_list(file_list, chunk_size):
    for i in range(0, len(file_list), chunk_size):
        yield file_list[i:i + chunk_size]

def init_worker(tqdm_lock):
    tqdm.set_lock(tqdm_lock)

def main(volume_dir):
    save_dir = os.path.dirname(os.path.normpath(volume_dir))
    print("Saving to {}".format(save_dir))
    volume_files = [os.path.join(volume_dir, f) for f in os.listdir(volume_dir) if f.endswith('.pt')]
    print("Found {} volume files.".format(len(volume_files)))
    num_workers = 16
    chunk_size = len(volume_files) // num_workers + (len(volume_files) % num_workers > 0)
    print("Chunk size: {}".format(chunk_size))
    
    tqdm_lock = multiprocessing.RLock()
    tqdm.set_lock(tqdm_lock)

    # Set up a multiprocessing pool and process each chunk in parallel
    with Pool(num_workers, initializer=init_worker, initargs=(tqdm_lock,)) as pool:
        chunk_results = list(tqdm(pool.imap(process_chunk, chunked_file_list(volume_files, chunk_size)),
                                  total=num_workers, desc="Processing chunks", unit="chunk"))

    global_sum = None
    global_sum_sq = None
    total_volumes = 0

    for partial_sum, partial_sum_sq, num_volumes in chunk_results:
        if global_sum is None:
            global_sum = partial_sum
            global_sum_sq = partial_sum_sq
        else:
            global_sum += partial_sum
            global_sum_sq += partial_sum_sq
        total_volumes += num_volumes

    mean_volume = global_sum / total_volumes
    std_volume = np.sqrt((global_sum_sq / total_volumes) - (mean_volume ** 2))

    torch.save(torch.from_numpy(mean_volume), os.path.join(save_dir, 'mean_volume_act.pt'))
    torch.save(torch.from_numpy(std_volume), os.path.join(save_dir, 'std_volume_act.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume_dir', type=str)
    args = parser.parse_args()
    main(args.volume_dir)
