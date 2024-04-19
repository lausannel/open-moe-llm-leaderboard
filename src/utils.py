import pandas as pd
from huggingface_hub import snapshot_download
import subprocess
import re
import os

try:
    from src.display.utils import GPU_TEMP, GPU_Mem, GPU_Power, GPU_Util, GPU_Name
except:
    print("local debug: from display.utils")
    from display.utils import GPU_TEMP, GPU_Mem, GPU_Power, GPU_Util, GPU_Name

def my_snapshot_download(repo_id, revision, local_dir, repo_type, max_workers):
    for i in range(10):
        try:
            snapshot_download(
                repo_id=repo_id, revision=revision, local_dir=local_dir, repo_type=repo_type, max_workers=max_workers
            )
            return
        except Exception as e:
            print(f"Failed to download {repo_id} at {revision} with error: {e}. Retrying...")
            import time

            time.sleep(60)
    return


def get_dataset_url(row):
    dataset_name = row["Benchmark"]
    dataset_url = row["Dataset Link"]
    benchmark = f'<a target="_blank" href="{dataset_url}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{dataset_name}</a>'
    return benchmark


def get_dataset_summary_table(file_path):
    df = pd.read_csv(file_path)

    df["Benchmark"] = df.apply(lambda x: get_dataset_url(x), axis=1)

    df = df[["Category", "Benchmark", "Data Split", "Data Size", "Language"]]

    return df

def parse_nvidia_smi():
    visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', None)
    if visible_devices is not None:
        gpu_indices = visible_devices.split(',')
    else:
        # Query all GPU indices if CUDA_VISIBLE_DEVICES is not set
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Failed to query GPU indices.")
            return []
        gpu_indices = result.stdout.strip().split('\n')
    print(f"gpu_indices: {gpu_indices}")
    gpu_stats = []

    gpu_info_pattern = re.compile(r'(\d+)C\s+P\d+\s+(\d+)W / \d+W\s+\|\s+(\d+)MiB / \d+MiB\s+\|\s+(\d+)%')
    gpu_name_pattern = re.compile(r'NVIDIA\s+([\w\s]+?\d+GB)')

    gpu_name = ""
    for index in gpu_indices:
        result = subprocess.run(['nvidia-smi', '-i', index], capture_output=True, text=True)
        output = result.stdout.strip()
        lines = output.split("\n")
        for line in lines:
            match = gpu_info_pattern.search(line)
            name_match = gpu_name_pattern.search(line)
            gpu_info = {}
            if name_match:
                gpu_name = name_match.group(1).strip()
            if match:
                temp, power_usage, mem_usage, gpu_util = map(int, match.groups())
                gpu_info.update({
                    GPU_TEMP: temp,
                    GPU_Power: power_usage,
                    GPU_Mem: mem_usage,
                    GPU_Util: gpu_util
                })

            if len(gpu_info) >= 4:
                gpu_stats.append(gpu_info)
    print(f"len(gpu_stats): {len(gpu_stats)}")
    gpu_name = f"{len(gpu_stats)}x{gpu_name}"
    gpu_stats_total = {
                        GPU_TEMP: 0,
                        GPU_Power: 0,
                        GPU_Mem: 0,
                        GPU_Util: 0,
                        GPU_Name: gpu_name
                    }
    for gpu_stat in gpu_stats:
        gpu_stats_total[GPU_TEMP] += gpu_stat[GPU_TEMP]
        gpu_stats_total[GPU_Power] += gpu_stat[GPU_Power]
        gpu_stats_total[GPU_Mem] += gpu_stat[GPU_Mem]
        gpu_stats_total[GPU_Util] += gpu_stat[GPU_Util]

    gpu_stats_total[GPU_TEMP] /= len(gpu_stats)
    gpu_stats_total[GPU_Power] /= len(gpu_stats)
    gpu_stats_total[GPU_Util] /= len(gpu_stats)
    return [gpu_stats_total]

def monitor_gpus(stop_event, interval, stats_list):
    while not stop_event.is_set():
        gpu_stats = parse_nvidia_smi()
        if gpu_stats:
            stats_list.extend(gpu_stats)
        stop_event.wait(interval)

def analyze_gpu_stats(stats_list):
    # Check if the stats_list is empty, and return None if it is
    if not stats_list:
        return None

    # Initialize dictionaries to store the stats
    avg_stats = {}
    max_stats = {}

    # Calculate average stats, excluding 'GPU_Mem'
    for key in stats_list[0].keys():
        if key != GPU_Mem and key != GPU_Name:
            total = sum(d[key] for d in stats_list)
            avg_stats[key] = total / len(stats_list)

    # Calculate max stats for 'GPU_Mem'
    max_stats[GPU_Mem] = max(d[GPU_Mem] for d in stats_list)
    if GPU_Name in stats_list[0]:
        avg_stats[GPU_Name] = stats_list[0][GPU_Name]
    # Update average stats with max GPU memory usage
    avg_stats.update(max_stats)

    return avg_stats

if __name__ == "__main__":
    print(analyze_gpu_stats(parse_nvidia_smi()))
