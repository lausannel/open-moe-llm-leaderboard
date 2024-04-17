import pandas as pd
from huggingface_hub import snapshot_download
import subprocess
import re
try:
    from src.display.utils import GPU_TEMP, GPU_Mem, GPU_Power, GPU_Util
except:
    print("local debug: from display.utils")
    from display.utils import GPU_TEMP, GPU_Mem, GPU_Power, GPU_Util

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
    # Execute the nvidia-smi command
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    output = result.stdout.strip()

    # Initialize data storage
    gpu_stats = []

    # Regex to extract the relevant data for each GPU
    gpu_info_pattern = re.compile(r'(\d+)C\s+P\d+\s+(\d+)W / \d+W\s+\|\s+(\d+)MiB / \d+MiB\s+\|\s+(\d+)%')
    lines = output.split('\n')

    for line in lines:
        match = gpu_info_pattern.search(line)
        if match:
            temp, power_usage, mem_usage, gpu_util = map(int, match.groups())
            gpu_stats.append({
                GPU_TEMP: temp,
                GPU_Power: power_usage,
                GPU_Mem: mem_usage,
                GPU_Util: gpu_util
            })

    gpu_stats_total = {
                        GPU_TEMP: 0,
                        GPU_Power: 0,
                        GPU_Mem: 0,
                        GPU_Util: 0
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
    if not stats_list:
        return None
    avg_stats = {key: sum(d[key] for d in stats_list) / len(stats_list) for key in stats_list[0]}
    return avg_stats


if __name__ == "__main__":
    print(analyze_gpu_stats(parse_nvidia_smi()))
