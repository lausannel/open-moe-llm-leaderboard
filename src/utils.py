import pandas as pd
from huggingface_hub import snapshot_download


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
