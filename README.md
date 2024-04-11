---
title: OPEN-MOE-LLM-LEADERBOARD
emoji: 🔥
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 4.9.0
app_file: app.py
pinned: true
license: apache-2.0
fullWidth: true
tags:
  - leaderboard
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Local development

Create a virtual environment and install the dependencies:

```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
pip install -r requirements.txt
```

**Follow the instructions in Dockerfile to install other necessary dependencies.**

Start the backend server in debug mode:

```bash
python backend-cli.py --debug
```