# FVTON - Virtual Try-On for Fooocus

This project integrates advanced Virtual Try-On and Masking improvements into Fooocus.

## Hardware Requirements (IMPORTANT)

> [!WARNING]
> **Disk Space**: Fooocus requires at least **50GB - 100GB** of disk space to accommodate base models, refiners, and the virtual try-on models. 
> Your current Pod (20GB) is too small and will crash during download.

## Quick Setup on RunPod

To deploy this project permanently on RunPod, follow these steps:

1.  **Open RunPod Terminal**
2.  **Start a persistent session (so it runs 24/7):**
    ```bash
    tmux new -s fvton
    ```
3.  **Run the setup script:**
    ```bash
    git clone https://github.com/umairwaheed12/FVTON.git && cd FVTON && bash runpod_setup.sh
    ```
4.  **Detach from session:** Press `Ctrl + B`, then `D`.

## Permanent Access (No 72-hour limit)

To access your app anytime without relying on Gradio share links:

1.  Go to your **RunPod Dashboard**.
2.  Find your Pod and click **"Connect"**.
3.  Select **"HTTP Proxy"** for Port **7865**.
4.  Share that URL with your users!

## Project Structure
- `Fooocus/`: The core application code.
- `runpod_setup.sh`: Automated environment setup and launcher.
- `models_downloader.py`: Centralized script to download all required AI models and dependencies.
- `config.txt`: Configuration for Fooocus.
- `run.bat`: Windows launcher.

---
*Created with ♥ for Virtual Try-On users.*
