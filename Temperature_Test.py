import subprocess
import time
from datetime import datetime, timedelta

# Parameters
temp_threshold = 80
duration_threshold = timedelta(minutes=5)
check_interval = 60  # seconds

# Initialize tracking
gpu_temp_history = {}

def get_gpu_temperatures():
    """Get the current temperatures of all GPUs."""
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,temperature.gpu', '--format=csv,noheader'], encoding='utf-8')
        temps = [(int(line.split(',')[0].strip()), int(line.split(',')[1].strip())) for line in output.strip().split('\n')]
        return temps
    except subprocess.CalledProcessError as e:
        print("Failed to query GPU temperatures:", e)
        return []

def get_high_usage_process_on_gpu(gpu_index):
    """Get the PID of the process with the highest GPU usage on a specific GPU."""
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], encoding='utf-8')
        if not output.strip():
            return None
        processes = [line.split(', ') for line in output.strip().split('\n')]
        # Filter processes by GPU index, using UUID as a proxy
        gpu_uuid = get_gpu_uuid(gpu_index)
        processes = [proc for proc in processes if proc[0] == gpu_uuid]
        if not processes:
            return None
        # Sort by GPU memory usage (descending)
        processes.sort(key=lambda x: int(x[2].replace(' MiB', '')), reverse=True)
        return processes[0][1]  # Return PID of the process using the most GPU memory
    except subprocess.CalledProcessError as e:
        print("Failed to query GPU processes:", e)
        return None

def get_gpu_uuid(gpu_index):
    """Get the UUID of the GPU by index."""
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], encoding='utf-8')
        for line in output.strip().split('\n'):
            index, uuid = line.split(', ')
            if int(index) == gpu_index:
                return uuid.strip()
        return None
    except subprocess.CalledProcessError as e:
        print("Failed to get GPU UUID:", e)
        return None

def kill_process(pid):
    """Kill the process with the given PID."""
    try:
        subprocess.check_call(['sudo', 'kill', '-9', str(pid)])
        print(f"Successfully killed process {pid}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to kill process {pid}:", e)

while True:
    temperatures = get_gpu_temperatures()
    print(f"Checking temperatures: {temperatures}")  # Diagnostic print
    for gpu_index, temp in temperatures:
        print(f"GPU {gpu_index} Temperature: {temp}")  # Diagnostic print
        if temp > temp_threshold:
            print(f"GPU {gpu_index} exceeds threshold with temperature {temp}")  # Diagnostic print
            if gpu_index not in gpu_temp_history:
                gpu_temp_history[gpu_index] = datetime.now()
            elif datetime.now() - gpu_temp_history[gpu_index] > duration_threshold:
                pid = get_high_usage_process_on_gpu(gpu_index)
                if pid:
                    kill_process(pid)
                    gpu_temp_history[gpu_index] = datetime.now()  # Reset timer after action
                else:
                    print(f"No high-usage process found for GPU {gpu_index}")
        else:
            if gpu_index in gpu_temp_history:
                del gpu_temp_history[gpu_index]  # Reset history if temperature goes below threshold
    time.sleep(check_interval)
