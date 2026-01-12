import time
import subprocess
import threading
import psutil
import csv
import datetime
import os
import sys
import torch

class ResourceMonitor:
    def __init__(self, log_file="benchmark_parallel_log.txt", interval=1.0):
        self.log_file = log_file
        self.interval = interval
        self.monitoring = False
        self.thread = None
        self.current_proc_count = 0
        self.headers_written = False

    def get_gpu_stats(self):
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            util, mem = result.strip().split(", ")
            return float(util), float(mem)
        except:
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / (1024 * 1024)
                return 0.0, mem
            return 0.0, 0.0

    def _monitor_loop(self):
        # Check if file exists to avoid overwriting headers if we append (though we restart logic here)
        mode = "a" if self.headers_written else "w"
        
        with open(self.log_file, mode, newline="") as f:
            writer = csv.writer(f)
            if not self.headers_written:
                writer.writerow(["Timestamp", "NumProcesses", "CPU_Percent", "RAM_Used_MB", "GPU_Util_Percent", "GPU_Mem_MB"])
                self.headers_written = True

            while self.monitoring:
                cpu_pct = psutil.cpu_percent(interval=None)
                ram_used = psutil.virtual_memory().used / (1024 * 1024)
                gpu_util, gpu_mem = self.get_gpu_stats()
                
                writer.writerow([
                    datetime.datetime.now().isoformat(),
                    self.current_proc_count,
                    f"{cpu_pct:.1f}",
                    f"{ram_used:.1f}",
                    f"{gpu_util:.1f}",
                    f"{gpu_mem:.1f}"
                ])
                f.flush()
                time.sleep(self.interval)

    def start(self, proc_count):
        self.current_proc_count = proc_count
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join()

def run_benchmark():
    log_file = "benchmark_parallel_log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)
    
    monitor = ResourceMonitor(log_file)
    
    # Process counts to test
    process_counts = [2, 4, 8, 16]
    
    print(f"Starting parallel benchmark. Logging to {log_file}")
    
    for count in process_counts:
        print(f"\n--- Testing with {count} parallel processes ---")
        
        # Start monitoring
        monitor.start(count)
        
        workers = []
        # Store output readers/queues if we wanted per-worker stats, 
        # but here we just read stdout line by line.
        
        # We need a way to count images. We'll start processes and read their stdout non-blockingly or using threads.
        
        worker_processes = []
        image_counts = [0] * count
        stop_events = [threading.Event() for _ in range(count)]
        
        def track_worker(proc, index):
            for line in iter(proc.stdout.readline, b''):
                line = line.decode('utf-8').strip()
                if "IMAGE_DONE" in line:
                    image_counts[index] += 1
                elif "READY" in line:
                    # Could use this to start timing effectively
                    pass
                if stop_events[index].is_set():
                    break

        # Launch workers
        start_launch = time.time()
        for i in range(count):
            p = subprocess.Popen(
                ["python3", "slop_worker.py", str(i)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            worker_processes.append(p)
            t = threading.Thread(target=track_worker, args=(p, i))
            t.daemon = True
            t.start()
            
        print(f"Launched {count} workers. Waiting for them to warm up/stabilize...")
        # Simple wait for startup - in a real rigorous bench we'd wait for READY signals
        time.sleep(10) 
        
        print("Resetting counters and starting 1-minute measurement...")
        # Reset counts after warmup
        for i in range(count):
            image_counts[i] = 0
            
        measurement_start = time.time()
        time.sleep(60)
        measurement_end = time.time()
        
        # Signal stop
        for evt in stop_events:
            evt.set()
            
        total_images = sum(image_counts)
        elapsed = measurement_end - measurement_start
        
        print(f"Killing workers...")
        for p in worker_processes:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                
        monitor.stop()
        
        print(f"Results for {count} processes:")
        print(f"  Total Images: {total_images}")
        print(f"  Throughput: {total_images / elapsed * 60:.2f} images/minute")
        
        # Cooldown
        print("Cooling down for 5 seconds...")
        time.sleep(5)

if __name__ == "__main__":
    run_benchmark()
