# Advanced Performance Monitoring System
import psutil
import GPUtil
import time
import threading
from collections import deque
import json
from datetime import datetime

class PerformanceMonitor:
    """Real-time system performance monitoring"""
    
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.cpu_history = deque(maxlen=max_history)
        self.memory_history = deque(maxlen=max_history)
        self.gpu_history = deque(maxlen=max_history)
        self.inference_history = deque(maxlen=max_history)
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start background performance monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("⏹️ Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_history.append({
                'timestamp': timestamp,
                'value': cpu_percent
            })
            
            # Memory Usage
            memory = psutil.virtual_memory()
            self.memory_history.append({
                'timestamp': timestamp,
                'value': memory.percent,
                'used_gb': memory.used / (1024**3),
                'total_gb': memory.total / (1024**3)
            })
            
            # GPU Usage (if available)
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    self.gpu_history.append({
                        'timestamp': timestamp,
                        'utilization': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'temperature': gpu.temperature
                    })
            except:
                pass  # GPU monitoring not available
            
            time.sleep(1)  # Monitor every second
    
    def log_inference(self, model_name, inference_time, memory_used, accuracy):
        """Log inference performance metrics"""
        self.inference_history.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'model': model_name,
            'inference_time': inference_time,
            'memory_used': memory_used,
            'accuracy': accuracy,
            'throughput': 1000 / inference_time  # FPS
        })
    
    def get_current_stats(self):
        """Get current system statistics"""
        stats = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add GPU stats if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                stats['gpu'] = {
                    'utilization': gpu.load * 100,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature
                }
        except:
            stats['gpu'] = None
        
        return stats
    
    def get_performance_summary(self):
        """Generate performance summary report"""
        if not self.inference_history:
            return "No inference data available"
        
        # Calculate statistics
        inference_times = [entry['inference_time'] for entry in self.inference_history]
        throughputs = [entry['throughput'] for entry in self.inference_history]
        
        summary = {
            'total_inferences': len(self.inference_history),
            'avg_inference_time': sum(inference_times) / len(inference_times),
            'min_inference_time': min(inference_times),
            'max_inference_time': max(inference_times),
            'avg_throughput': sum(throughputs) / len(throughputs),
            'models_tested': len(set(entry['model'] for entry in self.inference_history))
        }
        
        return summary
    
    def export_data(self, filename=None):
        """Export monitoring data to JSON"""
        if not filename:
            filename = f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'cpu_history': list(self.cpu_history),
            'memory_history': list(self.memory_history),
            'gpu_history': list(self.gpu_history),
            'inference_history': list(self.inference_history),
            'summary': self.get_performance_summary(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Performance data exported to {filename}")
        return filename

# Initialize global performance monitor
performance_monitor = PerformanceMonitor()

# Auto-start monitoring
performance_monitor.start_monitoring()

print("🚀 Advanced Performance Monitoring System Initialized")
print("📊 Monitoring CPU, Memory, GPU, and Inference Performance")
print("🔍 Real-time statistics collection active")
