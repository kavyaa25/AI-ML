# Advanced Performance Optimization System
import torch
import psutil
import GPUtil
import threading
import time
from collections import deque
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Advanced performance optimization and monitoring system"""
    
    def __init__(self):
        self.monitoring_active = False
        self.performance_history = deque(maxlen=1000)
        self.optimization_settings = {
            'auto_gpu_management': True,
            'memory_cleanup': True,
            'batch_optimization': True,
            'cache_management': True
        }
        self.system_stats = {}
        self._initialize_optimizer()
    
    def _initialize_optimizer(self):
        """Initialize performance optimization settings"""
        try:
            # Set optimal PyTorch settings
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                logger.info("✅ CUDA optimizations enabled")
            
            # Set thread settings for CPU
            torch.set_num_threads(min(8, psutil.cpu_count()))
            logger.info(f"✅ CPU threads set to {torch.get_num_threads()}")
            
            # Enable memory optimization
            if hasattr(torch.backends, 'opt_einsum'):
                torch.backends.opt_einsum.enabled = True
            
            logger.info("🚀 Performance optimizer initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Performance optimization warning: {str(e)}")
    
    def start_monitoring(self):
        """Start background performance monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("⏹️ Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                stats = self._collect_system_stats()
                self.performance_history.append(stats)
                self._auto_optimize(stats)
                time.sleep(1)  # Monitor every second
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                time.sleep(5)  # Wait longer on error
    
    def _collect_system_stats(self) -> Dict:
        """Collect comprehensive system statistics"""
        stats = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        }
        
        # GPU statistics
        try:
            if torch.cuda.is_available():
                stats['gpu_memory_used'] = torch.cuda.memory_allocated() / (1024**3)
                stats['gpu_memory_cached'] = torch.cuda.memory_reserved() / (1024**3)
                
                # Get GPU utilization if GPUtil is available
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    stats['gpu_utilization'] = gpu.load * 100
                    stats['gpu_temperature'] = gpu.temperature
        except Exception:
            pass
        
        return stats
    
    def _auto_optimize(self, stats: Dict):
        """Automatic optimization based on system stats"""
        try:
            # Memory cleanup if usage is high
            if stats.get('memory_percent', 0) > 85:
                self._cleanup_memory()
            
            # GPU memory cleanup if needed
            if torch.cuda.is_available() and stats.get('gpu_memory_used', 0) > 0.8:
                self._cleanup_gpu_memory()
            
            # CPU optimization
            if stats.get('cpu_percent', 0) > 90:
                self._optimize_cpu_usage()
                
        except Exception as e:
            logger.error(f"Auto-optimization error: {str(e)}")
    
    def _cleanup_memory(self):
        """Clean up system memory"""
        try:
            import gc
            gc.collect()
            logger.info("🧹 Memory cleanup performed")
        except Exception as e:
            logger.error(f"Memory cleanup error: {str(e)}")
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("🧹 GPU memory cleanup performed")
        except Exception as e:
            logger.error(f"GPU cleanup error: {str(e)}")
    
    def _optimize_cpu_usage(self):
        """Optimize CPU usage"""
        try:
            # Reduce thread count temporarily if CPU is overloaded
            current_threads = torch.get_num_threads()
            if current_threads > 2:
                torch.set_num_threads(max(2, current_threads - 1))
                logger.info(f"🔧 Reduced CPU threads to {torch.get_num_threads()}")
        except Exception as e:
            logger.error(f"CPU optimization error: {str(e)}")
    
    def optimize_model_inference(self, model, input_data):
        """Optimize model inference with various techniques"""
        try:
            # Move to appropriate device
            device = self._get_optimal_device()
            
            if hasattr(model, 'to'):
                model = model.to(device)
            
            # Enable inference mode
            with torch.inference_mode():
                # Use autocast for mixed precision if available
                if device.type == 'cuda' and hasattr(torch, 'autocast'):
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        result = model(input_data)
                else:
                    result = model(input_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Model optimization error: {str(e)}")
            # Fallback to standard inference
            return model(input_data)
    
    def _get_optimal_device(self) -> torch.device:
        """Get the optimal device for computation"""
        if torch.cuda.is_available():
            # Check GPU memory availability
            gpu_memory_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            if gpu_memory_free > 1e9:  # At least 1GB free
                return torch.device('cuda')
        
        return torch.device('cpu')
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        # Convert to numpy arrays for analysis
        recent_stats = list(self.performance_history)[-100:]  # Last 100 entries
        
        cpu_usage = [s.get('cpu_percent', 0) for s in recent_stats]
        memory_usage = [s.get('memory_percent', 0) for s in recent_stats]
        
        report = {
            'monitoring_duration_minutes': len(self.performance_history) / 60,
            'cpu_stats': {
                'average': np.mean(cpu_usage),
                'max': np.max(cpu_usage),
                'min': np.min(cpu_usage),
                'current': cpu_usage[-1] if cpu_usage else 0
            },
            'memory_stats': {
                'average': np.mean(memory_usage),
                'max': np.max(memory_usage),
                'min': np.min(memory_usage),
                'current': memory_usage[-1] if memory_usage else 0
            },
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'cuda_available': torch.cuda.is_available(),
                'torch_threads': torch.get_num_threads()
            }
        }
        
        # Add GPU stats if available
        if torch.cuda.is_available():
            try:
                gpu_stats = [s for s in recent_stats if 'gpu_memory_used' in s]
                if gpu_stats:
                    gpu_memory = [s['gpu_memory_used'] for s in gpu_stats]
                    report['gpu_stats'] = {
                        'average_memory_gb': np.mean(gpu_memory),
                        'max_memory_gb': np.max(gpu_memory),
                        'current_memory_gb': gpu_memory[-1] if gpu_memory else 0,
                        'device_name': torch.cuda.get_device_name(0)
                    }
            except Exception:
                pass
        
        return report
    
    def optimize_batch_processing(self, data_list: List, batch_size: Optional[int] = None) -> List:
        """Optimize batch processing based on system resources"""
        if not data_list:
            return []
        
        # Determine optimal batch size
        if batch_size is None:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            if available_memory_gb > 8:
                batch_size = 32
            elif available_memory_gb > 4:
                batch_size = 16
            elif available_memory_gb > 2:
                batch_size = 8
            else:
                batch_size = 4
        
        # Process in batches
        batches = []
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"🔧 Optimized batch processing: {len(batches)} batches of size {batch_size}")
        return batches
    
    def enable_optimization_features(self, features: Dict[str, bool]):
        """Enable/disable specific optimization features"""
        self.optimization_settings.update(features)
        logger.info(f"🔧 Optimization settings updated: {features}")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current system state"""
        recommendations = []
        
        if not self.performance_history:
            return ["Start performance monitoring to get recommendations"]
        
        recent_stats = list(self.performance_history)[-10:]  # Last 10 entries
        avg_cpu = np.mean([s.get('cpu_percent', 0) for s in recent_stats])
        avg_memory = np.mean([s.get('memory_percent', 0) for s in recent_stats])
        
        if avg_cpu > 80:
            recommendations.append("🔧 High CPU usage detected - consider reducing batch sizes")
        
        if avg_memory > 85:
            recommendations.append("💾 High memory usage - enable automatic memory cleanup")
        
        if torch.cuda.is_available():
            gpu_memory = [s.get('gpu_memory_used', 0) for s in recent_stats if 'gpu_memory_used' in s]
            if gpu_memory and np.mean(gpu_memory) > 0.8:
                recommendations.append("🎮 High GPU memory usage - consider model quantization")
        
        if not recommendations:
            recommendations.append("✅ System performance is optimal")
        
        return recommendations

# Initialize performance optimizer
print("🚀 Initializing Performance Optimizer...")
performance_optimizer = PerformanceOptimizer()
performance_optimizer.start_monitoring()
print("✅ Performance optimization active!")
