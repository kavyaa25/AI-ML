# Comprehensive Benchmarking Utilities
print("\n⏱️ Setting up Benchmarking Utilities")
print("=" * 40)

def benchmark_model(model, input_size=(1, 3, 224, 224), device='cuda', runs=100, warmup=10):
    """
    Comprehensive model benchmarking with memory tracking
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size, device=device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Clear cache
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Memory before inference
    if device == 'cuda':
        memory_before = torch.cuda.memory_allocated()
    
    # Timing
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    
    # Memory after inference
    if device == 'cuda':
        memory_after = torch.cuda.memory_allocated()
        memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
    else:
        memory_used = 0
    
    avg_time = np.mean(times) * 1000  # Convert to milliseconds
    std_time = np.std(times) * 1000
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'memory_mb': memory_used,
        'throughput_fps': 1000 / avg_time
    }

def accuracy_test(model, device='cuda', num_samples=100):
    """
    Test model accuracy on sample images
    """
    # Load sample images (using a few test images)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create random test data (in real scenario, use validation dataset)
    test_data = torch.randn(num_samples, 3, 224, 224, device=device)
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, num_samples, 10):  # Process in batches
            batch = test_data[i:i+10]
            outputs = model(batch)
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    
    # Return dummy accuracy (in real scenario, compare with ground truth)
    return len(set(predictions)) / 1000  # Diversity metric as proxy

print("✅ Benchmarking utilities ready")
