# Phase 3: Comprehensive Benchmarking
print("\n📊 Phase 3: Comprehensive Benchmarking")
print("=" * 40)

models_to_test = [
    ('Baseline', baseline_model, device),
    ('Quantized', quantized_model, 'cpu'),
    ('TorchScript', torchscript_model, device),
    ('Mixed Precision', mixed_precision_model, device),
    ('Pruned', pruned_model, device),
    ('Combined Optimized', optimized_model, device)
]

all_results = []

for name, model, test_device in models_to_test:
    print(f"\n🔍 Testing {name} model...")
    
    try:
        # Benchmark performance
        results = benchmark_model(model, device=test_device)
        
        # Test accuracy
        accuracy = accuracy_test(model, device=test_device)
        
        # Get model size
        model_size = get_model_size(model)
        
        result_dict = {
            'Model': name,
            'Inference_Time_ms': results['avg_time_ms'],
            'Std_Time_ms': results['std_time_ms'],
            'Throughput_FPS': results['throughput_fps'],
            'Memory_MB': results['memory_mb'],
            'Model_Size_MB': model_size,
            'Accuracy_Score': accuracy,
            'Device': test_device
        }
        
        all_results.append(result_dict)
        
        print(f"  ⏱️  Inference: {results['avg_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
        print(f"  🚀 Throughput: {results['throughput_fps']:.1f} FPS")
        print(f"  💾 Memory: {results['memory_mb']:.2f} MB")
        print(f"  📏 Size: {model_size:.2f} MB")
        print(f"  🎯 Accuracy: {accuracy:.3f}")
        
    except Exception as e:
        print(f"  ❌ Error testing {name}: {str(e)}")

# Create comprehensive results DataFrame
results_df = pd.DataFrame(all_results)
print("\n📈 Complete Benchmark Results:")
print("=" * 60)
print(results_df.to_string(index=False, float_format='%.2f'))
