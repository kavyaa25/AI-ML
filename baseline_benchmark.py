# Phase 1: Baseline Benchmarking
print("\n📈 Phase 1: Baseline Benchmarking")
print("=" * 40)

# Benchmark baseline model
print("Running baseline benchmarks...")
baseline_results = benchmark_model(baseline_model, device=device)

print(f"⏱️  Average inference time: {baseline_results['avg_time_ms']:.2f} ± {baseline_results['std_time_ms']:.2f} ms")
print(f"🚀 Throughput: {baseline_results['throughput_fps']:.1f} FPS")
print(f"💾 GPU Memory usage: {baseline_results['memory_mb']:.2f} MB")

# Test accuracy
baseline_accuracy = accuracy_test(baseline_model, device=device)
print(f"🎯 Baseline diversity score: {baseline_accuracy:.3f}")

# Store baseline results
results_df = pd.DataFrame({
    'Model': ['Baseline'],
    'Inference_Time_ms': [baseline_results['avg_time_ms']],
    'Throughput_FPS': [baseline_results['throughput_fps']],
    'Memory_MB': [baseline_results['memory_mb']],
    'Model_Size_MB': [baseline_size],
    'Accuracy_Score': [baseline_accuracy]
})

print("\n📊 Baseline Results Summary:")
print(results_df.to_string(index=False))
