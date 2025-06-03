# Phase 4: Performance Analysis and Visualization
print("\n📊 Phase 4: Performance Analysis & Visualization")
print("=" * 50)

# Calculate improvements
baseline_time = results_df[results_df['Model'] == 'Baseline']['Inference_Time_ms'].iloc[0]
baseline_size = results_df[results_df['Model'] == 'Baseline']['Model_Size_MB'].iloc[0]
baseline_memory = results_df[results_df['Model'] == 'Baseline']['Memory_MB'].iloc[0]

results_df['Speed_Improvement_%'] = ((baseline_time - results_df['Inference_Time_ms']) / baseline_time * 100)
results_df['Size_Reduction_%'] = ((baseline_size - results_df['Model_Size_MB']) / baseline_size * 100)
results_df['Memory_Reduction_%'] = ((baseline_memory - results_df['Memory_MB']) / baseline_memory * 100)

# Create visualizations
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('🚀 AI Inference Optimization Results', fontsize=16, fontweight='bold')

# 1. Inference Time Comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(results_df['Model'], results_df['Inference_Time_ms'], 
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
ax1.set_title('⏱️ Inference Time Comparison', fontweight='bold')
ax1.set_ylabel('Time (ms)')
ax1.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}ms', ha='center', va='bottom', fontweight='bold')

# 2. Model Size Comparison
ax2 = axes[0, 1]
bars2 = ax2.bar(results_df['Model'], results_df['Model_Size_MB'],
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
ax2.set_title('📏 Model Size Comparison', fontweight='bold')
ax2.set_ylabel('Size (MB)')
ax2.tick_params(axis='x', rotation=45)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}MB', ha='center', va='bottom', fontweight='bold')

# 3. Throughput Comparison
ax3 = axes[1, 0]
bars3 = ax3.bar(results_df['Model'], results_df['Throughput_FPS'],
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
ax3.set_title('🚀 Throughput Comparison', fontweight='bold')
ax3.set_ylabel('FPS')
ax3.tick_params(axis='x', rotation=45)

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# 4. Speed Improvement Percentage
ax4 = axes[1, 1]
colors = ['red' if x < 0 else 'green' for x in results_df['Speed_Improvement_%']]
bars4 = ax4.bar(results_df['Model'], results_df['Speed_Improvement_%'], color=colors, alpha=0.7)
ax4.set_title('📈 Speed Improvement (%)', fontweight='bold')
ax4.set_ylabel('Improvement (%)')
ax4.tick_params(axis='x', rotation=45)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)

for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
             f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

plt.tight_layout()
plt.show()

# Performance Summary Table
print("\n🏆 Optimization Summary:")
print("=" * 80)
summary_df = results_df[['Model', 'Inference_Time_ms', 'Speed_Improvement_%', 
                        'Model_Size_MB', 'Size_Reduction_%', 'Accuracy_Score']].copy()
print(summary_df.to_string(index=False, float_format='%.2f'))
