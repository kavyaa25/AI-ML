# Phase 2: Optimization Techniques Implementation
print("\n⚡ Phase 2: Optimization Techniques")
print("=" * 40)

# 1. Dynamic Quantization
print("\n1️⃣ Dynamic Quantization")
print("-" * 30)

class QuantizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(weights='IMAGENET1K_V2')
        self.model = torch.quantization.quantize_dynamic(
            base_model, 
            {nn.Linear, nn.Conv2d}, 
            dtype=torch.qint8
        )
        self.model.eval()
    
    def forward(self, x):
        return self.model(x)

quantized_model = QuantizedModel().to('cpu')  # Quantized models work on CPU
quantized_size = get_model_size(quantized_model)
print(f"✅ Quantized model size: {quantized_size:.2f} MB (Reduction: {((baseline_size - quantized_size) / baseline_size * 100):.1f}%)")

# 2. TorchScript Optimization
print("\n2️⃣ TorchScript Optimization")
print("-" * 30)

class TorchScriptModel(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(weights='IMAGENET1K_V2')
        base_model.eval()
        
        # Convert to TorchScript
        dummy_input = torch.randn(1, 3, 224, 224)
        self.model = torch.jit.trace(base_model, dummy_input)
        self.model = torch.jit.optimize_for_inference(self.model)
    
    def forward(self, x):
        return self.model(x)

torchscript_model = TorchScriptModel().to(device)
torchscript_size = get_model_size(torchscript_model)
print(f"✅ TorchScript model ready, size: {torchscript_size:.2f} MB")

# 3. Mixed Precision Model
print("\n3️⃣ Mixed Precision Model")
print("-" * 30)

class MixedPrecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        self.model.eval()
    
    def forward(self, x):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            return self.model(x)

mixed_precision_model = MixedPrecisionModel().to(device)
print("✅ Mixed precision model ready")

# 4. Pruned Model (Structured Pruning)
print("\n4️⃣ Pruned Model")
print("-" * 30)

import torch.nn.utils.prune as prune

class PrunedModel(nn.Module):
    def __init__(self, pruning_amount=0.2):
        super().__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        
        # Apply structured pruning to conv layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=pruning_amount, n=2, dim=0)
                prune.remove(module, 'weight')
        
        self.model.eval()
    
    def forward(self, x):
        return self.model(x)

pruned_model = PrunedModel().to(device)
pruned_size = get_model_size(pruned_model)
print(f"✅ Pruned model ready, size: {pruned_size:.2f} MB (Reduction: {((baseline_size - pruned_size) / baseline_size * 100):.1f}%)")

# 5. Optimized Model (Combined Techniques)
print("\n5️⃣ Combined Optimization Model")
print("-" * 30)

class OptimizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Start with a pruned model
        base_model = models.resnet50(weights='IMAGENET1K_V2')
        
        # Light pruning
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=0.1, n=2, dim=0)
                prune.remove(module, 'weight')
        
        base_model.eval()
        
        # Convert to TorchScript
        dummy_input = torch.randn(1, 3, 224, 224)
        self.model = torch.jit.trace(base_model, dummy_input)
        self.model = torch.jit.optimize_for_inference(self.model)
    
    def forward(self, x):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            return self.model(x)

optimized_model = OptimizedModel().to(device)
optimized_size = get_model_size(optimized_model)
print(f"✅ Combined optimization model ready, size: {optimized_size:.2f} MB")

print("\n🎯 All optimization models created successfully!")
