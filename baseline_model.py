# Phase 1: Baseline Implementation
print("\n📊 Phase 1: Baseline Implementation")
print("=" * 40)

class BaselineModel(nn.Module):
    """Original ResNet50 model for baseline comparison"""
    def __init__(self, num_classes=1000):
        super(BaselineModel, self).__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        self.model.eval()
        
    def forward(self, x):
        return self.model(x)

# Initialize baseline model
baseline_model = BaselineModel().to(device)
print(f"✅ Baseline model loaded: {sum(p.numel() for p in baseline_model.parameters()):,} parameters")

# Model size calculation
def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

baseline_size = get_model_size(baseline_model)
print(f"📏 Baseline model size: {baseline_size:.2f} MB")
