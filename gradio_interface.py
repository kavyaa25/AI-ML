# Advanced Gradio Interface with Professional UI and Error Handling
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
from typing import Dict, List, Any

# Custom CSS for professional styling
custom_css = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.feature-card {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    border: 1px solid #e1e5e9;
    margin-bottom: 1rem;
}

.metric-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin: 0.5rem;
}

.success-message {
    background: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #c3e6cb;
}

.error-message {
    background: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #f5c6cb;
}

.info-box {
    background: #d1ecf1;
    color: #0c5460;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #bee5eb;
    margin: 1rem 0;
}
"""

# Mock AI Models with Performance Metrics
class AIModelOptimizer:
    def __init__(self):
        self.models = {
            'baseline': {
                'name': 'Baseline ResNet50',
                'avg_time': 45.2,
                'memory': 1200,
                'accuracy': 76.5,
                'description': 'Original unoptimized model'
            },
            'quantized': {
                'name': 'Quantized Model',
                'avg_time': 28.1,
                'memory': 300,
                'accuracy': 75.8,
                'description': 'INT8 quantization optimization'
            },
            'torchscript': {
                'name': 'TorchScript Optimized',
                'avg_time': 32.5,
                'memory': 1100,
                'accuracy': 76.3,
                'description': 'JIT compilation with graph optimization'
            },
            'mixed_precision': {
                'name': 'Mixed Precision',
                'avg_time': 22.8,
                'memory': 800,
                'accuracy': 76.1,
                'description': 'FP16 optimization for modern GPUs'
            },
            'optimized': {
                'name': 'Combined Optimized',
                'avg_time': 18.3,
                'memory': 650,
                'accuracy': 75.5,
                'description': 'Best-of-all optimization techniques'
            }
        }
        self.performance_history = []
    
    def predict_sentiment(self, text, model_type, progress=gr.Progress()):
        """Simulate AI inference with progress tracking"""
        if not text.strip():
            return "Please enter some text to analyze", None, None, None
        
        progress(0, desc="Initializing model...")
        time.sleep(0.5)
        
        model = self.models[model_type]
        
        # Simulate model loading and optimization
        progress(0.2, desc="Loading model weights...")
        time.sleep(0.3)
        
        progress(0.4, desc="Applying optimizations...")
        time.sleep(0.4)
        
        progress(0.6, desc="Running inference...")
        # Simulate inference time based on model
        time.sleep(model['avg_time'] / 1000)
        
        progress(0.8, desc="Post-processing results...")
        time.sleep(0.2)
        
        # Generate prediction
        sentiments = [
            "🟢 Positive sentiment detected with high confidence",
            "🔴 Negative sentiment with moderate confidence", 
            "🟡 Neutral sentiment detected",
            "🟣 Mixed emotions - both positive and negative elements",
            "🟢 Highly positive sentiment with excitement indicators"
        ]
        
        prediction = random.choice(sentiments)
        
        # Calculate metrics with some randomness
        actual_time = model['avg_time'] + random.uniform(-5, 5)
        actual_memory = model['memory'] + random.uniform(-50, 50)
        actual_accuracy = model['accuracy'] + random.uniform(-2, 2)
        
        # Store performance data
        self.performance_history.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'model': model['name'],
            'time': actual_time,
            'memory': actual_memory,
            'accuracy': actual_accuracy
        })
        
        # Keep only last 20 entries
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]
        
        progress(1.0, desc="Complete!")
        
        # Create performance metrics display
        metrics_text = f"""
        **Performance Metrics:**
        - ⏱️ Inference Time: {actual_time:.1f}ms
        - 💾 Memory Usage: {actual_memory:.0f}MB  
        - 🎯 Accuracy: {actual_accuracy:.1f}%
        - 🚀 Throughput: {1000/actual_time:.1f} FPS
        """
        
        # Create comparison chart
        comparison_chart = self.create_comparison_chart()
        performance_chart = self.create_performance_history_chart()
        
        return prediction, metrics_text, comparison_chart, performance_chart
    
    def create_comparison_chart(self):
        """Create model comparison visualization"""
        models_data = []
        for key, model in self.models.items():
            models_data.append({
                'Model': model['name'],
                'Inference Time (ms)': model['avg_time'],
                'Memory (MB)': model['memory'],
                'Accuracy (%)': model['accuracy']
            })
        
        df = pd.DataFrame(models_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Inference Time', 'Memory Usage', 'Accuracy', 'Speed Improvement'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Inference Time
        fig.add_trace(
            go.Bar(x=df['Model'], y=df['Inference Time (ms)'], 
                   name='Inference Time', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Memory Usage  
        fig.add_trace(
            go.Bar(x=df['Model'], y=df['Memory (MB)'], 
                   name='Memory Usage', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Accuracy
        fig.add_trace(
            go.Bar(x=df['Model'], y=df['Accuracy (%)'], 
                   name='Accuracy', marker_color='lightcoral'),
            row=2, col=1
        )
        
        # Speed Improvement
        baseline_time = self.models['baseline']['avg_time']
        improvements = [(baseline_time - model['avg_time'])/baseline_time * 100 
                       for model in self.models.values()]
        
        fig.add_trace(
            go.Bar(x=df['Model'], y=improvements, 
                   name='Speed Improvement (%)', marker_color='gold'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="🚀 AI Model Performance Comparison",
            showlegend=False
        )
        
        return fig
    
    def create_performance_history_chart(self):
        """Create real-time performance monitoring chart"""
        if not self.performance_history:
            return go.Figure().add_annotation(
                text="No performance data yet. Run some predictions!",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font_size=16
            )
        
        df = pd.DataFrame(self.performance_history)
        
        fig = go.Figure()
        
        # Add inference time line
        fig.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['time'],
            mode='lines+markers',
            name='Inference Time (ms)',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="📈 Real-time Performance Monitoring",
            xaxis_title="Time",
            yaxis_title="Inference Time (ms)",
            height=400,
            hovermode='x unified'
        )
        
        return fig

# Initialize optimizer
optimizer = AIModelOptimizer()

# Sample texts for quick testing
sample_texts = [
    "I absolutely love this new AI optimization tool! It's incredibly fast and efficient.",
    "The performance improvements are disappointing. Expected much better results.", 
    "This is a decent tool with some useful features, though it could be improved.",
    "Amazing breakthrough in AI inference optimization! Revolutionary technology.",
    "The interface is confusing and the results are inconsistent."
]

def get_sample_text():
    return random.choice(sample_texts)

# Create Gradio Interface
# with gr.Blocks(
#     theme=gr.themes.Soft(
#         primary_hue="blue",
#         secondary_hue="purple",
#         neutral_hue="slate"
#     ),
#     title="🚀 Interactive AI Optimizer",
#     css="""
#     .gradio-container {
#         max-width: 1200px !important;
#     }
#     .main-header {
#         text-align: center;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 2rem;
#         border-radius: 10px;
#         margin-bottom: 2rem;
#     }
#     """
# ) as demo:
    
#     # Header
#     gr.HTML("""
#     <div class="main-header">
#         <h1>🚀 Interactive AI Optimizer</h1>
#         <p>Experience real-time AI model optimization with performance analysis</p>
#     </div>
#     """)
    
#     with gr.Tabs():
#         # Main Inference Tab
#         with gr.TabItem("🧠 AI Inference", elem_id="inference-tab"):
#             with gr.Row():
#                 with gr.Column(scale=2):
#                     gr.Markdown("### 📝 Input Configuration")
                    
#                     model_choice = gr.Dropdown(
#                         choices=[
#                             ("Baseline ResNet50", "baseline"),
#                             ("Quantized Model", "quantized"), 
#                             ("TorchScript Optimized", "torchscript"),
#                             ("Mixed Precision", "mixed_precision"),
#                             ("Combined Optimized", "optimized")
#                         ],
#                         value="baseline",
#                         label="Select AI Model",
#                         info="Choose the optimization technique to test"
#                     )
                    
#                     with gr.Row():
#                         input_text = gr.Textbox(
#                             label="Input Text for Sentiment Analysis",
#                             placeholder="Enter your text here...",
#                             lines=4,
#                             scale=3
#                         )
#                         sample_btn = gr.Button("🎲 Random Sample", scale=1)
                    
#                     predict_btn = gr.Button(
#                         "🚀 Run AI Inference", 
#                         variant="primary",
#                         size="lg"
#                     )
                
#                 with gr.Column(scale=1):
#                     gr.Markdown("### 📊 Model Information")
#                     model_info = gr.Markdown("""
#                     **Baseline ResNet50**
#                     - Original unoptimized model
#                     - Avg Time: 45.2ms
#                     - Memory: 1200MB
#                     - Accuracy: 76.5%
#                     """)
            
#             # Results Section
#             gr.Markdown("### 🎯 Results & Performance")
            
#             with gr.Row():
#                 with gr.Column():
#                     prediction_output = gr.Textbox(
#                         label="AI Prediction",
#                         lines=2,
#                         interactive=False
#                     )
                    
#                     metrics_output = gr.Markdown(
#                         label="Performance Metrics",
#                         value="Run inference to see metrics..."
#                     )
                
#                 with gr.Column():
#                     performance_plot = gr.Plot(
#                         label="Real-time Performance"
#                     )
        
#         # Performance Analysis Tab
#         with gr.TabItem("📊 Performance Analysis", elem_id="analysis-tab"):
#             gr.Markdown("### 🔍 Comprehensive Model Comparison")
            
#             comparison_plot = gr.Plot(
#                 label="Model Performance Comparison",
#                 value=optimizer.create_comparison_chart()
#             )
            
#             gr.Markdown("""
#             ### 🎯 Key Optimization Insights
            
#             **🏆 Best Performers:**
#             - **Combined Optimized**: 59% faster than baseline
#             - **Mixed Precision**: 50% faster with 33% less memory
#             - **Quantized Model**: 75% memory reduction
            
#             **⚡ Optimization Techniques:**
#             1. **Dynamic Quantization**: INT8 precision for model compression
#             2. **TorchScript**: JIT compilation with graph optimization  
#             3. **Mixed Precision**: FP16 for faster GPU computation
#             4. **Structured Pruning**: Channel-wise model compression
#             5. **Combined Approach**: Best-of-all-worlds optimization
#             """)
        
#         # Advanced Settings Tab
#         with gr.TabItem("⚙️ Advanced Settings", elem_id="settings-tab"):
#             gr.Markdown("### 🔧 Optimization Configuration")
            
#             with gr.Row():
#                 with gr.Column():
#                     gr.Markdown("**Quantization Settings**")
#                     quant_precision = gr.Slider(4, 16, value=8, step=4, label="Bit Precision")
#                     quant_calibration = gr.Checkbox(label="Use Calibration Dataset", value=True)
                
#                 with gr.Column():
#                     gr.Markdown("**Pruning Settings**") 
#                     prune_ratio = gr.Slider(0.0, 0.5, value=0.2, label="Pruning Ratio")
#                     prune_structured = gr.Checkbox(label="Structured Pruning", value=True)
                
#                 with gr.Column():
#                     gr.Markdown("**Compilation Settings**")
#                     jit_optimize = gr.Checkbox(label="JIT Optimization", value=True)
#                     graph_fusion = gr.Checkbox(label="Graph Fusion", value=True)
            
#             optimize_btn = gr.Button("🚀 Apply Custom Optimization", variant="primary")
            
#             gr.Markdown("""
#             ### 💡 Optimization Tips
#             - **Lower bit precision** = smaller model, potentially lower accuracy
#             - **Higher pruning ratio** = faster inference, risk of accuracy loss  
#             - **JIT optimization** = better performance after warmup
#             - **Graph fusion** = reduced memory bandwidth, faster execution
#             """)
    
#     # Event Handlers
#     sample_btn.click(
#         fn=get_sample_text,
#         outputs=input_text
#     )
    
#     def update_model_info(model_type):
#         model = optimizer.models[model_type]
#         return f"""
#         **{model['name']}**
#         - {model['description']}
#         - Avg Time: {model['avg_time']}ms
#         - Memory: {model['memory']}MB  
#         - Accuracy: {model['accuracy']}%
#         """
    
#     model_choice.change(
#         fn=update_model_info,
#         inputs=model_choice,
#         outputs=model_info
#     )
    
#     predict_btn.click(
#         fn=optimizer.predict_sentiment,
#         inputs=[input_text, model_choice],
#         outputs=[prediction_output, metrics_output, comparison_plot, performance_plot]
#     )

# Launch the interface
# if __name__ == "__main__":
#     demo.launch(
#         share=True,
#         server_name="0.0.0.0",
#         server_port=7860,
#         show_error=True,
#         debug=True
#     )
    
# print("🚀 Interactive AI Optimizer launched successfully!")
# print("📊 Features included:")
# print("  ✅ Real-time model optimization")
# print("  ✅ Interactive performance analysis") 
# print("  ✅ Multiple optimization techniques")
# print("  ✅ Live performance monitoring")
# print("  ✅ Professional UI/UX")

class GradioInterface:
    """Advanced Gradio Interface with professional UI and comprehensive features"""
    
    def __init__(self, ml_models):
        self.ml_models = ml_models
        self.usage_stats = {
            'text_classification': 0,
            'sentiment_analysis': 0,
            'image_classification': 0,
            'text_generation': 0
        }
        self.user_feedback = []
    
    def create_interface(self):
        """Create the main Gradio interface"""
        
        with gr.Blocks(
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="purple",
                neutral_hue="slate"
            ),
            css=custom_css,
            title="🚀 Advanced ML Studio"
        ) as demo:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🚀 Advanced ML Studio</h1>
                <p>Professional Machine Learning Interface with Real-time Analytics</p>
                <p>✨ Text Classification • 💭 Sentiment Analysis • 🖼️ Image Recognition • 📝 Text Generation</p>
            </div>
            """)
            
            # Main tabs
            with gr.Tabs():
                
                # Text Classification Tab
                with gr.TabItem("📝 Text Classification", elem_id="text-classification"):
                    self._create_text_classification_tab()
                
                # Sentiment Analysis Tab
                with gr.TabItem("💭 Sentiment Analysis", elem_id="sentiment-analysis"):
                    self._create_sentiment_analysis_tab()
                
                # Image Classification Tab
                with gr.TabItem("🖼️ Image Classification", elem_id="image-classification"):
                    self._create_image_classification_tab()
                
                # Text Generation Tab
                with gr.TabItem("📝 Text Generation", elem_id="text-generation"):
                    self._create_text_generation_tab()
                
                # Analytics Dashboard Tab
                with gr.TabItem("📊 Analytics Dashboard", elem_id="analytics"):
                    self._create_analytics_tab()
                
                # Model Information Tab
                with gr.TabItem("ℹ️ Model Info", elem_id="model-info"):
                    self._create_model_info_tab()
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; padding: 2rem; color: #666; border-top: 1px solid #eee; margin-top: 2rem;">
                <p>🚀 Advanced ML Studio - Built with Gradio • Optimized for Performance • Error-Free Experience</p>
                <p>💡 Tip: Try different inputs and explore all features!</p>
            </div>
            """)
        
        return demo
    
    def _create_text_classification_tab(self):
        """Create text classification interface"""
        gr.HTML("""
        <div class="feature-card">
            <h3>📝 Text Classification</h3>
            <p>Classify text into different categories using state-of-the-art NLP models</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="📝 Enter Text to Classify",
                    placeholder="Type your text here... (e.g., 'This movie is absolutely fantastic!')",
                    lines=4,
                    max_lines=10
                )
                
                with gr.Row():
                    classify_btn = gr.Button("🚀 Classify Text", variant="primary", size="lg")
                    sample_btn = gr.Button("🎲 Try Sample", variant="secondary")
                
                # Sample texts
                sample_texts = [
                    "This product is amazing! I love it so much.",
                    "The service was terrible and disappointing.",
                    "It's an okay product, nothing special.",
                    "Absolutely fantastic experience, highly recommended!",
                    "Not worth the money, very poor quality."
                ]
                
                def get_sample():
                    return np.random.choice(sample_texts)
                
                sample_btn.click(fn=get_sample, outputs=text_input)
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="info-box">
                    <h4>💡 How it works:</h4>
                    <ul>
                        <li>Enter any text in the input box</li>
                        <li>Click "Classify Text" to analyze</li>
                        <li>View prediction and confidence score</li>
                        <li>Try sample texts for quick testing</li>
                    </ul>
                </div>
                """)
        
        # Results section
        with gr.Row():
            with gr.Column():
                classification_result = gr.Textbox(
                    label="🎯 Classification Result",
                    interactive=False
                )
                
                confidence_score = gr.Number(
                    label="📊 Confidence Score",
                    interactive=False
                )
            
            with gr.Column():
                processing_time = gr.Number(
                    label="⏱️ Processing Time (seconds)",
                    interactive=False
                )
                
                error_display = gr.Textbox(
                    label="❌ Error Messages",
                    interactive=False,
                    visible=False
                )
        
        def classify_text_wrapper(text, progress=gr.Progress()):
            """Wrapper function for text classification with progress tracking"""
            self.usage_stats['text_classification'] += 1
            
            result = self.ml_models.predict_text_classification(
                text, 
                progress_callback=progress
            )
            
            if result['error']:
                return (
                    "",  # classification_result
                    0,   # confidence_score
                    0,   # processing_time
                    result['error'],  # error_display
                    gr.update(visible=True)  # error_display visibility
                )
            else:
                return (
                    result['prediction'],
                    round(result['confidence'], 4),
                    round(result['processing_time'], 4),
                    "",  # error_display
                    gr.update(visible=False)  # error_display visibility
                )
        
        classify_btn.click(
            fn=classify_text_wrapper,
            inputs=[text_input],
            outputs=[classification_result, confidence_score, processing_time, error_display, error_display]
        )
    
    def _create_sentiment_analysis_tab(self):
        """Create sentiment analysis interface"""
        gr.HTML("""
        <div class="feature-card">
            <h3>💭 Sentiment Analysis</h3>
            <p>Analyze the emotional tone and sentiment of text using advanced NLP models</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                sentiment_input = gr.Textbox(
                    label="💭 Enter Text for Sentiment Analysis",
                    placeholder="Share your thoughts, reviews, or any text...",
                    lines=4,
                    max_lines=10
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
                    sentiment_sample_btn = gr.Button("🎲 Try Sample", variant="secondary")
                
                # Sentiment sample texts
                sentiment_samples = [
                    "I'm so excited about this new opportunity!",
                    "This is the worst experience I've ever had.",
                    "The weather is nice today, nothing special.",
                    "Absolutely thrilled with the results!",
                    "I'm feeling quite disappointed with the outcome."
                ]
                
                def get_sentiment_sample():
                    return np.random.choice(sentiment_samples)
                
                sentiment_sample_btn.click(fn=get_sentiment_sample, outputs=sentiment_input)
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="info-box">
                    <h4>💡 Sentiment Categories:</h4>
                    <ul>
                        <li>😊 <strong>Positive</strong>: Happy, excited, satisfied</li>
                        <li>😐 <strong>Neutral</strong>: Objective, factual</li>
                        <li>😞 <strong>Negative</strong>: Sad, angry, disappointed</li>
                    </ul>
                </div>
                """)
        
        # Results section
        with gr.Row():
            with gr.Column():
                sentiment_result = gr.Textbox(
                    label="💭 Detected Sentiment",
                    interactive=False
                )
                
                sentiment_confidence = gr.Number(
                    label="📊 Confidence Score",
                    interactive=False
                )
            
            with gr.Column():
                sentiment_time = gr.Number(
                    label="⏱️ Processing Time (seconds)",
                    interactive=False
                )
                
                sentiment_error = gr.Textbox(
                    label="❌ Error Messages",
                    interactive=False,
                    visible=False
                )
        
        def analyze_sentiment_wrapper(text, progress=gr.Progress()):
            """Wrapper function for sentiment analysis"""
            self.usage_stats['sentiment_analysis'] += 1
            
            result = self.ml_models.predict_sentiment_analysis(
                text,
                progress_callback=progress
            )
            
            if result['error']:
                return (
                    "",
                    0,
                    0,
                    result['error'],
                    gr.update(visible=True)
                )
            else:
                # Add emoji based on sentiment
                sentiment_emoji = {
                    'Positive': '😊 Positive',
                    'Negative': '😞 Negative',
                    'Neutral': '😐 Neutral'
                }
                
                display_sentiment = sentiment_emoji.get(result['sentiment'], result['sentiment'])
                
                return (
                    display_sentiment,
                    round(result['confidence'], 4),
                    round(result['processing_time'], 4),
                    "",
                    gr.update(visible=False)
                )
        
        analyze_btn.click(
            fn=analyze_sentiment_wrapper,
            inputs=[sentiment_input],
            outputs=[sentiment_result, sentiment_confidence, sentiment_time, sentiment_error, sentiment_error]
        )
    
    def _create_image_classification_tab(self):
        """Create image classification interface"""
        gr.HTML("""
        <div class="feature-card">
            <h3>🖼️ Image Classification</h3>
            <p>Classify images using state-of-the-art computer vision models</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(
                    label="🖼️ Upload Image for Classification",
                    type="pil",
                    height=300
                )
                
                classify_image_btn = gr.Button("🔍 Classify Image", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="info-box">
                    <h4>💡 Supported formats:</h4>
                    <ul>
                        <li>📸 JPEG, PNG, BMP, TIFF</li>
                        <li>🎨 RGB and grayscale images</li>
                        <li>📏 Any resolution (auto-resized)</li>
                        <li>🔄 Automatic preprocessing</li>
                    </ul>
                </div>
                """)
        
        # Results section
        with gr.Row():
            with gr.Column():
                image_predictions = gr.JSON(
                    label="🎯 Top Predictions",
                    show_label=True
                )
            
            with gr.Column():
                image_time = gr.Number(
                    label="⏱️ Processing Time (seconds)",
                    interactive=False
                )
                
                image_error = gr.Textbox(
                    label="❌ Error Messages",
                    interactive=False,
                    visible=False
                )
        
        def classify_image_wrapper(image, progress=gr.Progress()):
            """Wrapper function for image classification"""
            self.usage_stats['image_classification'] += 1
            
            result = self.ml_models.predict_image_classification(
                image,
                progress_callback=progress
            )
            
            if result['error']:
                return (
                    {},
                    0,
                    result['error'],
                    gr.update(visible=True)
                )
            else:
                # Format predictions for display
                formatted_predictions = {}
                for i, pred in enumerate(result['predictions'], 1):
                    formatted_predictions[f"#{i} {pred['label']}"] = f"{pred['confidence']:.4f}"
                
                return (
                    formatted_predictions,
                    round(result['processing_time'], 4),
                    "",
                    gr.update(visible=False)
                )
        
        classify_image_btn.click(
            fn=classify_image_wrapper,
            inputs=[image_input],
            outputs=[image_predictions, image_time, image_error, image_error]
        )
    
    def _create_text_generation_tab(self):
        """Create text generation interface"""
        gr.HTML("""
        <div class="feature-card">
            <h3>📝 Text Generation</h3>
            <p>Generate creative text using advanced language models</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                generation_prompt = gr.Textbox(
                    label="📝 Enter Text Prompt",
                    placeholder="Start your story, article, or any text...",
                    lines=3,
                    max_lines=5
                )
                
                max_length_slider = gr.Slider(
                    minimum=50,
                    maximum=300,
                    value=100,
                    step=10,
                    label="📏 Maximum Length",
                    info="Maximum number of tokens to generate"
                )
                
                with gr.Row():
                    generate_btn = gr.Button("✨ Generate Text", variant="primary", size="lg")
                    prompt_sample_btn = gr.Button("🎲 Try Sample", variant="secondary")
                
                # Sample prompts
                sample_prompts = [
                    "Once upon a time in a magical forest,",
                    "The future of artificial intelligence is",
                    "In a world where technology has advanced beyond imagination,",
                    "The secret to happiness lies in",
                    "As the sun set over the mountains,"
                ]
                
                def get_prompt_sample():
                    return np.random.choice(sample_prompts)
                
                prompt_sample_btn.click(fn=get_prompt_sample, outputs=generation_prompt)
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="info-box">
                    <h4>💡 Generation Tips:</h4>
                    <ul>
                        <li>🎯 Be specific with your prompt</li>
                        <li>📏 Adjust length for desired output</li>
                        <li>🔄 Try different prompts for variety</li>
                        <li>✨ Creative prompts work best</li>
                    </ul>
                </div>
                """)
        
        # Results section
        with gr.Row():
            with gr.Column():
                generated_text = gr.Textbox(
                    label="✨ Generated Text",
                    lines=8,
                    max_lines=15,
                    interactive=False
                )
            
            with gr.Column():
                generation_time = gr.Number(
                    label="⏱️ Processing Time (seconds)",
                    interactive=False
                )
                
                generation_error = gr.Textbox(
                    label="❌ Error Messages",
                    interactive=False,
                    visible=False
                )
        
        def generate_text_wrapper(prompt, max_length, progress=gr.Progress()):
            """Wrapper function for text generation"""
            self.usage_stats['text_generation'] += 1
            
            result = self.ml_models.generate_text(
                prompt,
                max_length=max_length,
                progress_callback=progress
            )
            
            if result['error']:
                return (
                    "",
                    0,
                    result['error'],
                    gr.update(visible=True)
                )
            else:
                return (
                    result['generated_text'],
                    round(result['processing_time'], 4),
                    "",
                    gr.update(visible=False)
                )
        
        generate_btn.click(
            fn=generate_text_wrapper,
            inputs=[generation_prompt, max_length_slider],
            outputs=[generated_text, generation_time, generation_error, generation_error]
        )
    
    def _create_analytics_tab(self):
        """Create analytics dashboard"""
        gr.HTML("""
        <div class="feature-card">
            <h3>📊 Analytics Dashboard</h3>
            <p>Real-time performance metrics and usage statistics</p>
        </div>
        """)
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Analytics", variant="primary")
        
        with gr.Row():
            usage_stats_display = gr.JSON(
                label="📈 Usage Statistics",
                show_label=True
            )
            
            performance_stats_display = gr.JSON(
                label="⚡ Performance Metrics",
                show_label=True
            )
        
        performance_chart = gr.Plot(
            label="📊 Performance Visualization"
        )
        
        def refresh_analytics():
            """Refresh analytics data"""
            # Usage statistics
            usage_data = {
                "Total Requests": sum(self.usage_stats.values()),
                "Text Classification": self.usage_stats['text_classification'],
                "Sentiment Analysis": self.usage_stats['sentiment_analysis'],
                "Image Classification": self.usage_stats['image_classification'],
                "Text Generation": self.usage_stats['text_generation']
            }
            
            # Performance statistics
            perf_stats = self.ml_models.get_performance_stats()
            
            # Create performance chart
            if perf_stats:
                models = list(perf_stats.keys())
                avg_times = [perf_stats[model]['avg_time'] for model in models]
                
                fig = go.Figure(data=[
                    go.Bar(x=models, y=avg_times, marker_color='lightblue')
                ])
                
                fig.update_layout(
                    title="Average Processing Time by Model",
                    xaxis_title="Model",
                    yaxis_title="Time (seconds)",
                    height=400
                )
            else:
                fig = go.Figure().add_annotation(
                    text="No performance data available yet",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
            
            return usage_data, perf_stats, fig
        
        refresh_btn.click(
            fn=refresh_analytics,
            outputs=[usage_stats_display, performance_stats_display, performance_chart]
        )
        
        # Auto-refresh on load
        refresh_analytics()
    
    def _create_model_info_tab(self):
        """Create model information tab"""
        gr.HTML("""
        <div class="feature-card">
            <h3>ℹ️ Model Information</h3>
            <p>Details about the loaded ML models and system information</p>
        </div>
        """)
        
        model_info_display = gr.JSON(
            label="🤖 Loaded Models",
            value=self.ml_models.get_model_info(),
            show_label=True
        )
        
        gr.HTML("""
        <div class="info-box">
            <h4>🚀 System Features:</h4>
            <ul>
                <li>✅ <strong>Error Handling</strong>: Comprehensive error catching and user-friendly messages</li>
                <li>✅ <strong>Performance Monitoring</strong>: Real-time processing time tracking</li>
                <li>✅ <strong>Progress Tracking</strong>: Visual progress indicators for all operations</li>
                <li>✅ <strong>Fallback Models</strong>: Backup models when primary models fail</li>
                <li>✅ <strong>Responsive UI</strong>: Professional interface with custom styling</li>
                <li>✅ <strong>Analytics Dashboard</strong>: Usage statistics and performance metrics</li>
                <li>✅ <strong>Sample Data</strong>: Quick testing with pre-loaded examples</li>
                <li>✅ <strong>Multi-format Support</strong>: Various input formats and preprocessing</li>
            </ul>
        </div>
        """)
        
        gr.HTML("""
        <div class="feature-card">
            <h4>🔧 Technical Specifications:</h4>
            <ul>
                <li><strong>Text Models</strong>: DistilBERT, RoBERTa, GPT-2</li>
                <li><strong>Image Models</strong>: Vision Transformer (ViT)</li>
                <li><strong>Optimization</strong>: GPU acceleration when available</li>
                <li><strong>Preprocessing</strong>: Automatic image and text preprocessing</li>
                <li><strong>Error Recovery</strong>: Graceful fallback mechanisms</li>
                <li><strong>Performance</strong>: Sub-second inference for most models</li>
            </ul>
        </div>
        """)

# Initialize the interface
print("🎨 Creating Advanced Gradio Interface...")
interface = GradioInterface(ml_models)
demo = interface.create_interface()

# Launch the interface
if __name__ == "__main__":
    print("🚀 Launching Advanced ML Studio...")
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        debug=False,
        favicon_path=None,
        ssl_verify=False
    )
    
print("✅ Advanced ML Studio launched successfully!")
print("🌟 Features: Text Classification, Sentiment Analysis, Image Classification, Text Generation")
print("📊 Analytics: Real-time performance monitoring and usage statistics")
print("🛡️ Reliability: Comprehensive error handling and fallback mechanisms")
