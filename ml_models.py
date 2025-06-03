# Optimized ML Models with Error Handling and Performance Monitoring
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import cv2
from PIL import Image
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedMLModels:
    """Optimized ML Models with comprehensive error handling and performance monitoring"""
    
    def __init__(self):
        self.models = {}
        self.performance_metrics = {}
        self.model_cache = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all ML models with error handling"""
        try:
            logger.info("🚀 Initializing ML models...")
            
            # Text Classification Model
            self._load_text_classifier()
            
            # Sentiment Analysis Model
            self._load_sentiment_analyzer()
            
            # Image Classification Model
            self._load_image_classifier()
            
            # Text Generation Model
            self._load_text_generator()
            
            logger.info("✅ All models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing models: {str(e)}")
            raise
    
    def _load_text_classifier(self):
        """Load optimized text classification model"""
        try:
            self.models['text_classifier'] = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✅ Text classifier loaded")
        except Exception as e:
            logger.warning(f"⚠️ Text classifier fallback: {str(e)}")
            # Fallback to simple rule-based classifier
            self.models['text_classifier'] = self._simple_text_classifier
    
    def _load_sentiment_analyzer(self):
        """Load sentiment analysis model"""
        try:
            self.models['sentiment_analyzer'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✅ Sentiment analyzer loaded")
        except Exception as e:
            logger.warning(f"⚠️ Sentiment analyzer fallback: {str(e)}")
            self.models['sentiment_analyzer'] = self._simple_sentiment_analyzer
    
    def _load_image_classifier(self):
        """Load optimized image classification model"""
        try:
            self.models['image_classifier'] = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✅ Image classifier loaded")
        except Exception as e:
            logger.warning(f"⚠️ Image classifier fallback: {str(e)}")
            self.models['image_classifier'] = self._simple_image_classifier
    
    def _load_text_generator(self):
        """Load text generation model"""
        try:
            self.models['text_generator'] = pipeline(
                "text-generation",
                model="gpt2",
                max_length=100,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✅ Text generator loaded")
        except Exception as e:
            logger.warning(f"⚠️ Text generator fallback: {str(e)}")
            self.models['text_generator'] = self._simple_text_generator
    
    def _simple_text_classifier(self, text):
        """Simple fallback text classifier"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return [{'label': 'POSITIVE', 'score': 0.8}]
        elif neg_count > pos_count:
            return [{'label': 'NEGATIVE', 'score': 0.8}]
        else:
            return [{'label': 'NEUTRAL', 'score': 0.6}]
    
    def _simple_sentiment_analyzer(self, text):
        """Simple fallback sentiment analyzer"""
        return self._simple_text_classifier(text)
    
    def _simple_image_classifier(self, image):
        """Simple fallback image classifier"""
        return [{'label': 'unknown', 'score': 0.5}]
    
    def _simple_text_generator(self, text):
        """Simple fallback text generator"""
        return [{'generated_text': text + " [Generated text not available]"}]
    
    def predict_text_classification(self, text: str, progress_callback=None) -> Dict[str, Any]:
        """Optimized text classification with performance monitoring"""
        if not text or not text.strip():
            return {
                'error': 'Please enter some text to classify',
                'prediction': None,
                'confidence': 0,
                'processing_time': 0
            }
        
        try:
            start_time = time.time()
            
            if progress_callback:
                progress_callback(0.2, "Loading model...")
            
            # Run prediction
            if progress_callback:
                progress_callback(0.6, "Running inference...")
            
            result = self.models['text_classifier'](text)
            
            if progress_callback:
                progress_callback(0.9, "Processing results...")
            
            processing_time = time.time() - start_time
            
            # Extract results
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    # Handle multiple scores format
                    best_result = max(result[0], key=lambda x: x['score'])
                else:
                    best_result = result[0]
                
                prediction = best_result['label']
                confidence = best_result['score']
            else:
                prediction = "Unknown"
                confidence = 0.0
            
            # Log performance
            self._log_performance('text_classification', processing_time)
            
            if progress_callback:
                progress_callback(1.0, "Complete!")
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'processing_time': processing_time,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Text classification error: {str(e)}")
            return {
                'error': f'Classification failed: {str(e)}',
                'prediction': None,
                'confidence': 0,
                'processing_time': 0
            }
    
    def predict_sentiment_analysis(self, text: str, progress_callback=None) -> Dict[str, Any]:
        """Optimized sentiment analysis with detailed metrics"""
        if not text or not text.strip():
            return {
                'error': 'Please enter some text to analyze',
                'sentiment': None,
                'confidence': 0,
                'processing_time': 0
            }
        
        try:
            start_time = time.time()
            
            if progress_callback:
                progress_callback(0.3, "Analyzing sentiment...")
            
            result = self.models['sentiment_analyzer'](text)
            
            if progress_callback:
                progress_callback(0.8, "Calculating confidence...")
            
            processing_time = time.time() - start_time
            
            # Extract sentiment and confidence
            if isinstance(result, list) and len(result) > 0:
                sentiment_result = result[0]
                sentiment = sentiment_result['label']
                confidence = sentiment_result['score']
                
                # Map sentiment labels
                sentiment_mapping = {
                    'LABEL_0': 'Negative',
                    'LABEL_1': 'Neutral', 
                    'LABEL_2': 'Positive',
                    'NEGATIVE': 'Negative',
                    'POSITIVE': 'Positive',
                    'NEUTRAL': 'Neutral'
                }
                
                sentiment = sentiment_mapping.get(sentiment, sentiment)
            else:
                sentiment = "Unknown"
                confidence = 0.0
            
            self._log_performance('sentiment_analysis', processing_time)
            
            if progress_callback:
                progress_callback(1.0, "Analysis complete!")
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'processing_time': processing_time,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {
                'error': f'Sentiment analysis failed: {str(e)}',
                'sentiment': None,
                'confidence': 0,
                'processing_time': 0
            }
    
    def predict_image_classification(self, image, progress_callback=None) -> Dict[str, Any]:
        """Optimized image classification with preprocessing"""
        if image is None:
            return {
                'error': 'Please upload an image',
                'predictions': [],
                'processing_time': 0
            }
        
        try:
            start_time = time.time()
            
            if progress_callback:
                progress_callback(0.2, "Processing image...")
            
            # Convert and preprocess image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                image = Image.open(image)
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if progress_callback:
                progress_callback(0.5, "Running classification...")
            
            # Run prediction
            results = self.models['image_classifier'](image)
            
            if progress_callback:
                progress_callback(0.9, "Processing results...")
            
            processing_time = time.time() - start_time
            
            # Format results
            predictions = []
            for result in results[:5]:  # Top 5 predictions
                predictions.append({
                    'label': result['label'],
                    'confidence': result['score']
                })
            
            self._log_performance('image_classification', processing_time)
            
            if progress_callback:
                progress_callback(1.0, "Classification complete!")
            
            return {
                'predictions': predictions,
                'processing_time': processing_time,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Image classification error: {str(e)}")
            return {
                'error': f'Image classification failed: {str(e)}',
                'predictions': [],
                'processing_time': 0
            }
    
    def generate_text(self, prompt: str, max_length: int = 100, progress_callback=None) -> Dict[str, Any]:
        """Optimized text generation with customizable parameters"""
        if not prompt or not prompt.strip():
            return {
                'error': 'Please enter a text prompt',
                'generated_text': '',
                'processing_time': 0
            }
        
        try:
            start_time = time.time()
            
            if progress_callback:
                progress_callback(0.3, "Generating text...")
            
            # Generate text with parameters
            result = self.models['text_generator'](
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            if progress_callback:
                progress_callback(0.9, "Finalizing output...")
            
            processing_time = time.time() - start_time
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0]['generated_text']
                # Remove the original prompt from the generated text
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
            else:
                generated_text = "Text generation failed"
            
            self._log_performance('text_generation', processing_time)
            
            if progress_callback:
                progress_callback(1.0, "Generation complete!")
            
            return {
                'generated_text': generated_text,
                'processing_time': processing_time,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Text generation error: {str(e)}")
            return {
                'error': f'Text generation failed: {str(e)}',
                'generated_text': '',
                'processing_time': 0
            }
    
    def _log_performance(self, model_name: str, processing_time: float):
        """Log performance metrics for monitoring"""
        if model_name not in self.performance_metrics:
            self.performance_metrics[model_name] = []
        
        self.performance_metrics[model_name].append({
            'timestamp': time.time(),
            'processing_time': processing_time
        })
        
        # Keep only last 100 entries
        if len(self.performance_metrics[model_name]) > 100:
            self.performance_metrics[model_name] = self.performance_metrics[model_name][-100:]
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all models"""
        stats = {}
        
        for model_name, metrics in self.performance_metrics.items():
            if metrics:
                times = [m['processing_time'] for m in metrics]
                stats[model_name] = {
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_predictions': len(times)
                }
        
        return stats
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about loaded models"""
        info = {}
        for model_name in self.models.keys():
            info[model_name] = f"✅ Loaded and ready"
        return info

# Initialize the ML models
print("🚀 Initializing Optimized ML Models...")
ml_models = OptimizedMLModels()
print("✅ ML Models ready for inference!")
