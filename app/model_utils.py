# app/model_utils.py
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def safe_load_model(
    model_name: str, 
    device: str = 'cpu', 
    low_cpu_mem_usage: bool = True
) -> Tuple[Optional[pipeline], Optional[str]]:
    """
    Safely load a transformer model handling meta tensor issues.
    
    Args:
        model_name: Name or path of the model to load
        device: Device to load the model on ('cpu' or 'cuda')
        low_cpu_mem_usage: Whether to use low CPU memory usage
        
    Returns:
        Tuple of (classifier pipeline, error message)
    """
    try:
        logger.info(f"Loading model: {model_name} on device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # First, try loading with extra options to handle meta tensors
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                low_cpu_mem_usage=low_cpu_mem_usage,
                torch_dtype=torch.float32  # Use float32 for better compatibility
            )
            
            # Check if model has meta tensors
            if hasattr(model, 'is_meta') and getattr(model, 'is_meta', False):
                logger.info("Model has meta tensors, using to_empty()")
                model = model.to_empty(device)
            else:
                # Regular to() for non-meta models
                model = model.to(device)
                
        except NotImplementedError as e:
            if "Cannot copy out of meta tensor" in str(e):
                # Fallback approach using to_empty()
                logger.info("Meta tensor detected, using alternative loading method")
                from accelerate import init_empty_weights
                
                # Load with empty weights
                with init_empty_weights():
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name, 
                        torch_dtype=torch.float32
                    )
                
                # Move to device using to_empty
                model = model.to_empty(device)
                
                # Load weights separately
                model.load_state_dict(
                    AutoModelForSequenceClassification.from_pretrained(
                        model_name, 
                        torch_dtype=torch.float32
                    ).state_dict()
                )
            else:
                raise e
        
        # Create the pipeline
        classifier = pipeline(
            'text-classification', 
            model=model, 
            tokenizer=tokenizer,
            device=device if device != 'cpu' else -1
        )
        
        logger.info("Model loaded successfully")
        return classifier, None
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logger.error(error_msg)
        return None, error_msg