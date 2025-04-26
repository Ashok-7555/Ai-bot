from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

class BaseModel(ABC):
    """
    Abstract base class for all language models in GAKR
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """Load and initialize the model"""
        pass
    
    @abstractmethod
    def generate(self, 
                input_text: str, 
                context: Optional[List[Dict[str, str]]] = None, 
                **kwargs) -> Tuple[str, float]:
        """
        Generate a response to the input text
        
        Args:
            input_text: The text to generate a response for
            context: Optional conversation context
            kwargs: Additional generation parameters
            
        Returns:
            Tuple of (generated text, confidence score)
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        pass
