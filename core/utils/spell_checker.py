import logging
from typing import List, Dict, Any, Optional

from spellchecker import SpellChecker as PySpellChecker

logger = logging.getLogger(__name__)

class SpellChecker:
    """
    Spell checking and correction utility for GAKR AI
    """
    
    def __init__(self, language: str = 'en'):
        """
        Initialize the spell checker
        
        Args:
            language: Language code for spell checking
        """
        try:
            self.spell_checker = PySpellChecker(language=language)
            self.enabled = True
        except Exception as e:
            logger.error(f"Error initializing spell checker: {e}")
            self.enabled = False
    
    def correct(self, text: str) -> str:
        """
        Correct spelling in the input text
        
        Args:
            text: Text with potential spelling errors
            
        Returns:
            Text with corrected spelling
        """
        if not self.enabled or not text:
            return text
        
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Keep punctuation separate from word
            prefix = ""
            suffix = ""
            
            # Extract leading punctuation
            while word and not word[0].isalnum():
                prefix += word[0]
                word = word[1:]
                
            # Extract trailing punctuation
            while word and not word[-1].isalnum():
                suffix = word[-1] + suffix
                word = word[:-1]
            
            # Only spell check if word has letters
            if word and any(c.isalpha() for c in word):
                # Skip correction for proper nouns (assumed to start with uppercase)
                if word[0].isupper():
                    corrected = word
                else:
                    corrected = self.spell_checker.correction(word)
                    if not corrected:  # If no correction found, keep original
                        corrected = word
            else:
                corrected = word
                
            # Reconstruct with punctuation
            corrected_words.append(prefix + corrected + suffix)
        
        return ' '.join(corrected_words)
    
    def check_text(self, text: str) -> Dict[str, Any]:
        """
        Check text for spelling errors and return detailed info
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with spelling information
        """
        if not self.enabled:
            return {'original': text, 'corrected': text, 'has_errors': False, 'corrections': {}}
        
        words = text.split()
        misspelled = self.spell_checker.unknown(words)
        corrections = {}
        
        for word in misspelled:
            correction = self.spell_checker.correction(word)
            if correction and correction != word:
                corrections[word] = correction
        
        corrected_text = self.correct(text)
        
        return {
            'original': text,
            'corrected': corrected_text,
            'has_errors': len(corrections) > 0,
            'corrections': corrections
        }
