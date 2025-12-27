"""
LinguaBridge NMT Engine
=======================
Neural Machine Translation using Argos Translate (OpenNMT).

Models: ~40-60 MB per language pair
Stored at: backend/storage/voice_models/argos/
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple

from .constants import (
    ARGOS_MODELS_DIR,
    ARGOS_LANGUAGE_PAIRS,
    LANGUAGE_NAMES,
)

logger = logging.getLogger(__name__)


class NMTEngine:
    """
    Neural Machine Translation engine using Argos Translate.
    
    Translates text between languages offline.
    Uses OpenNMT models downloaded on-demand.
    """
    
    def __init__(self):
        self._installed_languages: Optional[List[Tuple[str, str]]] = None
        self._argos_available = False
        self._check_argos_installed()
    
    def _check_argos_installed(self) -> None:
        """Check if argostranslate is installed and available."""
        try:
            import argostranslate.package
            import argostranslate.translate
            self._argos_available = True
            logger.info("✅ Argos Translate is available")
        except ImportError:
            self._argos_available = False
            logger.warning("⚠️ argostranslate not installed. Run: pip install argostranslate")
    
    def _ensure_models_dir(self) -> None:
        """Ensure the Argos models directory exists."""
        ARGOS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    def get_installed_languages(self) -> List[Tuple[str, str]]:
        """
        Get list of installed language pairs.
        
        Returns:
            List of (source_lang, target_lang) tuples
        """
        if not self._argos_available:
            return []
        
        try:
            import argostranslate.translate
            
            installed = []
            languages = argostranslate.translate.get_installed_languages()
            
            for lang in languages:
                # translations_to returns Translation objects for translations TO this language
                # Each Translation has from_lang and to_lang
                target_translations = getattr(lang, 'translations_to', None) or getattr(lang, 'translations', [])
                for trans in target_translations:
                    # Get source and target from the Translation object
                    from_code = getattr(getattr(trans, 'from_lang', None), 'code', None)
                    to_code = getattr(getattr(trans, 'to_lang', None), 'code', None)
                    
                    # Only add valid pairs where source != target
                    if from_code and to_code and from_code != to_code:
                        pair = (from_code, to_code)
                        if pair not in installed:
                            installed.append(pair)
            
            self._installed_languages = installed
            return installed
            
        except Exception as e:
            logger.error(f"Failed to get installed languages: {e}")
            return []
    
    def get_available_packages(self) -> List[dict]:
        """
        Get list of available language packages for download.
        
        Returns:
            List of package info dicts
        """
        if not self._argos_available:
            return []
        
        try:
            import argostranslate.package
            
            # Update package index
            argostranslate.package.update_package_index()
            available = argostranslate.package.get_available_packages()
            
            return [
                {
                    "from_code": pkg.from_code,
                    "from_name": pkg.from_name,
                    "to_code": pkg.to_code,
                    "to_name": pkg.to_name,
                }
                for pkg in available
            ]
            
        except Exception as e:
            logger.error(f"Failed to get available packages: {e}")
            return []
    
    def install_language_pair(self, source_lang: str, target_lang: str) -> bool:
        """
        Download and install a language pair.
        
        Args:
            source_lang: Source language code (e.g., 'en')
            target_lang: Target language code (e.g., 'ja')
        
        Returns:
            True if successful, False otherwise
        
        Stress Tests:
            - Edge Case 1: Network failure during download → return False, don't corrupt state
            - Edge Case 2: Package already installed → skip download, return True
        """
        if not self._argos_available:
            logger.error("Argos Translate not available")
            return False
        
        self._ensure_models_dir()
        
        try:
            import argostranslate.package
            import argostranslate.translate
            
            # Check if already installed
            installed = self.get_installed_languages()
            if (source_lang, target_lang) in installed:
                logger.info(f"Language pair {source_lang}→{target_lang} already installed")
                return True
            
            logger.info(f"Installing language pair: {source_lang}→{target_lang}")
            
            # Update package index
            argostranslate.package.update_package_index()
            available = argostranslate.package.get_available_packages()
            
            # Find matching package
            package = next(
                (pkg for pkg in available 
                 if pkg.from_code == source_lang and pkg.to_code == target_lang),
                None
            )
            
            if package is None:
                logger.error(f"No package found for {source_lang}→{target_lang}")
                return False
            
            # Download and install
            download_path = package.download()
            argostranslate.package.install_from_path(download_path)
            
            logger.info(f"✅ Installed {source_lang}→{target_lang} successfully")
            
            # Refresh installed languages cache
            self._installed_languages = None
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to install {source_lang}→{target_lang}: {e}")
            return False
    
    def install_required_pairs(self) -> dict:
        """
        Install all required language pairs defined in constants.
        
        Returns:
            Dict with installation results for each pair
        """
        results = {}
        for source_lang, target_lang in ARGOS_LANGUAGE_PAIRS:
            pair_key = f"{source_lang}→{target_lang}"
            results[pair_key] = self.install_language_pair(source_lang, target_lang)
        return results
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        auto_install: bool = False,
    ) -> str:
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'en')
            target_lang: Target language code (e.g., 'ja')
            auto_install: If True, automatically install missing language pair
        
        Returns:
            Translated text
        
        Raises:
            RuntimeError: If language pair not installed and auto_install=False
        
        Stress Tests:
            - Edge Case 1: Empty text → return empty string
            - Edge Case 2: Same source and target → return original text
        """
        if not self._argos_available:
            raise RuntimeError("Argos Translate not installed")
        
        # Handle empty text
        if not text or not text.strip():
            return ""
        
        # Same language - no translation needed
        if source_lang == target_lang:
            return text
        
        try:
            import argostranslate.translate
            
            # Check if language pair is installed
            installed = self.get_installed_languages()
            if (source_lang, target_lang) not in installed:
                if auto_install:
                    logger.info(f"Auto-installing {source_lang}→{target_lang}")
                    if not self.install_language_pair(source_lang, target_lang):
                        raise RuntimeError(f"Failed to install {source_lang}→{target_lang}")
                else:
                    raise RuntimeError(
                        f"Language pair {source_lang}→{target_lang} not installed. "
                        f"Call install_language_pair() first."
                    )
            
            # Get translation
            translation = argostranslate.translate.translate(
                text, source_lang, target_lang
            )
            
            return translation
            
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise RuntimeError(f"Translation failed: {e}")
    
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        """
        Translate multiple texts.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            List of translated texts
        """
        return [
            self.translate(text, source_lang, target_lang)
            for text in texts
        ]
    
    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of the given text.
        
        Returns:
            Language code or None if detection fails
        
        Note: Argos doesn't have built-in detection.
        This is a placeholder for future implementation.
        """
        # TODO: Implement language detection
        # Could use langdetect or fasttext
        logger.warning("Language detection not yet implemented")
        return None
    
    def get_language_name(self, code: str) -> str:
        """Get human-readable name for a language code."""
        return LANGUAGE_NAMES.get(code, code.upper())
    
    def smart_translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        use_llm: bool = True,
        use_memory: bool = True,
        use_context: bool = True,
    ) -> Tuple[str, dict]:
        """
        Smart translation with memory → LLM → Argos fallback.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            use_llm: Whether to use LLM for quality (slower)
            use_memory: Whether to check/save translation memory
            use_context: Whether to include context in LLM prompt
            
        Returns:
            (translated_text, metadata_dict)
        """
        # Handle edge cases
        if not text or not text.strip():
            return "", {"source": "empty"}
        
        if source_lang == target_lang:
            return text, {"source": "same_lang"}
        
        metadata = {"source": "unknown", "latency": 0}
        
        import time
        start = time.perf_counter()
        
        # 1. Check exact match first (fastest - <1ms)
        if use_memory:
            try:
                from . import translation_memory
                cached = translation_memory.find_exact(text, source_lang, target_lang)
                if cached:
                    metadata["source"] = "memory_exact"
                    metadata["latency"] = round(time.perf_counter() - start, 3)
                    logger.debug(f"Exact match: '{text[:30]}...'")
                    return cached, metadata
            except Exception as e:
                logger.debug(f"Exact lookup failed: {e}")
        
        # 2. Try vector search for fuzzy match (~50ms)
        if use_memory:
            try:
                from . import translation_memory
                similar = translation_memory.find_similar(text, source_lang, target_lang)
                if similar:
                    matched_src, tgt_text, similarity = similar
                    metadata["source"] = "memory_vector"
                    metadata["similarity"] = similarity
                    metadata["matched"] = matched_src[:50]
                    metadata["latency"] = round(time.perf_counter() - start, 3)
                    logger.debug(f"Vector match ({similarity:.2f}): '{text[:30]}...'")
                    return tgt_text, metadata
            except Exception as e:
                logger.debug(f"Vector search failed: {e}")
        
        # 3. For short phrases (<=4 words), use Argos (fast)
        #    For longer text, use LLM (quality)
        word_count = len(text.split())
        
        if use_llm and word_count > 4:
            try:
                from . import engine_llm
                
                # Build context from recent translations
                context = None
                if use_context and use_memory:
                    try:
                        from . import translation_memory
                        context = translation_memory.build_context(source_lang, target_lang, limit=5)
                    except Exception:
                        pass
                
                # Use LLM
                translation, llm_meta = engine_llm.translate(
                    text, source_lang, target_lang, context=context
                )
                
                metadata["source"] = "llm"
                metadata["model"] = llm_meta.get("model", "unknown")
                metadata["latency"] = round(time.perf_counter() - start, 3)
                
                # Save to memory for future
                if use_memory and translation:
                    try:
                        from . import translation_memory
                        translation_memory.save_translation(
                            text, source_lang, translation, target_lang, source="llm"
                        )
                    except Exception as e:
                        logger.debug(f"Failed to save to memory: {e}")
                
                return translation, metadata
                
            except Exception as e:
                logger.warning(f"LLM translation failed, falling back to Argos: {e}")
        
        # 3. Fallback to Argos (fast, always available)
        try:
            translation = self.translate(text, source_lang, target_lang, auto_install=True)
            metadata["source"] = "argos"
            metadata["latency"] = round(time.perf_counter() - start, 3)
            
            # Save to memory
            if use_memory and translation:
                try:
                    from . import translation_memory
                    translation_memory.save_translation(
                        text, source_lang, translation, target_lang, source="argos"
                    )
                except Exception as e:
                    logger.debug(f"Failed to save to memory: {e}")
            
            return translation, metadata
            
        except Exception as e:
            logger.error(f"All translation methods failed: {e}")
            raise RuntimeError(f"Translation failed: {e}")
    
    def get_model_info(self) -> dict:
        """Return information about installed models."""
        installed = self.get_installed_languages()
        
        # Check LLM availability
        llm_info = {}
        try:
            from . import engine_llm
            llm_info = engine_llm.get_model_info()
        except Exception:
            llm_info = {"available": False}
        
        # Check memory stats
        memory_info = {}
        try:
            from . import translation_memory
            memory_info = translation_memory.get_stats()
        except Exception:
            memory_info = {"total_entries": 0}
        
        return {
            "available": self._argos_available,
            "installed_pairs": [
                f"{src}->{tgt}" for src, tgt in installed
            ],
            "installed_count": len(installed),
            "models_dir": str(ARGOS_MODELS_DIR),
            "llm": llm_info,
            "memory": memory_info,
        }


# Singleton instance for global access
_nmt_engine: Optional[NMTEngine] = None


def get_nmt_engine() -> NMTEngine:
    """Get or create the global NMT engine instance."""
    global _nmt_engine
    if _nmt_engine is None:
        _nmt_engine = NMTEngine()
    return _nmt_engine
