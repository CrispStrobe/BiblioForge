#!/usr/bin/env python3
"""
Cross-platform PDF, EPUB, DJVU, and MOBI text extraction tool

Installation:
1. Python packages (all platforms):
   pip install pymupdf pdfplumber pypdf pdfminer.six pytesseract pdf2image kraken easyocr paddleocr python-doctr ocrmypdf camelot-py opencv-python importlib-metadata tqdm

   Additional format dependencies:
   pip install ebooklib beautifulsoup4 html2text mobi djvu kindleunpack 

2. System dependencies:
   Windows:
   - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
   - Ghostscript: https://ghostscript.com/releases/gsdnld.html
   - Poppler: https://github.com/oschwartz10612/poppler-windows/releases/
   - DjVuLibre: https://sourceforge.net/projects/djvu/files/DjVuLibre_Windows/
   - Calibre: https://calibre-ebook.com/download

   macOS:
   brew install tesseract poppler ghostscript djvulibre calibre

   Linux:
   sudo apt-get install tesseract-ocr poppler-utils ghostscript djvulibre-bin calibre
"""
import warnings
# Suppress common warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')

import os
import re
import sys
import logging
import argparse
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
import textwrap
import pkg_resources
import traceback
from contextlib import contextmanager
import shutil
import platform
import subprocess
import tempfile
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import signal
import threading
from datetime import datetime
from types import MappingProxyType
import time

# Thread-local storage for LLM clients
thread_local = threading.local()

# Limit concurrent connections to Ollama - prevents overwhelming the server
ollama_semaphore = threading.Semaphore(2)  # Allow only 2 concurrent connections
llm_semaphore = threading.Semaphore(2)

# Global lock for thread-safe file operations
file_lock = threading.Lock()

# Define a shutdown flag for graceful termination
shutdown_flag = threading.Event()

# Constants for LLM model
#MODEL_NAME = "cas/spaetzle-v85-7b" 
MODEL_NAME = "cas/llama-3.2-3b-instruct:latest"
# Can also use "cas/llama-3.1-8b-instruct" or other Ollama models


# Import OpenAI client for Ollama
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI client library is not installed. Please install it using 'pip install openai'.")
    print("Required for --sort functionality")

class timeout:
    """Context manager for timeout"""
    def __init__(self, seconds):
        self.seconds = seconds
        self.timer = None

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(f"Operation timed out after {self.seconds} seconds")

    def __enter__(self):
        if self.seconds > 0:
            self.timer = signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        if self.seconds > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.timer)
            
class ProgressProxy:
    """Proxy for tqdm progress bar that can update description with engine info"""
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar
        self.current_engine = None
        
    def update(self, n: int = 1, engine: Optional[str] = None):
        """Update progress and optionally show current engine"""
        if engine and engine != self.current_engine:
            self.current_engine = engine
            self.progress_bar.set_description(f"Extracting text [{engine}]")
        self.progress_bar.update(n)


class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """
        Send chat completion request to the LLM provider
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            timeout: Timeout in seconds
            
        Returns:
            Dict with response content
        """
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Either pass as api_key or set OPENAI_API_KEY environment variable")
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            if not hasattr(thread_local, "openai_client"):
                thread_local.openai_client = OpenAI(api_key=self.api_key)
            return thread_local.openai_client
        except ImportError:
            raise ImportError("OpenAI client library is required. Install with 'pip install openai'")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """Send chat completion request to OpenAI"""
        client = self._init_client()
        
        with llm_semaphore:
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                
                # Convert response to a standard format
                return {
                    "id": getattr(response, "id", "unknown"),
                    "content": response.choices[0].message.content.strip(),
                    "finish_reason": response.choices[0].finish_reason,
                    "model": self.model_name
                }
            except Exception as e:
                logging.error(f"OpenAI request failed: {str(e)}")
                raise

class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API provider"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        # HF can work without API key for some models
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """Send chat completion request to HuggingFace Inference API"""
        try:
            from huggingface_hub import InferenceClient
            
            # Format messages into HF-compatible prompt
            prompt = self._format_messages_for_hf(messages)
            
            # Initialize client with or without token
            client = InferenceClient(token=self.api_key) if self.api_key else InferenceClient()
            
            with llm_semaphore:
                response = client.text_generation(
                    prompt,
                    model=self.model_name,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    do_sample=True,
                    timeout=timeout
                )
                
                return {
                    "id": "hf-inference",
                    "content": str(response).strip(),
                    "finish_reason": "stop",
                    "model": self.model_name
                }
                
        except Exception as e:
            logging.error(f"HuggingFace inference error: {str(e)}")
            raise
    
    def _format_messages_for_hf(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages for HuggingFace text generation"""
        formatted_prompt = ""
        
        # Add system message if present
        system_msgs = [msg for msg in messages if msg["role"] == "system"]
        if system_msgs:
            formatted_prompt = f"<s>[INST] {system_msgs[0]['content']} [/INST]</s>\n\n"
        
        # Add user/assistant messages
        for msg in [m for m in messages if m["role"] != "system"]:
            if msg["role"] == "user":
                formatted_prompt += f"<s>[INST] {msg['content']} [/INST]"
            elif msg["role"] == "assistant":
                formatted_prompt += f" {msg['content']}</s>\n"
        
        # Handle last message if it's from user
        if messages[-1]["role"] == "user":
            formatted_prompt += "</s>"
            
        return formatted_prompt

class CohereProvider(LLMProvider):
    """Cohere API provider"""
    
    def __init__(self, model_name: str = "command-r-plus", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key required. Either pass as api_key or set COHERE_API_KEY environment variable")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """Send chat completion request to Cohere"""
        try:
            import cohere
            
            with llm_semaphore:
                try:
                    # Try ClientV2 first
                    client = cohere.ClientV2(self.api_key)
                    cohere_messages = self._format_messages_for_cohere(messages)
                    
                    response = client.chat(
                        model=self.model_name,
                        messages=cohere_messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    return {
                        "id": "cohere-v2",
                        "content": response.message.content[0].text,
                        "finish_reason": "stop",
                        "model": self.model_name
                    }
                    
                except Exception as e:
                    logging.warning(f"Cohere V2 failed, trying V1: {e}")
                    
                    # Fallback to V1
                    client = cohere.Client(self.api_key)
                    # Get the last user message
                    last_user_msg = next((msg["content"] for msg in reversed(messages) 
                                        if msg["role"] == "user"), "")
                    
                    response = client.chat(
                        message=last_user_msg,
                        model=self.model_name,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    return {
                        "id": "cohere-v1",
                        "content": response.text,
                        "finish_reason": "stop",
                        "model": self.model_name
                    }
                    
        except ImportError:
            raise ImportError("Cohere client library is required. Install with 'pip install cohere'")
        except Exception as e:
            logging.error(f"Cohere request failed: {str(e)}")
            raise
    
    def _format_messages_for_cohere(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for Cohere API"""
        cohere_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                cohere_messages.append({"role": "SYSTEM", "content": msg["content"]})
            elif msg["role"] == "user":
                cohere_messages.append({"role": "USER", "content": msg["content"]})
            elif msg["role"] == "assistant":
                cohere_messages.append({"role": "CHATBOT", "content": msg["content"]})
                
        return cohere_messages

class GLHFProvider(LLMProvider):
    """GLHF API provider (uses OpenAI-compatible API)"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("GLHF_API_KEY")
        if not self.api_key:
            raise ValueError("GLHF API key required. Either pass as api_key or set GLHF_API_KEY environment variable")
        self._init_client()
    
    def _init_client(self):
        """Initialize GLHF client (using OpenAI client with custom base URL)"""
        try:
            from openai import OpenAI
            if not hasattr(thread_local, "glhf_client"):
                thread_local.glhf_client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://glhf.chat/api/openai/v1"
                )
            return thread_local.glhf_client
        except ImportError:
            raise ImportError("OpenAI client library is required for GLHF. Install with 'pip install openai'")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """Send chat completion request to GLHF"""
        client = self._init_client()
        
        # Format model id for GLHF
        model_id = f"hf:{self.model_name}" if not self.model_name.startswith("hf:") else self.model_name
        
        with llm_semaphore:
            try:
                # GLHF works best with streaming
                response_chunks = []
                
                completion = client.chat.completions.create(
                    stream=True,
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        response_chunks.append(chunk.choices[0].delta.content)
                        
                full_response = "".join(response_chunks)
                
                return {
                    "id": "glhf",
                    "content": full_response,
                    "finish_reason": "stop",
                    "model": self.model_name
                }
                
            except Exception as e:
                logging.error(f"GLHF request failed: {str(e)}")
                raise

class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local LLMs"""
    
    def __init__(self, model_name: str = "cas/llama-3.2-3b-instruct:latest", 
                base_url: str = "http://localhost:11434/v1/"):
        super().__init__(model_name)
        self.base_url = base_url
        # Use OpenAI client with Ollama
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client for Ollama"""
        try:
            from openai import OpenAI
            if not hasattr(thread_local, "ollama_client"):
                thread_local.ollama_client = OpenAI(
                    base_url=self.base_url,
                    api_key="ollama"  # Placeholder value for Ollama
                )
            return thread_local.ollama_client
        except ImportError:
            raise ImportError("OpenAI client library is required. Install with 'pip install openai'")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """Send chat completion request to Ollama"""
        client = self._init_client()
        
        with llm_semaphore:
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=messages,
                    timeout=timeout
                )
                
                # Convert response to a standard format
                return {
                    "id": getattr(response, "id", "unknown"),
                    "content": response.choices[0].message.content.strip(),
                    "finish_reason": response.choices[0].finish_reason,
                    "model": self.model_name
                }
            except Exception as e:
                logging.error(f"Ollama request failed: {str(e)}")
                raise


class GroqProvider(LLMProvider):
    """Groq LLM provider for cloud LLMs"""
    
    def __init__(self, model_name: str = "llama3-70b-8192", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required. Either pass as api_key or set GROQ_API_KEY environment variable")
        self._init_client()
    
    def _init_client(self):
        """Initialize Groq client"""
        try:
            from groq import Groq
            if not hasattr(thread_local, "groq_client"):
                thread_local.groq_client = Groq(api_key=self.api_key)
            return thread_local.groq_client
        except ImportError:
            raise ImportError("Groq client library is required. Install with 'pip install groq'")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """Send chat completion request to Groq"""
        client = self._init_client()
        
        with llm_semaphore:
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                
                # Convert response to a standard format
                return {
                    "id": getattr(response, "id", "unknown"),
                    "content": response.choices[0].message.content.strip(),
                    "finish_reason": response.choices[0].finish_reason,
                    "model": self.model_name
                }
            except Exception as e:
                logging.error(f"Groq request failed: {str(e)}")
                raise


class PoeProvider(LLMProvider):
    """Poe.com LLM provider"""
    
    def __init__(self, model_name: str = "claude-3-opus", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("POE_API_KEY")
        if not self.api_key:
            raise ValueError("Poe API key required. Either pass as api_key or set POE_API_KEY environment variable")
        self.api_url = "https://api.poe.com/api/chat"
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7, 
                       max_tokens: int = 500,
                       timeout: int = 120) -> Dict[str, Any]:
        """Send chat completion request to Poe.com"""
        
        # Format messages for Poe API
        formatted_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                # Poe doesn't support system messages directly,
                # prepend to first user message instead
                continue
            formatted_messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        # Add system message to the first user message if present
        system_messages = [msg for msg in messages if msg['role'] == 'system']
        if system_messages and formatted_messages:
            for user_msg in formatted_messages:
                if user_msg['role'] == 'user':
                    user_msg['content'] = f"[System Instructions: {system_messages[0]['content']}]\n\n{user_msg['content']}"
                    break
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        with llm_semaphore:
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()
                response_data = response.json()
                
                # Extract and return content in a standard format
                return {
                    "id": response_data.get("id", "unknown"),
                    "content": response_data.get("choices", [{"message": {"content": ""}}])[0]["message"]["content"].strip(),
                    "finish_reason": response_data.get("choices", [{"finish_reason": "unknown"}])[0]["finish_reason"],
                    "model": self.model_name
                }
            except requests.RequestException as e:
                logging.error(f"Poe request failed: {str(e)}")
                raise


def get_llm_provider(provider_type: str = "ollama", 
                    model_name: Optional[str] = None,
                    api_key: Optional[str] = None) -> LLMProvider:
    """
    Factory function to get an LLM provider based on type
    """
    provider_type = provider_type.lower()
    
    # Default model names for each provider
    default_models = {
        "ollama": "cas/spaetzle-v85-7b",
        "groq": "llama-3.1-70b-versatile",
        "openai": "gpt-3.5-turbo",
        "cohere": "command-r-plus",
        "huggingface": "mistralai/Mistral-7B-Instruct-v0.3",
        "glhf": "mistralai/Mistral-7B-Instruct-v0.3"
    }
    
    # Use default model if none specified
    if not model_name:
        model_name = default_models.get(provider_type, default_models["ollama"])
    
    if provider_type == "ollama":
        return OllamaProvider(model_name=model_name)
    elif provider_type == "groq":
        return GroqProvider(model_name=model_name, api_key=api_key)
    elif provider_type == "openai":
        return OpenAIProvider(model_name=model_name, api_key=api_key)
    elif provider_type == "cohere":
        return CohereProvider(model_name=model_name, api_key=api_key)
    elif provider_type == "huggingface":
        return HuggingFaceProvider(model_name=model_name, api_key=api_key)
    elif provider_type == "glhf":
        return GLHFProvider(model_name=model_name, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


def send_to_llm(text: str, filename: str, provider: Union[str, LLMProvider], 
              model_name: Optional[str] = None,
              api_key: Optional[str] = None,
              max_attempts: int = 5,
              verbose: bool = False) -> str:
    """
    Extract metadata from text using an LLM provider
    
    Args:
        text: Text to analyze
        filename: Filename for context
        provider: Provider type string ('ollama', 'groq', 'poe') or LLMProvider instance
        model_name: Optional model name to use
        api_key: Optional API key for cloud providers
        max_attempts: Maximum retry attempts
        verbose: Whether to print debug information
        
    Returns:
        str: The formatted metadata response
    """
    if isinstance(provider, (int, float)):
        logging.error(f"Invalid provider parameter: {provider} (type: {type(provider)}). Expected provider string or LLMProvider instance.")
        return ""
        
    # Get or validate provider
    if isinstance(provider, str):
        try:
            llm_provider = get_llm_provider(provider, model_name, api_key)
        except ImportError as e:
            logging.error(f"Failed to initialize {provider} provider: {str(e)}")
            # Fall back to Ollama if available
            try:
                llm_provider = get_llm_provider("ollama")
                logging.info(f"Falling back to Ollama provider")
            except ImportError:
                raise ImportError("No LLM providers available. Install at least one client library.")
    else:
        llm_provider = provider
    
    base_retry_wait = 2  # Base wait time in seconds
    
    # Prepare different prompt templates to try if earlier ones fail
    prompt_templates = [
        # First attempt - simple structured format
        (
            f"Extract the author name (lastname surname) of the main author (ignore other authors), "
            f"year of publication, title, and language from the following text, considering the filename '{os.path.basename(filename)}' "
            f"which may contain clues. I need the output **only** in the following format with no additional text or explanations: \n"
            f"<TITLE>The publication title</TITLE>\n<YEAR>2023</YEAR>\n<AUTHOR>Lastname Firstname</AUTHOR>\n<LANGUAGE>en</LANGUAGE>\n\n"
        ),
        # Second attempt - emphasize exact format
        (
            f"I need to extract metadata from a document. Please give me ONLY these four tags with the information, and nothing else:\n"
            f"<TITLE>The exact title</TITLE>\n<YEAR>The publication year (4 digits)</YEAR>\n<AUTHOR>The author's name in 'Lastname Firstname' format</AUTHOR>\n<LANGUAGE>The language code</LANGUAGE>\n\n"
            f"Document filename: {os.path.basename(filename)}\n"
        ),
        # Third attempt - even more explicit
        (
            f"You are a metadata extraction tool. Extract these fields from the text:\n"
            f"1. TITLE (the full publication title)\n"
            f"2. YEAR (the 4-digit publication year, use 'Unknown' if not found)\n"
            f"3. AUTHOR (format as 'Lastname Firstname', preserve any commas)\n"
            f"4. LANGUAGE (the 2-letter language code, e.g., 'en', 'de', 'fr')\n\n"
            f"Format your response EXACTLY like this with no other text:\n"
            f"<TITLE>The title</TITLE>\n<YEAR>2023</YEAR>\n<AUTHOR>Smith John</AUTHOR>\n<LANGUAGE>en</LANGUAGE>\n\n"
        )
    ]
    
    # Try different prompt templates if we encounter format issues
    attempt = 1
    while attempt <= max_attempts:
        # Choose prompt template based on attempt number
        template_index = min(attempt - 1, len(prompt_templates) - 1)
        prompt_template = prompt_templates[template_index]
        
        logging.debug(f"Consulting LLM {provider} on file: {filename} (Attempt: {attempt}, Template: {template_index + 1})")


        # Build the final prompt with text sample
        prompt = prompt_template + f"Here is the document text:\n{text[:3000]}"  # Limit text to avoid token limits
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = llm_provider.chat_completion(
                messages=messages,
                temperature=0.5,  # Reduced temperature for more consistent formatting
                max_tokens=250,
                timeout=120  # 2 minute timeout
            )
            
            output = response["content"]
            if verbose:
                logging.debug(f"Metadata content received from LLM: {output}")
            
            # Validate the response has the expected format
            metadata = parse_metadata(output, verbose=verbose)
            if metadata:
                return output
            else:
                logging.warning(f"Unexpected response format from LLM: {output}")
                # Less aggressive backoff for format issues
                if attempt < max_attempts:
                    wait_time = base_retry_wait * (1.5 ** (attempt - 1))  # Gentler exponential backoff
                    logging.info(f"Retrying with different prompt in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                return output
            
        except Exception as e:
            if "rate_limit" in str(e).lower() or "timeout" in str(e).lower():
                # Use exponential backoff for rate limiting/timeouts
                wait_time = base_retry_wait * (2 ** (attempt - 1))  # Exponential backoff
                logging.info(f"Rate limit or timeout encountered. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                attempt += 1
                continue
            else:
                logging.error(f"Error communicating with LLM for {filename}: {e}")
                if attempt < max_attempts:
                    wait_time = base_retry_wait * (1.5 ** (attempt - 1))  # Gentler exponential backoff
                    logging.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
            return ""
            
    logging.error(f"Maximum retry attempts reached for LLM request.")
    return ""

class ImportCache:
    """Global cache for imports and their availability"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._modules = {}
            cls._instance._available = {}
        return cls._instance
    
    def is_available(self, module_name: str, submodules: List[str] = None) -> bool:
        """Check if module and optional submodules can be imported"""
        # Import importlib here to ensure it's available
        import importlib.util
        
        cache_key = f"{module_name}:{','.join(submodules or [])}"
        if cache_key not in self._available:
            try:
                # Check main module
                if importlib.util.find_spec(module_name) is None:
                    self._available[cache_key] = False
                    return False
                # Check submodules if specified
                if submodules:
                    for submodule in submodules:
                        full_name = f"{module_name}.{submodule}"
                        if importlib.util.find_spec(full_name) is None:
                            self._available[cache_key] = False
                            return False
                self._available[cache_key] = True
            except Exception:
                self._available[cache_key] = False
        return self._available[cache_key]

    def import_module(self, module_name: str, submodule: str = None) -> Any:
        """Import and cache a module"""
        # Import importlib here to ensure it's available
        import importlib
        
        cache_key = f"{module_name}{f'.{submodule}' if submodule else ''}"
        if cache_key not in self._modules:
            try:
                if submodule:
                    main_module = importlib.import_module(module_name)
                    self._modules[cache_key] = getattr(main_module, submodule)
                else:
                    self._modules[cache_key] = importlib.import_module(module_name)
            except ImportError as e:
                raise ImportError(f"Failed to import {cache_key}: {e}")
        return self._modules[cache_key]

class ExtractionManager:
    """Central manager for text extraction operations"""
    
    # Define supported file extensions
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'PDF',
        '.epub': 'EPUB',
        '.djvu': 'DJVU',
        '.djv': 'DJVU',
        '.mobi': 'MOBI',
        '.azw': 'MOBI',
        '.azw3': 'MOBI',
        '.txt': 'Text',
        '.text': 'Text',
        '.md': 'Text',
        '.html': 'HTML',
        '.htm': 'HTML',
        '.xhtml': 'HTML',
        '.docx': 'Text',  # Use Text extractor but with Calibre as method
        '.doc': 'Text',   # Use Text extractor but with Calibre as method
        '.rtf': 'Text',   # Use Text extractor but with Calibre as method
        '.fb2': 'Text',   # Use Text extractor but with Calibre as method
        '.pdb': 'Text',   # Use Text extractor but with Calibre as method
        '.lit': 'Text',   # Use Text extractor but with Calibre as method
        '.odt': 'Text',   # Use Text extractor but with Calibre as method
        '.lrf': 'Text',   # Use Text extractor but with Calibre as method
        '.cbz': 'Text',   # Use Text extractor but with Calibre as method
        '.cbr': 'Text',   # Use Text extractor but with Calibre as method
        '.azw3': 'MOBI',  # Use MOBI extractor but with Calibre as method
        '.azw4': 'MOBI',  # Use MOBI extractor but with Calibre as method
        '.chm': 'Text',   # Use Text extractor but with Calibre as method
        '.snb': 'Text',   # Use Text extractor but with Calibre as method
        '.tcr': 'Text',   # Use Text extractor but with Calibre as method
    }

    def __init__(self, debug: bool = False):
        self._debug = debug
        self._setup_logging(debug)
        # Use optimized binary detection that works across all platforms
        self._binary_paths, self._binaries = self._check_system_dependencies()
        
        self._versions = self._check_versions()
        self._extractors = {}

        # Share binary paths with extractors
        self._shared_binary_paths = self._binary_paths
        
    def _setup_logging(self, debug: bool):
        """Configure logging with appropriate level and handlers"""
        level = logging.DEBUG if debug else logging.INFO
        logging.getLogger().setLevel(level)

        # Suppress common warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Suppress detailed logs from specific libraries
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('pdf2image').setLevel(logging.WARNING)
        logging.getLogger('pytesseract').setLevel(logging.WARNING)
        logging.getLogger('pdfminer').setLevel(logging.WARNING)
        logging.getLogger('pypdf').setLevel(logging.WARNING)
        logging.getLogger('camelot').setLevel(logging.WARNING)
        logging.getLogger('pymupdf').setLevel(logging.WARNING)
        
        # Additional debug logging suppression for PyPDF
        if not debug:
            logging.getLogger('pypdf').setLevel(logging.ERROR)
        else:
            # Even in debug mode, limit some excessive loggers
            logging.getLogger('pypdf.filters').setLevel(logging.INFO)
            logging.getLogger('pypdf.xref').setLevel(logging.INFO)
            logging.getLogger('pypdf.generic').setLevel(logging.INFO)

    def integrate_binary_paths(binary_paths, extractor):
        """
        Ensure the extractor has all necessary binary paths from the manager.
        
        Args:
            binary_paths: Dictionary of binary paths from ExtractionManager
            extractor: The extractor instance to update
        """
        # Make sure the extractor has a _binary_paths attribute
        if not hasattr(extractor, '_binary_paths'):
            extractor._binary_paths = {}
        
        # Update the extractor's binary paths with the manager's paths
        for binary, path in binary_paths.items():
            extractor._binary_paths[binary] = path
        
        # Update the extractor's binaries boolean dict if it has one
        if hasattr(extractor, '_binaries'):
            extractor._binaries = {
                'tesseract': bool(binary_paths.get('tesseract')),
                'poppler': bool(binary_paths.get('pdftoppm')),
                'ghostscript': bool(binary_paths.get('gs')),
                'djvulibre': bool(binary_paths.get('djvutxt')) or bool(binary_paths.get('ddjvu')),
                'calibre': bool(binary_paths.get('ebook-converter'))
            }


    def _check_system_dependencies(self) -> Tuple[Dict[str, str], Dict[str, bool]]:
        """
        Optimized system dependencies check with correct version flags for each binary.
        Returns both the full paths and a boolean compatibility dict.
        """
        # Dictionary to store binary paths (not just boolean values)
        binary_paths = {
            'tesseract': None,       # OCR engine
            'pdftoppm': None,        # PDF to image conversion (poppler)
            'gs': None,              # Ghostscript
            'djvutxt': None,         # For DJVU text extraction
            'ddjvu': None,           # For DJVU conversion
            'ebook-converter': None  # Calibre converter
        }
        
        # Executable name variations by platform and binary
        executable_names = {
            'tesseract': ['tesseract', 'tesseract.exe'],
            'pdftoppm': ['pdftoppm', 'pdftoppm.exe'],
            'gs': ['gs', 'gswin64c', 'gswin64c.exe', 'gswin32c.exe'],
            'djvutxt': ['djvutxt', 'djvutxt.exe'],
            'ddjvu': ['ddjvu', 'ddjvu.exe'],
            'ebook-converter': ['ebook-converter', 'ebook-convert', 'ebook-converter.exe', 'ebook-convert.exe']
        }
        
        # Version check flags for different binaries
        version_flags = {
            'tesseract': '--version',
            'pdftoppm': '-v',        # pdftoppm uses -v instead of --version
            'gs': '--version',
            'djvutxt': '--help',     # DjVuLibre tools don't have version flags
            'ddjvu': '--help',       # Use help instead
            'ebook-converter': '--version'
        }
        
        # Common installation directories by platform
        system = platform.system()
        common_dirs = {}
        
        if system == 'Windows':
            common_dirs = {
                'tesseract': [
                    r'C:\Program Files\Tesseract-OCR',
                    r'C:\Program Files (x86)\Tesseract-OCR'
                ],
                'pdftoppm': [
                    r'C:\Program Files\poppler-24.08.0\Library\bin',
                    r'C:\Program Files\poppler-24.02.0\Library\bin',
                    r'C:\Program Files\poppler\bin',
                    r'C:\Program Files (x86)\poppler\bin',
                    r'C:\poppler\bin'
                ],
                'gs': [
                    r'C:\Program Files\gs\gs10.04.0\bin',
                    r'C:\Program Files\gs\gs10.02.0\bin',
                    r'C:\Program Files (x86)\gs\gs10.04.0\bin',
                    r'C:\Program Files (x86)\gs\gs10.02.0\bin',
                    r'C:\Program Files\gs\bin'
                ],
                'ebook-converter': [
                    r'C:\Program Files\Calibre2',
                    r'C:\Program Files (x86)\Calibre2',
                    r'C:\Program Files\Calibre',
                    r'C:\Program Files (x86)\Calibre'
                ],
                'djvutxt': [
                    r'C:\Program Files\DjVuLibre',
                    r'C:\Program Files (x86)\DjVuLibre'
                ],
                'ddjvu': [
                    r'C:\Program Files\DjVuLibre',
                    r'C:\Program Files (x86)\DjVuLibre'
                ]
            }
        elif system == 'Darwin':  # macOS
            common_dirs = {
                'tesseract': [
                    '/usr/local/bin',
                    '/opt/homebrew/bin',  # Apple Silicon homebrew
                    '/opt/local/bin'      # MacPorts
                ],
                'pdftoppm': [
                    '/usr/local/bin',
                    '/opt/homebrew/bin',
                    '/opt/local/bin'
                ],
                'gs': [
                    '/usr/local/bin',
                    '/opt/homebrew/bin',
                    '/opt/local/bin'
                ],
                'ebook-converter': [
                    '/Applications/calibre.app/Contents/MacOS',
                    '/opt/homebrew/bin',
                    '/usr/local/bin',
                    '~/bin'  # User's bin directory
                ],
                'djvutxt': [
                    '/usr/local/bin',
                    '/opt/homebrew/bin',
                    '/opt/local/bin'
                ],
                'ddjvu': [
                    '/usr/local/bin',
                    '/opt/homebrew/bin',
                    '/opt/local/bin'
                ]
            }
        else:  # Linux and others
            common_dirs = {
                'tesseract': [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/opt/bin'
                ],
                'pdftoppm': [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/opt/bin'
                ],
                'gs': [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/opt/bin'
                ],
                'ebook-converter': [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/opt/bin',
                    '~/bin'  # User's bin directory
                ],
                'djvutxt': [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/opt/bin'
                ],
                'ddjvu': [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/opt/bin'
                ]
            }
        
        # Expand home directories in paths
        for binary in common_dirs:
            expanded_paths = []
            for path in common_dirs[binary]:
                if '~' in path:
                    expanded_paths.append(os.path.expanduser(path))
                else:
                    expanded_paths.append(path)
            common_dirs[binary] = expanded_paths
        
        # First check PATH for each binary
        for binary, names in executable_names.items():
            for name in names:
                path = shutil.which(name)
                if path:
                    binary_paths[binary] = path
                    if self._debug:
                        logging.debug(f"Found {binary} in PATH: {path}")
                    break
        
        # For binaries not found in PATH, check common directories
        for binary, path in binary_paths.items():
            if path is None and binary in common_dirs:
                for directory in common_dirs[binary]:
                    if not os.path.exists(directory):
                        continue
                        
                    for name in executable_names[binary]:
                        full_path = os.path.join(directory, name)
                        if os.path.exists(full_path) and os.access(full_path, os.X_OK):
                            binary_paths[binary] = full_path
                            if self._debug:
                                logging.debug(f"Found {binary} in common directory: {full_path}")
                            break
                    if binary_paths[binary]:  # Stop checking if found
                        break
        
        # Verify binary versions if found
        for binary, path in binary_paths.items():
            if path:
                # Skip verification for Calibre on Mac as it might not be executable directly
                if binary == 'ebook-converter' and system == 'Darwin' and '/Applications/calibre.app' in path:
                    if self._debug:
                        logging.debug(f"Skipping version check for {binary} in Mac app bundle")
                    continue
                    
                # Get the appropriate version flag
                version_flag = version_flags.get(binary, '--version')
                try:
                    # For binaries that might error on version check but still work
                    try:
                        result = subprocess.run(
                            [path, version_flag],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=2  # Prevent hanging
                        )
                        
                        # Extract version info from output
                        if result.stdout:
                            version_info = result.stdout.splitlines()[0]
                        elif result.stderr:
                            version_info = result.stderr.splitlines()[0]
                        else:
                            version_info = f"Available (no version info)"
                        
                        if self._debug:
                            logging.debug(f"Verified {binary}: {version_info}")
                    except subprocess.CalledProcessError as e:
                        # For binaries that return non-zero but might still work
                        if self._debug:
                            logging.debug(f"Version check for {binary} returned non-zero: {e.returncode}")
                            if e.stdout:
                                logging.debug(f"  stdout: {e.stdout.splitlines()[0] if e.stdout else ''}")
                            if e.stderr:
                                logging.debug(f"  stderr: {e.stderr.splitlines()[0] if e.stderr else ''}")
                        # Consider it available despite error
                        if self._debug:
                            logging.debug(f"Considering {binary} available despite version check failure")
                except Exception as e:
                    if self._debug:
                        logging.debug(f"Failed to verify {binary} version: {e}")
        
        # Log missing binaries with download information
        download_info = {
            'tesseract': "Tesseract not found. Download from: https://github.com/UB-Mannheim/tesseract/wiki",
            'pdftoppm': "Poppler (pdftoppm) not found. Download from: https://github.com/oschwartz10612/poppler-windows/releases/",
            'gs': "Ghostscript not found. Download from: https://ghostscript.com/releases/gsdnld.html",
            'ebook-converter': "Calibre not found. Download from: https://calibre-ebook.com/download",
            'djvutxt': "DjVuLibre not found. For Windows: https://sourceforge.net/projects/djvu/files/DjVuLibre_Windows/",
            'ddjvu': "DjVuLibre (ddjvu) not found. For Windows: https://sourceforge.net/projects/djvu/files/DjVuLibre_Windows/"
        }
        
        # Only show warnings for missing binaries once
        warned_about = getattr(self, '_warned_about_binaries', set())
        for binary, path in binary_paths.items():
            if not path and binary in download_info and binary not in warned_about:
                logging.info(download_info[binary])
                warned_about.add(binary)
        
        # Store which binaries we've warned about
        self._warned_about_binaries = warned_about
        
        # Create backward-compatible boolean results dictionary
        binaries_bool = {
            'tesseract': bool(binary_paths['tesseract']),
            'poppler': bool(binary_paths['pdftoppm']),
            'ghostscript': bool(binary_paths['gs']),
            'djvulibre': bool(binary_paths['djvutxt']) or bool(binary_paths['ddjvu']),
            'calibre': bool(binary_paths['ebook-converter'])
        }
        
        return binary_paths, binaries_bool

    def get_binary_path(self, binary_name: str) -> Optional[str]:
        """
        Get the full path to a binary if available
        
        Args:
            binary_name: Name of the binary
            
        Returns:
            str or None: Path to the binary or None if not found
        """
        # Map common names to the actual binary keys
        binary_map = {
            'tesseract': 'tesseract',
            'pdftoppm': 'pdftoppm',
            'poppler': 'pdftoppm',
            'gs': 'gs',
            'ghostscript': 'gs',
            'djvutxt': 'djvutxt',
            'ddjvu': 'ddjvu',
            'djvulibre': 'djvutxt',  # Default to djvutxt for djvulibre
            'ebook-converter': 'ebook-converter',
            'ebook-convert': 'ebook-converter',
            'calibre': 'ebook-converter'
        }
        
        # Get the actual binary key
        binary_key = binary_map.get(binary_name, binary_name)
        
        # Return the path from the stored binary paths
        return self._binary_paths.get(binary_key)

    def _check_versions(self) -> Dict[str, str]:
        """Get versions of installed Python packages"""
        versions = {}
        packages = [
            'pymupdf', 'pdfplumber', 'pypdf', 'pdfminer.six',
            'pytesseract', 'pdf2image', 'easyocr',
            'paddleocr', 'python-doctr', 'ocrmypdf', 'camelot-py',
            'ebooklib', 'beautifulsoup4', 'html2text', 'kraken',
            'djvu', 'mobi', 'kindleunpack',
            'chardet', 'ftfy', 'lxml'
        ]
        
        for package in packages:
            try:
                version = pkg_resources.get_distribution(package).version
                versions[package] = version
                logging.debug(f"Found {package} version {version}")
            except Exception as e:
                logging.debug(f"Package {package} not found: {e}")
        
        return versions

    def _get_extractor(self, file_path: str) -> Union['PDFExtractor', 'EPUBExtractor', 'DJVUExtractor', 'MOBIExtractor', 'TextExtractor', 'HTMLExtractor']:
        """Get or create appropriate extractor for file type"""
        file_ext = os.path.splitext(file_path)[1].lower()
        cache_key = f"{file_ext}:{file_path}"
        
        if cache_key not in self._extractors:
            import_cache = ImportCache()  # Create an import cache instance for all extractors
            
            # Get the extractor type from supported extensions
            extractor_type = self.SUPPORTED_EXTENSIONS.get(file_ext)
            
            if not extractor_type:
                raise ValueError(f"Unsupported file type: {file_path} (extension: {file_ext})")
                
            # Create appropriate extractor
            if extractor_type == 'PDF':
                # Pass the binary paths to the PDF extractor
                self._extractors[cache_key] = PDFExtractor(
                    debug=self._debug, 
                    binary_paths=self._binary_paths  # Pass binary paths to avoid duplication
                )
            elif extractor_type == 'EPUB':
                self._extractors[cache_key] = EPUBExtractor(
                    import_cache=import_cache, 
                    debug=self._debug,
                    binary_paths=self._binary_paths
                )
            elif extractor_type == 'DJVU':
                self._extractors[cache_key] = DJVUExtractor(
                    import_cache=import_cache, 
                    debug=self._debug,
                    binary_paths=self._binary_paths
                )
            elif extractor_type == 'MOBI':
                self._extractors[cache_key] = MOBIExtractor(
                    import_cache=import_cache, 
                    debug=self._debug,
                    binary_paths=self._binary_paths
                )
            elif extractor_type == 'Text':
                self._extractors[cache_key] = TextExtractor(
                    import_cache=import_cache, 
                    debug=self._debug,
                    binary_paths=self._binary_paths
                )
            elif extractor_type == 'HTML':
                self._extractors[cache_key] = HTMLExtractor(
                    import_cache=import_cache, 
                    debug=self._debug,
                    binary_paths=self._binary_paths
                )
        
        return self._extractors[cache_key]
    
    def extract(self, input_path: str, 
        output_path: Optional[str] = None,
        method: Optional[str] = None,
        ocr_method: Optional[str] = None,
        password: Optional[str] = None,
        extract_tables: bool = False,
        force_ocr: bool = False,
        sort: bool = False,     # with callback for sort processing
        llm_provider = None,      
        rename_script_path: Optional[str] = None,
        **kwargs) -> Union[str, bool]:
        """
        Extract text from a document with progress reporting and error handling
        
        Args:
            input_path: Path to input document
            output_path: Optional path for output text file
            method: Preferred extraction method
            ocr_method: Optional OCR method (only passed to PDF extractors)
            password: Password for encrypted documents
            extract_tables: Whether to extract tables (PDF only)
            force_ocr: Whether to force OCR even if text layer exists
            sort: Whether to sort files based on content
            llm_provider: Provider for LLM communication
            rename_script_path: Path to write rename commands
            **kwargs: Additional extraction options
            
        Returns:
            Extracted text if output_path is None, else success boolean
        """
        try:
            # Check if file type is supported
            file_ext = os.path.splitext(input_path)[1].lower()
            if file_ext not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {input_path} (extension: {file_ext})")
                
            # Get appropriate extractor
            extractor = self._get_extractor(input_path)
            
            # Configure extraction
            if password and hasattr(extractor, 'set_password'):
                extractor.set_password(password)
            
            # Log extraction details clearly
            logging.debug(f"Extracting text from: {input_path}")
            if method:
                logging.debug(f"Preferred method: {method}")
            if ocr_method:
                logging.debug(f"OCR method: {ocr_method}")
            if force_ocr:
                logging.debug("Force OCR mode enabled")
                
            # Create a progress bar with explicit total and careful setup
            pbar = tqdm(
                total=100,  # Set an arbitrary total of 100 units
                desc=f"Extracting text",
                disable=False,
                unit='%',
                position=0,
                leave=True,
                ncols=100  # Fixed width to avoid display issues
            )
                
            # Safe progress callback that won't cause errors
            def safe_progress_callback(n=1, engine=None):
                try:
                    # Only update description if engine is provided and changed
                    if engine is not None:
                        pbar.set_description(f"Extracting text [{engine}]")
                    
                    # Safe update with consistent increment
                    pbar.update(1)
                except:
                    # Suppress all errors in progress updates
                    pass
            
            # Extract text with the safe progress callback
            try:
                # Prepare extraction parameters
                extraction_kwargs = kwargs.copy()
                
                # Only pass specific parameters to PDF extractors
                if isinstance(extractor, PDFExtractor):
                    if ocr_method:
                        extraction_kwargs['ocr_method'] = ocr_method
                    extraction_kwargs['force_ocr'] = force_ocr
                    if extract_tables:
                        extraction_kwargs['extract_tables'] = extract_tables
                
                # Perform the actual extraction
                text = extractor.extract_text(
                    input_path,
                    preferred_method=method,
                    progress_callback=safe_progress_callback,
                    **extraction_kwargs
                )
            except Exception as e:
                logging.error(f"Extraction failed: {str(e)}")
                raise
            finally:
                # Always close the progress bar
                try:
                    pbar.close()
                except:
                    pass
                    
            # Check if we got valid text
            if not text or not text.strip():
                logging.warning(f"No text extracted from {input_path}")
                return False if output_path else ""
                
            # Validate text quality
            if not self._validate_text(text):
                logging.warning("Extracted text may be of low quality")
            else:
                if self._debug:
                    logging.info(f"Successfully extracted text ({len(text)} characters)")
            
            # Handle sorting if requested
            if sort and llm_provider and rename_script_path and text:
                try:
                    logging.info("Extracting metadata for sorting...")
                    metadata_content = extract_metadata(text, input_path, llm_provider)
                    
                    if metadata_content:
                        metadata = parse_metadata(metadata_content)
                        if metadata:
                            # Process author names
                            author = metadata['author']
                            logging.debug(f"Extracted author: {author}")
                            corrected_author = sort_author_names(
                                author_names=author,
                                provider=llm_provider
                            )
                            logging.debug(f"Corrected author: {corrected_author}")
                            metadata['author'] = corrected_author
                            
                            # Get file details
                            title = metadata['title']
                            year = metadata['year']
                            if not year or year == "Unknown":
                                year = "UnknownYear"
                                
                            if not author or not title:
                                logging.warning(f"Missing author or title for {input_path}. Skipping rename.")
                            else:
                                # Create target paths
                                first_author = sanitize_filename(corrected_author)

                                # Use the output directory as the base for author directories
                                base_dir = os.path.dirname(output_path) if output_path else os.path.abspath('.')
                                target_dir = os.path.join(base_dir, first_author)
                                
                                file_extension = os.path.splitext(input_path)[1].lower()
                                new_filename = f"{year} {sanitize_filename(title)}{file_extension}"
                                logging.info(f"File will be renamed to: {os.path.join(first_author, new_filename)}")

                                # Add rename command
                                output_dir = os.path.dirname(output_path) if output_path else None
                                add_rename_command(
                                    rename_script_path, 
                                    input_path, 
                                    target_dir, 
                                    new_filename, 
                                    output_dir=output_dir
                                )
                        else:
                            logging.warning(f"Failed to parse metadata for {input_path}")
                    else:
                        logging.warning(f"Failed to get metadata from LLM provider for {input_path}")
                except Exception as sort_e:
                    logging.error(f"Error sorting file {input_path}: {sort_e}")
                        
            # Write output file if path provided
            if output_path:
                # Make sure the directory exists
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                logging.info(f"Writing text file: {output_path}")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                return True
            return text
            
        except Exception as e:
            error_context = self._recover_from_error(e, "extraction")
            if error_context:
                logging.error(f"Extraction failed: {error_context}")
            else:
                logging.error(f"Extraction failed: {str(e)}")
                if self._debug:
                    traceback.print_exc()
            return False if output_path else ""

    @contextmanager
    def _progress_context(self, message: str):
        """Context manager for progress reporting"""
        progress = tqdm(
            desc=message,
            disable=not self._debug,
            unit='pages'
        )
        try:
            yield progress
        finally:
            progress.close()

    def _validate_text(self, text: str, min_length: int = 50) -> bool:
        """Validate extracted text quality"""
        if not text or len(text.strip()) < min_length:
            return False
            
        # Check for garbage content
        garbage_ratio = sum(1 for c in text if not c.isprintable()) / len(text)
        if garbage_ratio > 0.1:
            return False
            
        # Check line lengths
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return False
            
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        if avg_line_length < 20:
            return False
            
        return True

    def _recover_from_error(self, error: Exception, context: str = "") -> Optional[str]:
        """Try to recover from extraction errors"""
        error_str = str(error)
        
        if isinstance(error, MemoryError):
            self._clear_memory()
            return "Memory error occurred - cleared memory"
            
        if "PDF file is encrypted" in error_str:
            return "PDF is encrypted - try providing a password"
            
        if "PDF file is damaged" in error_str:
            return "PDF file appears to be damaged"
            
        if "no text extractable" in error_str.lower():
            return "No extractable text found - try OCR"
            
        if "not enough memory" in error_str.lower():
            self._clear_memory()
            return "Memory allocation failed - try reducing batch size"
            
        if "permission error" in error_str.lower():
            return "Permission denied - check file access"
            
        if "timeout" in error_str.lower():
            return "Operation timed out - try again"
            
        return None

    def _clear_memory(self):
        """Clear memory and cached data"""
        import gc
        gc.collect()
        
        # Clear GPU memory if available
        if self._check_gpu_available():
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except:
                    try:
                        import paddle
                        paddle.device.cuda.empty_cache()
                    except:
                        pass

    def _check_gpu_available(self) -> bool:
        """Enhanced GPU availability check"""
        try:
            import torch
            if torch.cuda.is_available():
                # Check if CUDA initialization works
                try:
                    torch.cuda.init()
                    device = torch.cuda.current_device()
                    capability = torch.cuda.get_device_capability(device)
                    logging.info(f"CUDA device available: {torch.cuda.get_device_name(device)} "
                            f"(Compute {capability[0]}.{capability[1]})")
                    return True
                except Exception as e:
                    logging.warning(f"CUDA available but initialization failed: {e}")
                    return False
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logging.info("MPS (Metal Performance Shaders) device available")
                return True
                
            logging.warning("No GPU detected - using CPU only")
            return False
        except ImportError:
            logging.warning("PyTorch not available - using CPU only")
            return False

class MOBIExtractor:
    """MOBI text extraction with multiple fallback methods"""
    
    def __init__(self, import_cache: ImportCache, debug: bool = False, binary_paths=None):
        self._import_cache = import_cache
        self._debug = debug
        self._available_methods = None
        self._binary_paths = binary_paths or {}
        self._kindleunpack_script = None

        # Check for kindleunpack specifically
        kindleunpack_type, kindleunpack_path = self.find_kindleunpack()
        self._kindleunpack_type = kindleunpack_type
        self._kindleunpack_path = kindleunpack_path
        
        if kindleunpack_type == 'script':
            self._kindleunpack_script = kindleunpack_path
            
        if self._debug and kindleunpack_type:
            logging.debug(f"Found kindleunpack as {kindleunpack_type} at: {kindleunpack_path}")

    @property
    def available_methods(self) -> Dict[str, bool]:
        """Lazy load available methods"""
        if self._available_methods is None:
            self._available_methods = {
                'mobi': self._import_cache.is_available('mobi'),
                'kindleunpack': self._kindleunpack_type is not None,  # Use our direct detection
                'calibre': self._check_calibre_available(),
                'zipfile': True  # Basic fallback always available
            }
        return self._available_methods

    def _check_calibre_available(self):
        """Check if Calibre converter is available"""
        try:
            for bin_name in ['ebook-converter', 'ebook-convert']:
                if shutil.which(bin_name):
                    return True
            return False
        except:
            return False

    def extract_text(self, mobi_path: str, preferred_method: Optional[str] = None,
                    progress_callback: Optional[Callable] = None, **kwargs) -> str:
        """
        Extract text from MOBI with fallback methods
        
        Args:
            mobi_path: Path to MOBI file
            preferred_method: Optional preferred extraction method
            progress_callback: Optional callback for progress updates
            **kwargs: Additional options (ignored)
            
        Returns:
            Extracted text
        """
        methods = ['mobi', 'kindleunpack', 'calibre', 'zipfile']
        
        # Reorder methods if preferred method is specified
        if preferred_method and preferred_method in methods:
            methods.insert(0, methods.pop(methods.index(preferred_method)))

        text = ""
        with tqdm(total=len(methods), desc="Trying MOBI extraction methods", unit="method") as method_pbar:
            for method in methods:
                if not self.available_methods.get(method, False):
                    method_pbar.update(1)
                    continue
                    
                try:
                    if progress_callback:
                        progress_callback(0, f"mobi_{method}")  # Signal start with method name
                    
                    extraction_func = getattr(self, f'extract_with_{method}')
                    text = extraction_func(
                        mobi_path,
                        lambda n: progress_callback(n, f"mobi_{method}") if progress_callback else None
                    )
                    
                    if text and text.strip():
                        method_pbar.update(1)
                        break
                        
                except Exception as e:
                    logging.debug(f"Error with MOBI {method}: {e}")
                    
                method_pbar.update(1)

        return text.strip()

    def extract_with_mobi(self, mobi_path: str, progress_callback=None) -> str:
        """Extract text using mobi Python library"""
        if not self._import_cache.is_available('mobi'):
            return ""
            
        try:
            mobi = self._import_cache.import_module('mobi')
            
            # The mobi Python package provides direct MOBI parsing
            tempdir = None
            try:
                # Extract the MOBI file
                with tempfile.TemporaryDirectory() as tempdir:
                    book = mobi.Mobi(mobi_path)
                    book.parse()
                    
                    # Extract the content
                    text_parts = []
                    
                    # Process raw html content
                    if hasattr(book, 'raw_html') and book.raw_html:
                        # Try to process HTML with BeautifulSoup
                        if self._import_cache.is_available('bs4'):
                            BeautifulSoup = self._import_cache.import_module('bs4').BeautifulSoup
                            
                            soup = BeautifulSoup(book.raw_html, 'html.parser')
                            # Remove script and style elements
                            for script in soup(["script", "style"]):
                                script.extract()
                            
                            # Get text
                            text = soup.get_text()
                            lines = (line.strip() for line in text.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            text = '\n'.join(chunk for chunk in chunks if chunk)
                            text_parts.append(text)
                        else:
                            # Basic HTML cleaning if BeautifulSoup not available
                            import re
                            text = re.sub(r'<[^>]+>', ' ', book.raw_html)
                            text = re.sub(r'\s+', ' ', text).strip()
                            text_parts.append(text)
                    
                    # Alternative: try to get book contents
                    if hasattr(book, 'book_header') and book.book_header:
                        if hasattr(book.book_header, 'title') and book.book_header.title:
                            text_parts.insert(0, f"Title: {book.book_header.title}\n")
                        
                        if hasattr(book.book_header, 'author') and book.book_header.author:
                            text_parts.insert(1, f"Author: {book.book_header.author}\n")
                    
                    if progress_callback:
                        progress_callback(1)
                    
                    return "\n\n".join(text_parts)
            finally:
                if tempdir and os.path.exists(tempdir):
                    try:
                        shutil.rmtree(tempdir)
                    except:
                        pass
        except Exception as e:
            logging.debug(f"MOBI library extraction failed: {e}")
            return ""

    def find_kindleunpack():
        """
        Find kindleunpack module or script in various locations.
        
        Returns:
            tuple: (type, path) where type is 'module', 'script', or None if not found
        """
        
        # Look for kindleunpack script in common locations
        
        # Check PATH for kindleunpack executable
        kindleunpack_cmd = shutil.which('kindleunpack')
        if kindleunpack_cmd:
            return ('script', kindleunpack_cmd)
        
        # Check common locations for the script
        potential_locations = [
            # User's home directory
            os.path.expanduser("~/code/KindleUnpack/lib/kindleunpack.py"),
            os.path.expanduser("~/KindleUnpack/lib/kindleunpack.py"),
            os.path.expanduser("~/kindle/KindleUnpack/lib/kindleunpack.py"),
            # System locations
            "/usr/local/bin/kindleunpack.py",
            "/usr/local/lib/kindleunpack/kindleunpack.py",
            # Add more potential locations as needed
        ]
        
        for location in potential_locations:
            if os.path.exists(location):
                logging.debug(f"Found kindleunpack.py at: {location}")
                return ('script', location)
        
        return (None, None)
    
    def _init_kindleunpack(self):
        """Initialize KindleUnpack functionality"""
        kindleunpack_type, kindleunpack_path = self.find_kindleunpack()
        
        if kindleunpack_type == 'module':
            try:
                # Import as a module
                if not hasattr(self, '_kindleunpack'):
                    self._kindleunpack = self._import_cache.import_module('kindleunpack')
                return True
            except ImportError as e:
                if self._debug:
                    logging.debug(f"Failed to import KindleUnpack module: {e}")
                # Continue to try other methods
        
        if kindleunpack_type == 'script':
            self._kindleunpack_script = kindleunpack_path
            return True
        
        # If we get here, KindleUnpack is not available
        if self._debug:
            logging.debug("KindleUnpack not found")
        return False

    def extract_with_kindleunpack(self, mobi_path: str, progress_callback=None) -> str:
        """Extract text using kindleunpack library or script"""
        if not hasattr(self, '_kindleunpack') and not hasattr(self, '_kindleunpack_script'):
            if not self._init_kindleunpack():
                return ""
        
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                # Check whether to use module or script
                if hasattr(self, '_kindleunpack'):
                    # Use the Python module
                    try:
                        self._kindleunpack.unpack(mobi_path, tempdir)
                    except Exception as e:
                        logging.debug(f"KindleUnpack module unpack failed: {e}")
                        return ""
                elif hasattr(self, '_kindleunpack_script'):
                    # Use the script
                    try:
                        cmd = [sys.executable, self._kindleunpack_script, mobi_path, tempdir]
                        if not self._kindleunpack_script.endswith('.py'):
                            # If it's not a .py file, assume it's directly executable
                            cmd = [self._kindleunpack_script, mobi_path, tempdir]
                        
                        result = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        
                        if result.returncode != 0:
                            logging.debug(f"KindleUnpack script execution failed: {result.stderr}")
                            return ""
                    except Exception as e:
                        logging.debug(f"KindleUnpack script execution error: {e}")
                        return ""
                else:
                    # This shouldn't happen, but just in case
                    return ""
                
                # Process extracted files to get text
                text_parts = []
                
                # Check for HTML files
                html_files = []
                for root, _, files in os.walk(tempdir):
                    for file in files:
                        if file.endswith('.html') or file.endswith('.htm') or file.endswith('.xhtml'):
                            html_files.append(os.path.join(root, file))
                
                # Process HTML files
                if html_files:
                    if self._import_cache.is_available('bs4'):
                        BeautifulSoup = self._import_cache.import_module('bs4').BeautifulSoup
                        
                        for html_file in html_files:
                            try:
                                with open(html_file, 'r', encoding='utf-8', errors='replace') as f:
                                    html_content = f.read()
                                
                                soup = BeautifulSoup(html_content, 'html.parser')
                                # Remove script and style elements
                                for script in soup(["script", "style"]):
                                    script.extract()
                                
                                # Get text
                                text = soup.get_text()
                                lines = (line.strip() for line in text.splitlines())
                                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                                text = '\n'.join(chunk for chunk in chunks if chunk)
                                text_parts.append(text)
                            except Exception as e:
                                logging.debug(f"Error processing HTML file {html_file}: {e}")
                    else:
                        # Basic HTML cleaning if BeautifulSoup not available
                        import re
                        for html_file in html_files:
                            try:
                                with open(html_file, 'r', encoding='utf-8', errors='replace') as f:
                                    html_content = f.read()
                                
                                text = re.sub(r'<[^>]+>', ' ', html_content)
                                text = re.sub(r'\s+', ' ', text).strip()
                                text_parts.append(text)
                            except Exception as e:
                                logging.debug(f"Error processing HTML file {html_file}: {e}")
                
                # Check for text files as well
                for root, _, files in os.walk(tempdir):
                    for file in files:
                        if file.endswith('.txt'):
                            try:
                                with open(os.path.join(root, file), 'r', encoding='utf-8', errors='replace') as f:
                                    text_parts.append(f.read())
                            except Exception as e:
                                logging.debug(f"Error reading text file: {e}")
                
                if progress_callback:
                    progress_callback(1)
                
                return "\n\n".join(text_parts)
                
        except Exception as e:
            logging.debug(f"KindleUnpack extraction failed: {e}")
            return ""

    def extract_with_calibre(self, mobi_path: str, progress_callback=None) -> str:
        """Extract text using Calibre's ebook-converter"""
        try:
            import subprocess
            import tempfile
            
            # Check if ebook-converter or ebook-convert is available
            calibre_bin = None
            for bin_name in ['ebook-converter', 'ebook-convert']:
                try:
                    calibre_bin = shutil.which(bin_name)
                    if calibre_bin:
                        break
                except:
                    pass
            
            if not calibre_bin:
                logging.debug("Calibre ebook-converter/ebook-convert not found")
                return ""
                
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
                output_path = temp.name
                
            # Run Calibre to convert to text
            cmd = [calibre_bin, mobi_path, output_path]
            
            logging.debug(f"Running Calibre command: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                check=False  # Don't raise exception on error
            )
            
            if process.returncode != 0:
                logging.warning(f"Calibre conversion returned error: {process.stderr}")
                
            # Read the output file if it exists
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                    
                # Clean up temporary file
                try:
                    os.unlink(output_path)
                except:
                    pass
                    
                if progress_callback:
                    progress_callback(1)
                    
                return text
            else:
                logging.warning(f"Calibre output file not created: {output_path}")
                return ""
                
        except Exception as e:
            logging.debug(f"Calibre extraction failed: {e}")
            return ""

    def extract_with_zipfile(self, mobi_path: str, progress_callback=None) -> str:
        """Extract text using low-level archive extraction methods"""
        try:
            import tempfile
            import shutil
            import os
            import re
            
            # Since MOBI is a binary format, this is a last resort method
            # Try to extract any text from it as a binary file
            
            # Create a temporary file and folder
            with tempfile.TemporaryDirectory() as tempdir:
                # Try to copy and process as a zip file first
                # (some MOBI files are basically zip containers)
                dest_file = os.path.join(tempdir, 'temp.mobi')
                shutil.copy2(mobi_path, dest_file)
                
                text_parts = []
                
                # Try to extract as a ZIP archive
                try:
                    import zipfile
                    if zipfile.is_zipfile(dest_file):
                        with zipfile.ZipFile(dest_file) as zf:
                            # Extract HTML and text files
                            for item in zf.infolist():
                                if item.filename.endswith('.html') or item.filename.endswith('.htm') or \
                                   item.filename.endswith('.xhtml') or item.filename.endswith('.txt'):
                                    try:
                                        content = zf.read(item).decode('utf-8', errors='replace')
                                        
                                        # Basic HTML cleaning
                                        if item.filename.endswith(('.html', '.htm', '.xhtml')):
                                            content = re.sub(r'<[^>]+>', ' ', content)
                                            content = re.sub(r'\s+', ' ', content).strip()
                                        
                                        text_parts.append(content)
                                    except:
                                        pass
                except:
                    pass
                
                # If no text found, try to extract any printable characters
                if not text_parts:
                    try:
                        with open(dest_file, 'rb') as f:
                            content = f.read()
                            
                            # Try to decode with different encodings
                            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                                try:
                                    text = content.decode(encoding, errors='replace')
                                    
                                    # Extract only printable ASCII characters
                                    printable = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t', ' '])
                                    
                                    # Get only reasonably long words (likely real text, not binary junk)
                                    words = re.findall(r'\b\w{3,}\b', printable)
                                    
                                    if words and len(words) > 100:  # Only if we have a reasonable amount of text
                                        clean_text = ' '.join(words)
                                        text_parts.append(clean_text)
                                        break
                                except:
                                    continue
                    except:
                        pass
                
                if progress_callback:
                    progress_callback(1)
                
                return "\n\n".join(text_parts)
                
        except Exception as e:
            logging.debug(f"MOBI binary extraction failed: {e}")
            return ""
        
class DJVUExtractor:
    """DJVU text extraction with multiple fallback methods"""
    
    def __init__(self, import_cache: ImportCache, debug: bool = False, binary_paths=None):
        self._import_cache = import_cache
        self._debug = debug
        self._available_methods = None
        self._binary_paths = binary_paths or {}
        
        # Check for djvu library specifically
        djvu_type, djvu_path = self.find_djvu_lib()
        self._djvu_type = djvu_type
        self._djvu_path = djvu_path
        
        if self._debug and djvu_type:
            logging.debug(f"Found djvu as {djvu_type} at: {djvu_path}")

    def find_djvu_lib():
        """
        Find djvu Python bindings or command-line tools in various locations.
        
        Returns:
            tuple: (type, path) where type is 'module', 'command', or None if not found
        """
        # First check if python-djvulibre is available
        try:
            import djvu
            return ('module', djvu.__file__)
        except ImportError:
            pass
        
        # Check for djvulibre command line tools
        import shutil
        for binary in ['djvutxt', 'ddjvu']:
            try:
                path = shutil.which(binary)
                if path:
                    return ('command', path)
            except:
                pass
        
        return (None, None)

    @property
    def available_methods(self) -> Dict[str, bool]:
        """Lazy load available methods"""
        if self._available_methods is None:

            
            self._available_methods = {
                'djvulibre': self._check_djvulibre(),
                'pdf_conversion': self._check_pdf_conversion(),
                'ocr': self._check_ocr_dependencies(),
            }
        return self._available_methods

    def _check_djvulibre(self) -> bool:
        """Check if djvulibre tools are available"""
        # First check if python-djvulibre is available
        if self._import_cache.is_available('djvu'):
            return True
            
        # Otherwise check for djvutxt command line tool
        try:
            import shutil
            return shutil.which('djvutxt') is not None
        except:
            return False
    
    def _check_pdf_conversion(self) -> bool:
        """Check if DJVU to PDF conversion is available"""
        try:
            import shutil
            return shutil.which('ddjvu') is not None
        except:
            return False
    
    def _check_ocr_dependencies(self) -> bool:
        """Check if OCR dependencies are available"""
        # We'll reuse the existing OCR infrastructure for images
        ocr_methods = ['tesseract', 'pdf2image', 'pytesseract']
        return all(self._import_cache.is_available(m) for m in ocr_methods)

    def extract_text(self, djvu_path: str, preferred_method: Optional[str] = None,
                    progress_callback: Optional[Callable] = None, **kwargs) -> str:
        """
        Extract text from DJVU with fallback methods
        
        Args:
            djvu_path: Path to DJVU file
            preferred_method: Optional preferred extraction method
            progress_callback: Optional callback for progress updates
            **kwargs: Additional options (ignored)
            
        Returns:
            Extracted text
        """
        methods = ['djvulibre', 'pdf_conversion', 'ocr']
        
        # Reorder methods if preferred method is specified
        if preferred_method and preferred_method in methods:
            methods.insert(0, methods.pop(methods.index(preferred_method)))

        text = ""
        with tqdm(total=len(methods), desc="Trying DJVU extraction methods", unit="method") as method_pbar:
            for method in methods:
                if not self.available_methods.get(method, False):
                    method_pbar.update(1)
                    continue
                    
                try:
                    if progress_callback:
                        progress_callback(0, f"djvu_{method}")  # Signal start with method name
                    
                    extraction_func = getattr(self, f'extract_with_{method}')
                    text = extraction_func(
                        djvu_path,
                        lambda n: progress_callback(n, f"djvu_{method}") if progress_callback else None
                    )
                    
                    if text and text.strip():
                        method_pbar.update(1)
                        break
                        
                except Exception as e:
                    logging.debug(f"Error with DJVU {method}: {e}")
                    
                method_pbar.update(1)

        return text.strip()

    def extract_with_djvulibre(self, djvu_path: str, progress_callback=None) -> str:
        """Extract text using djvulibre library or command-line tools"""
        # Try python-djvulibre first if available
        if self._import_cache.is_available('djvu'):
            try:
                djvu = self._import_cache.import_module('djvu')
                
                # Use the Python bindings for DjVuLibre
                text_parts = []
                
                with djvu.DjVuDocument.create_by_filename(djvu_path) as doc:
                    total_pages = doc.pages_number
                    
                    with tqdm(total=total_pages, desc="DjVuLibre extraction", unit="pages") as pbar:
                        for i in range(total_pages):
                            try:
                                page = doc.pages[i]
                                page_text = page.text.decode('utf-8', errors='replace')
                                if page_text.strip():
                                    text_parts.append(page_text.strip())
                                
                                pbar.update(1)
                                if progress_callback:
                                    progress_callback(1)
                            except Exception as e:
                                logging.debug(f"Error extracting page {i+1}: {e}")
                                if progress_callback:
                                    progress_callback(1)
                
                return "\n\n".join(text_parts)
            except Exception as e:
                logging.debug(f"Python-djvulibre extraction failed: {e}")
                # Fall back to command line
        
        # Use djvutxt command line tool
        try:
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
                temp_path = temp.name
            
            # Run djvutxt to extract text
            cmd = ['djvutxt', djvu_path, temp_path]
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                raise RuntimeError(f"djvutxt failed: {process.stderr}")
            
            # Read the output file
            with open(temp_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
                
            # Clean up temporary file
            import os
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return text
                
        except Exception as e:
            logging.debug(f"DjVuLibre command-line extraction failed: {e}")
            return ""

    def extract_with_pdf_conversion(self, djvu_path: str, progress_callback=None) -> str:
        """Extract text by converting to PDF first, then using PDF extraction"""
        try:
            import tempfile
            import subprocess
            import os
            
            # Create a temporary file for the PDF
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
                pdf_path = temp.name
            
            # Convert DJVU to PDF
            cmd = ['ddjvu', '-format=pdf', djvu_path, pdf_path]
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                raise RuntimeError(f"DJVU to PDF conversion failed: {process.stderr}")
            
            # Use PDFExtractor to extract text from the converted PDF
            from importlib import import_module
            
            # We need to import PDFExtractor from the module
            # But avoid circular imports, so we'll create it dynamically
            text = ""
            try:
                pdf_extractor = PDFExtractor(debug=self._debug)
                text = pdf_extractor.extract_text(
                    pdf_path,
                    progress_callback=progress_callback
                )
            except Exception as e:
                logging.debug(f"PDF extraction after conversion failed: {e}")
            
            # Clean up temporary PDF
            try:
                os.unlink(pdf_path)
            except:
                pass
                
            return text
            
        except Exception as e:
            logging.debug(f"DJVU to PDF conversion failed: {e}")
            return ""

    def extract_with_ocr(self, djvu_path: str, progress_callback=None) -> str:
        """Extract text using OCR by converting to images first"""
        try:
            # First convert DJVU to images
            import tempfile
            import subprocess
            import os
            import glob
            
            # Create a temporary directory for the images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert DJVU to images
                output_pattern = os.path.join(temp_dir, 'page_%d.tif')
                cmd = ['ddjvu', '-format=tiff', djvu_path, output_pattern]
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                if process.returncode != 0:
                    raise RuntimeError(f"DJVU to images conversion failed: {process.stderr}")
                
                # Get list of generated images
                image_files = sorted(glob.glob(os.path.join(temp_dir, 'page_*.tif')))
                
                if not image_files:
                    raise RuntimeError("No images generated from DJVU")
                
                # Perform OCR on the images using tesseract
                import pytesseract
                from PIL import Image
                
                text_parts = []
                with tqdm(total=len(image_files), desc="OCR processing", unit="page") as pbar:
                    for image_file in image_files:
                        try:
                            img = Image.open(image_file)
                            page_text = pytesseract.image_to_string(img, lang='eng')
                            if page_text.strip():
                                text_parts.append(page_text.strip())
                            img.close()
                            
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                        except Exception as e:
                            logging.debug(f"OCR failed for image {image_file}: {e}")
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                
                return "\n\n".join(text_parts)
                
        except Exception as e:
            logging.debug(f"DJVU OCR extraction failed: {e}")
            return ""

class TextExtractor:
    """Plain text file extraction with encoding detection"""
    
    def __init__(self, import_cache: ImportCache, debug: bool = False, binary_paths=None):
        self._import_cache = import_cache
        self._debug = debug
        self._available_methods = None
        self._binary_paths = binary_paths or {}

    @property
    def available_methods(self) -> Dict[str, bool]:
        """Lazy load available methods"""
        if self._available_methods is None:
            self._available_methods = {
                'direct': True,  # Direct file reading is always available
                'charset_detection': self._import_cache.is_available('chardet'),
                'encoding_detection': self._import_cache.is_available('ftfy'),
                'calibre': self._check_calibre_available() 
            }
        return self._available_methods
    
    def _check_calibre_available(self):
        """Check if Calibre converter is available"""
        try:
            # First check if we have the path in binary_paths
            if self._binary_paths.get('ebook-converter'):
                return True
                
            # Otherwise check standard locations
            for bin_name in ['ebook-converter', 'ebook-convert']:
                if shutil.which(bin_name):
                    return True
            return False
        except:
            return False

    def extract_text(self, txt_path: str, preferred_method: Optional[str] = None,
                    progress_callback: Optional[Callable] = None, **kwargs) -> str:
        """
        Extract text from plain text file with encoding detection
        
        Args:
            txt_path: Path to text file
            preferred_method: Optional preferred extraction method
            progress_callback: Optional callback for progress updates
            **kwargs: Additional options (ignored)
            
        Returns:
            Extracted text
        """
        # Determine if we should use Calibre based on file extension
        file_ext = os.path.splitext(txt_path)[1].lower()
        try_calibre_first = file_ext in ['.docx', '.doc', '.rtf', '.fb2', '.pdb', '.lit', 
                                         '.odt', '.lrf', '.cbz', '.cbr', '.chm', '.snb', '.tcr']
        
        # Override preferred_method if file extension suggests Calibre
        if try_calibre_first and not preferred_method:
            preferred_method = 'calibre'
        
        # If user explicitly asks for calibre, use it
        if preferred_method == 'calibre':
            methods = ['calibre', 'charset_detection', 'encoding_detection', 'direct']
        else:
            methods = ['charset_detection', 'encoding_detection', 'direct', 'calibre']
        
        # Reorder methods if preferred method is specified (but not 'calibre', handled above)
        if preferred_method and preferred_method != 'calibre':
            methods.insert(0, methods.pop(methods.index(preferred_method)))

        text = ""
        with tqdm(total=len(methods), desc="Trying text extraction methods", unit="method") as method_pbar:
            for method in methods:
                if not self.available_methods.get(method, False) and method != 'direct':
                    method_pbar.update(1)
                    continue
                    
                try:
                    if progress_callback:
                        progress_callback(0, f"text_{method}")  # Signal start with method name
                    
                    extraction_func = getattr(self, f'extract_with_{method}')
                    text = extraction_func(
                        txt_path,
                        lambda n: progress_callback(n, f"text_{method}") if progress_callback else None
                    )
                    
                    if text and text.strip():
                        method_pbar.update(1)
                        break
                        
                except Exception as e:
                    logging.debug(f"Error with text {method}: {e}")
                    
                method_pbar.update(1)

        return text.strip()

    def extract_with_direct(self, txt_path: str, progress_callback=None) -> str:
        """Extract text directly with UTF-8 encoding"""
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
                
            if progress_callback:
                progress_callback(1)
                
            return text
        except Exception as e:
            logging.debug(f"Direct text extraction failed: {e}")
            return ""
        
    def extract_with_calibre(self, txt_path: str, progress_callback=None) -> str:
        """Extract text using Calibre's ebook-converter"""
        try:
            import subprocess
            import tempfile
            
            # Check if ebook-converter or ebook-convert is available
            calibre_bin = None
            for bin_name in ['ebook-converter', 'ebook-convert']:
                try:
                    calibre_bin = shutil.which(bin_name)
                    if calibre_bin:
                        break
                except:
                    pass
            
            if not calibre_bin:
                logging.debug("Calibre ebook-converter/ebook-convert not found")
                return ""
                
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
                output_path = temp.name
                
            # Run Calibre to convert to text
            cmd = [calibre_bin, txt_path, output_path]
            
            logging.debug(f"Running Calibre command: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                check=False  # Don't raise exception on error
            )
            
            if process.returncode != 0:
                logging.warning(f"Calibre conversion returned error: {process.stderr}")
                
            # Read the output file if it exists
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                    
                # Clean up temporary file
                try:
                    os.unlink(output_path)
                except:
                    pass
                    
                if progress_callback:
                    progress_callback(1)
                    
                return text
            else:
                logging.warning(f"Calibre output file not created: {output_path}")
                return ""
                
        except Exception as e:
            logging.debug(f"Calibre extraction failed: {e}")
            return ""

    def extract_with_charset_detection(self, txt_path: str, progress_callback=None) -> str:
        """Extract text with charset detection"""
        try:
            chardet = self._import_cache.import_module('chardet')
            
            # First read the file as binary
            with open(txt_path, 'rb') as f:
                raw_data = f.read()
                
            # Detect encoding
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            if confidence > 0.7:  # Only use if confidence is reasonable
                logging.debug(f"Detected encoding {encoding} with confidence {confidence:.2f}")
                # Decode using detected encoding
                text = raw_data.decode(encoding, errors='replace')
                
                if progress_callback:
                    progress_callback(1)
                    
                return text
            else:
                logging.debug(f"Low encoding confidence: {confidence:.2f} for {encoding}")
                return ""
                
        except Exception as e:
            logging.debug(f"Charset detection failed: {e}")
            return ""

    def extract_with_encoding_detection(self, txt_path: str, progress_callback=None) -> str:
        """Extract text using ftfy for fixing encoding issues"""
        try:
            ftfy = self._import_cache.import_module('ftfy')
            
            # First try direct reading
            with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
                raw_text = f.read()
                
            # Fix text encoding
            fixed_text = ftfy.fix_text(raw_text)
            
            if progress_callback:
                progress_callback(1)
                
            return fixed_text
            
        except Exception as e:
            logging.debug(f"Encoding detection failed: {e}")
            return ""


class HTMLExtractor:
    """HTML file extraction with multiple fallback methods"""
    
    def __init__(self, import_cache: ImportCache, debug: bool = False, binary_paths=None):
        self._import_cache = import_cache
        self._debug = debug
        self._available_methods = None
        self._binary_paths = binary_paths or {}

    @property
    def available_methods(self) -> Dict[str, bool]:
        """Lazy load available methods"""
        if self._available_methods is None:
            self._available_methods = {
                'calibre': self._check_calibre_available(),
                'bs4': self._import_cache.is_available('bs4'),
                'html2text': self._import_cache.is_available('html2text'),
                'lxml': self._import_cache.is_available('lxml'),
                'regex': True  # Basic regex is always available
            }
        return self._available_methods
    
    def _check_calibre_available(self):
        """Check if Calibre converter is available"""
        try:
            # First check if we have the path in binary_paths
            if self._binary_paths.get('ebook-converter'):
                return True
                
            # Otherwise check standard locations
            for bin_name in ['ebook-converter', 'ebook-convert']:
                if shutil.which(bin_name):
                    return True
            return False
        except:
            return False

    def extract_text(self, html_path: str, preferred_method: Optional[str] = None,
                    progress_callback: Optional[Callable] = None, **kwargs) -> str:
        """
        Extract text from HTML file with multiple fallback methods
        
        Args:
            html_path: Path to HTML file
            preferred_method: Optional preferred extraction method
            progress_callback: Optional callback for progress updates
            **kwargs: Additional options (ignored)
            
        Returns:
            Extracted text
        """
        methods = ['bs4', 'html2text', 'lxml', 'regex']
        
        # Reorder methods if preferred method is specified
        if preferred_method and preferred_method in methods:
            methods.insert(0, methods.pop(methods.index(preferred_method)))

        text = ""
        with tqdm(total=len(methods), desc="Trying HTML extraction methods", unit="method") as method_pbar:
            for method in methods:
                if not self.available_methods.get(method, False) and method != 'regex':
                    method_pbar.update(1)
                    continue
                    
                try:
                    if progress_callback:
                        progress_callback(0, f"html_{method}")  # Signal start with method name
                    
                    extraction_func = getattr(self, f'extract_with_{method}')
                    text = extraction_func(
                        html_path,
                        lambda n: progress_callback(n, f"html_{method}") if progress_callback else None
                    )
                    
                    if text and text.strip():
                        method_pbar.update(1)
                        break
                        
                except Exception as e:
                    logging.debug(f"Error with HTML {method}: {e}")
                    
                method_pbar.update(1)

        return text.strip()

    def extract_with_bs4(self, html_path: str, progress_callback=None) -> str:
        """Extract text using BeautifulSoup"""
        try:
            BeautifulSoup = self._import_cache.import_module('bs4').BeautifulSoup
            
            # First read the file
            with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()
                
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            if progress_callback:
                progress_callback(1)
                
            return text
            
        except Exception as e:
            logging.debug(f"BeautifulSoup extraction failed: {e}")
            return ""

    def extract_with_html2text(self, html_path: str, progress_callback=None) -> str:
        """Extract text using html2text"""
        try:
            html2text = self._import_cache.import_module('html2text').HTML2Text()
            
            # Configure html2text
            html2text.ignore_links = True
            html2text.ignore_images = True
            html2text.ignore_tables = False
            html2text.body_width = 0  # No wrapping
            
            # Read the file
            with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()
                
            # Convert to markdown
            text = html2text.handle(html_content)
            
            if progress_callback:
                progress_callback(1)
                
            return text
            
        except Exception as e:
            logging.debug(f"html2text extraction failed: {e}")
            return ""

    def extract_with_lxml(self, html_path: str, progress_callback=None) -> str:
        """Extract text using lxml"""
        try:
            lxml_html = self._import_cache.import_module('lxml.html')
            
            # Parse HTML file
            with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()
                
            # Parse with lxml
            root = lxml_html.fromstring(html_content)
            
            # Remove script and style elements
            for elem in root.xpath('//script | //style | //meta | //noscript | //header | //footer | //nav'):
                elem.getparent().remove(elem)
            
            # Extract text
            text = ' '.join(root.xpath('//text()'))
            
            # Clean up text
            import re
            text = re.sub(r'\s+', ' ', text).strip()
            
            if progress_callback:
                progress_callback(1)
                
            return text
            
        except Exception as e:
            logging.debug(f"lxml extraction failed: {e}")
            return ""

    def extract_with_regex(self, html_path: str, progress_callback=None) -> str:
        """Extract text using basic regex patterns"""
        try:
            import re
            
            # Read the file
            with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()
                
            # Remove script and style sections
            html_content = re.sub(r'<script[^>]*>.*?</script>', ' ', html_content, flags=re.DOTALL)
            html_content = re.sub(r'<style[^>]*>.*?</style>', ' ', html_content, flags=re.DOTALL)
            
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', html_content)
            
            # Clean up whitespace
            text = re.sub(r'&nbsp;', ' ', text)
            text = re.sub(r'&amp;', '&', text)
            text = re.sub(r'&lt;', '<', text)
            text = re.sub(r'&gt;', '>', text)
            text = re.sub(r'&quot;', '"', text)
            text = re.sub(r'&apos;', "'", text)
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split into paragraphs based on multiple newlines
            paragraphs = re.split(r'\n\s*\n', text)
            clean_paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            if progress_callback:
                progress_callback(1)
                
            return '\n\n'.join(clean_paragraphs)
            
        except Exception as e:
            logging.debug(f"Regex extraction failed: {e}")
            return ""

class EPUBExtractor:
    """EPUB text extraction with multiple fallback methods"""
    
    def __init__(self, import_cache: ImportCache, debug: bool = False, binary_paths=None):
        self._import_cache = import_cache
        self._debug = debug
        self._checked_methods = {}
        self._available_methods = None
        self._binary_paths = binary_paths or {}

    @property
    def available_methods(self) -> Dict[str, bool]:
        """Lazy load available methods"""
        if self._available_methods is None:
            self._available_methods = {
                'ebooklib': self._import_cache.is_available('ebooklib'),
                'bs4': self._import_cache.is_available('bs4'),
                'html2text': self._import_cache.is_available('html2text'),
                'calibre': self._check_calibre_available(),  
                'zipfile': True  # Basic fallback always available
            }
        return self._available_methods
    
    def _check_calibre_available(self):
        """Check if Calibre converter is available"""
        try:
            # First check if we have the path in binary_paths
            if self._binary_paths.get('ebook-converter'):
                return True
                
            # Otherwise check standard locations
            for bin_name in ['ebook-converter', 'ebook-convert']:
                if shutil.which(bin_name):
                    return True
            return False
        except:
            return False

    def extract_text(self, epub_path: str, preferred_method: Optional[str] = None,
                    progress_callback: Optional[Callable] = None) -> str:
        """
        Extract text with fallback methods and progress reporting
        
        Args:
            epub_path: Path to EPUB file
            preferred_method: Optional preferred extraction method
            progress_callback: Optional callback for progress updates
            
        Returns:
            Extracted text
        """
        methods = ['ebooklib', 'bs4', 'calibre', 'zipfile']
        if preferred_method:
            if preferred_method not in methods:
                raise ValueError(f"Invalid method: {preferred_method}")
            methods.insert(0, methods.pop(methods.index(preferred_method)))

        text = ""
        with tqdm(total=len(methods), desc="Trying extraction methods", unit="method") as method_pbar:
            for method in methods:
                if not self.available_methods.get(method, False):
                    method_pbar.update(1)
                    continue
                    
                try:
                    if progress_callback:
                        progress_callback(0, method)  # Signal start with method name
                    
                    extraction_func = getattr(self, f'extract_with_{method}')
                    text = extraction_func(
                        epub_path,
                        lambda n: progress_callback(n, method) if progress_callback else None
                    )
                    
                    if text and text.strip():
                        method_pbar.update(1)
                        break
                        
                except Exception as e:
                    logging.debug(f"Error with {method}: {e}")
                    
                method_pbar.update(1)

        return text.strip()

    def extract_with_ebooklib(self, epub_path: str, progress_callback=None) -> str:
        """Extract using ebooklib with BeautifulSoup parsing and progress bars"""
        ebooklib = self._import_cache.import_module('ebooklib')
        BeautifulSoup = self._import_cache.import_module('bs4').BeautifulSoup
        
        text_parts = []
        book = None
        
        try:
            with tqdm(desc="Loading EPUB", unit="file") as pbar:
                book = ebooklib.epub.read_epub(epub_path)
                pbar.update(1)
            
            items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
            
            with tqdm(total=len(items), desc="Extracting content", unit="item") as pbar:
                for i, item in enumerate(items):
                    try:
                        content = item.get_content().decode('utf-8')
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Remove unwanted elements
                        for tag in soup(['script', 'style', 'nav']):
                            tag.decompose()
                        
                        # Extract text with layout preservation
                        text = self._process_html_content(soup)
                        if text.strip():
                            text_parts.append(text.strip())
                        
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(1)
                            
                    except Exception as e:
                        logging.debug(f"Item extraction failed: {e}")
                        continue
                    
        finally:
            book = None  # Release memory
            
        return '\n\n'.join(text_parts)

    def _process_html_content(self, soup) -> str:
        """Process HTML content with layout preservation"""
        text_parts = []
        
        # Process headings with progress
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        with tqdm(total=len(headings), desc="Processing headings", unit="heading", leave=False) as pbar:
            for tag in headings:
                text = tag.get_text(strip=True)
                if text:
                    text_parts.append(f"\n{text}\n")
                pbar.update(1)
        
        # Process paragraphs and other block elements with progress
        blocks = soup.find_all(['p', 'div', 'section'])
        with tqdm(total=len(blocks), desc="Processing blocks", unit="block", leave=False) as pbar:
            for tag in blocks:
                text = tag.get_text(strip=True)
                if text:
                    text_parts.append(text)
                pbar.update(1)
        
        return '\n\n'.join(text_parts)
    
    def extract_with_calibre(self, epub_path: str, progress_callback=None) -> str:
        """Extract text using Calibre's ebook-converter"""
        try:
            import subprocess
            import tempfile
            
            # Check if ebook-converter or ebook-convert is available
            calibre_bin = None
            for bin_name in ['ebook-converter', 'ebook-convert']:
                try:
                    calibre_bin = shutil.which(bin_name)
                    if calibre_bin:
                        break
                except:
                    pass
            
            if not calibre_bin:
                logging.debug("Calibre ebook-converter/ebook-convert not found")
                return ""
                
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
                output_path = temp.name
                
            # Run Calibre to convert to text
            cmd = [calibre_bin, epub_path, output_path]
            
            logging.debug(f"Running Calibre command: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                check=False  # Don't raise exception on error
            )
            
            if process.returncode != 0:
                logging.warning(f"Calibre conversion returned error: {process.stderr}")
                
            # Read the output file if it exists
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                    
                # Clean up temporary file
                try:
                    os.unlink(output_path)
                except:
                    pass
                    
                if progress_callback:
                    progress_callback(1)
                    
                return text
            else:
                logging.warning(f"Calibre output file not created: {output_path}")
                return ""
                
        except Exception as e:
            logging.debug(f"Calibre extraction failed: {e}")
            return ""

    def extract_with_bs4(self, epub_path: str, progress_callback=None) -> str:
        """Extract using BeautifulSoup with zipfile and progress bars"""
        BeautifulSoup = self._import_cache.import_module('bs4').BeautifulSoup
        html2text = self._import_cache.import_module('html2text').HTML2Text()
        zipfile = self._import_cache.import_module('zipfile')
        
        text_parts = []
        
        try:
            with zipfile.ZipFile(epub_path) as zf:
                # Get HTML files
                html_files = [f for f in zf.namelist() 
                            if f.endswith(('.html', '.xhtml', '.htm'))]
                
                with tqdm(total=len(html_files), desc="Processing HTML files", unit="file") as pbar:
                    for i, html_file in enumerate(html_files):
                        try:
                            content = zf.read(html_file).decode('utf-8')
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Remove unwanted elements
                            for tag in soup(['script', 'style', 'nav']):
                                tag.decompose()
                            
                            # Convert to markdown-style text
                            html2text.ignore_links = True
                            html2text.ignore_images = True
                            text = html2text.handle(str(soup))
                            
                            if text.strip():
                                text_parts.append(text.strip())
                            
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                                
                        except Exception as e:
                            logging.debug(f"File extraction failed: {e}")
                            continue
                            
        except Exception as e:
            logging.error(f"EPUB extraction failed: {e}")
            
        return '\n\n'.join(text_parts)

    def extract_with_zipfile(self, epub_path: str, progress_callback=None) -> str:
        """Basic fallback extraction using zipfile with progress bars"""
        zipfile = self._import_cache.import_module('zipfile')
        import re
        
        text_parts = []
        html_pattern = re.compile(r'<[^>]+>')
        
        try:
            with zipfile.ZipFile(epub_path) as zf:
                html_files = [f for f in zf.namelist() 
                            if f.endswith(('.html', '.xhtml', '.htm'))]
                
                with tqdm(total=len(html_files), desc="Extracting text", unit="file") as pbar:
                    for i, html_file in enumerate(html_files):
                        try:
                            content = zf.read(html_file).decode('utf-8')
                            
                            # Basic HTML cleaning
                            content = re.sub(r'<script.*?</script>', '', content, 
                                           flags=re.DOTALL)
                            content = re.sub(r'<style.*?</style>', '', content, 
                                           flags=re.DOTALL)
                            content = html_pattern.sub(' ', content)
                            
                            # Clean up whitespace
                            content = re.sub(r'\s+', ' ', content).strip()
                            
                            if content:
                                text_parts.append(content)
                            
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                                
                        except Exception as e:
                            logging.debug(f"File extraction failed: {e}")
                            continue
                            
        except Exception as e:
            logging.error(f"EPUB extraction failed: {e}")
            
        return '\n\n'.join(text_parts)

class TableExtractor:
    """PDF table extraction using Camelot"""
    
    def __init__(self, import_cache: ImportCache):
        self._import_cache = import_cache
        self._camelot = None
        
    def extract_tables(self, pdf_path: str) -> List[Any]:
        """Extract tables using multiple methods"""
        if not self._init_camelot():
            return []
            
        tables = []
        methods = [
            ('lattice', {'line_scale': 40}),
            ('stream', {'edge_tol': 500})
        ]
        
        for method, params in methods:
            try:
                current_tables = self._camelot.read_pdf(
                    pdf_path,
                    flavor=method,
                    pages='all',
                    **params
                )
                
                if len(current_tables) > 0:
                    tables.extend(current_tables)
                    
            except Exception as e:
                logging.debug(f"Table extraction failed with {method}: {e}")
                continue
                
        return tables
        
    def _init_camelot(self) -> bool:
        """Initialize Camelot library only when needed."""
        if self._camelot is None:
            try:
                # Lazy import for Camelot (which may in turn import pydot)
                self._camelot = self._import_cache.import_module('camelot')
                return True
            except Exception as e:
                logging.error(f"Failed to initialize Camelot: {e}")
                return False
        return True

    
class PDFExtractor:
    """Enhanced PDF text extraction with lazy loading and multiple fallback methods"""

    TEXT_METHODS = [
        'pymupdf',      # Fast native PDF parsing
        'pdfplumber',   # Good balance of speed and accuracy
        'calibre',      # proven
        'pypdf',        # Simple but reliable
        'pdfminer',     # Good layout preservation
        'tesseract',    # OCR support
        'easyocr',      # Alternative OCR
        'paddleocr',    # multilingual: https://paddlepaddle.github.io/PaddleOCR/main/en/ppocr/blog/multi_languages.html
        'doctr',        # Deep learning OCR 
        'kraken_cli',   # Kraken CLI method (seems we still need e.g. numpy==1.26.4 & tensorflow-macos-2.15.0)
        'kraken'        # Kraken API method (lower priority)
        
    ]
    # Lists for categorizing methods
    CORE_METHODS = ['pymupdf', 'calibre', 'pdfplumber', 'pypdf', 'pdfminer']
    OCR_METHODS = ['tesseract', 'easyocr', 'paddleocr', 'doctr', 'kraken', 'kraken_cli']
    
    TABLE_METHODS = ['camelot']
    
    def __init__(self, debug=False, binary_paths=None):
        """
        Initialize PDF extractor with optional binary paths
        
        Args:
            debug: Enable debug logging
            binary_paths: Optional dictionary of binary paths to use instead of detecting
        """
        self._debug = debug
        self._import_cache = ImportCache()
        self._initialized_methods = set()
        self._password = None
        self._current_doc = None
        self._ocr_initialized = {}
        self._available_methods = None
        self._ocr_failed_methods = set()
        
        # Setup Windows paths first
        self._setup_windows_paths()
        
        # Use provided binary paths or detect them
        if binary_paths:
            self._binary_paths = binary_paths
            self._binaries = {
                'tesseract': bool(binary_paths.get('tesseract')),
                'poppler': bool(binary_paths.get('pdftoppm')),
                'ghostscript': bool(binary_paths.get('gs')),
                'djvulibre': bool(binary_paths.get('djvutxt')) or bool(binary_paths.get('ddjvu')),
                'calibre': bool(binary_paths.get('ebook-converter'))
            }
            logging.debug("Using provided binary paths for PDF extractor")
        else:
            # If no binary paths provided, detect them
            # This should never happen if properly initialized from ExtractionManager
            logging.warning("No binary paths provided to PDFExtractor, detecting binaries")
            self._binary_paths, self._binaries = self._check_system_dependencies()
        
        # Check core dependencies first to prioritize stable methods
        self._check_core_dependencies()
        
        # Only check OCR dependencies if debug mode is on or we have binaries
        if debug or self._binaries.get('tesseract', False):
            self._check_ocr_dependencies()
        
        available = sorted(list(self._initialized_methods))
        logging.debug(f"Available extraction methods: {', '.join(available)}")

    def _setup_windows_paths(self):
        """Add binary paths to system PATH for Windows"""
        if platform.system() == 'Windows':
            # Define paths to check with most specific paths first
            paths_to_check = [
                # Ghostscript - specific versions
                r'C:\Program Files\gs\gs10.04.0\bin',
                r'C:\Program Files\gs\gs10.02.0\bin',
                r'C:\Program Files (x86)\gs\gs10.04.0\bin',
                r'C:\Program Files (x86)\gs\gs10.02.0\bin',
                # Poppler - specific versions
                r'C:\Users\stc\Downloads\code\poppler-24.08.0\Library\bin',
                r'C:\Program Files\poppler-24.02.0\Library\bin',
                # Tesseract
                r'C:\Program Files\Tesseract-OCR',
                r'C:\Program Files (x86)\Tesseract-OCR',
            ]
            
            # Add each existing path to PATH
            paths_added = []
            for path in paths_to_check:
                if os.path.exists(path) and path not in os.environ['PATH']:
                    os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
                    paths_added.append(path)
            
            if paths_added and self._debug:
                logging.debug(f"Added to PATH: {', '.join(paths_added)}")

    def _check_calibre_available(self):
        """Check if Calibre converter is available"""
        try:
            # First check binary_paths
            if self._binary_paths and 'ebook-converter' in self._binary_paths and self._binary_paths['ebook-converter']:
                if self._debug:
                    logging.debug(f"Found Calibre in binary_paths: {self._binary_paths['ebook-converter']}")
                return True
                
            # Then check PATH
            for bin_name in ['ebook-converter', 'ebook-convert']:
                calibre_path = shutil.which(bin_name)
                if calibre_path:
                    if self._debug:
                        logging.debug(f"Found Calibre in PATH: {calibre_path}")
                    return True
                    
            if self._debug:
                logging.debug("Calibre not found")
            return False
        except Exception as e:
            if self._debug:
                logging.debug(f"Error checking calibre: {e}")
            return False
    
    def _safe_import(self, module_name):
        """Safely import a module with error handling"""
        try:
            # Use importlib directly instead of relying on a global import
            import importlib
            
            # Split module_name to handle submodules (e.g., 'PIL.Image')
            parts = module_name.split('.')
            if len(parts) > 1:
                # For submodules, import the base and then get the attribute
                base_module = importlib.import_module(parts[0])
                current = base_module
                
                # Navigate through the module hierarchy
                for part in parts[1:]:
                    current = getattr(current, part)
                
                return current
            else:
                # Direct import for simple module names
                return importlib.import_module(module_name)
        except ImportError as e:
            if self._debug:
                logging.debug(f"Cannot import {module_name}: {e}")
            return None
        except Exception as e:
            if self._debug:
                logging.debug(f"Error importing {module_name}: {e}")
            return None
    
    @property
    def languages(self):
        """Return list of OCR languages"""
        return ['eng']  # Add more languages as needed

    def get_binary_path(self, binary_name: str) -> Optional[str]:
        """
        Get the full path to a binary if available
        
        Args:
            binary_name: Name of the binary
            
        Returns:
            str or None: Path to the binary or None if not found
        """
        # Map common names to the actual binary keys
        binary_map = {
            'tesseract': 'tesseract',
            'pdftoppm': 'pdftoppm',
            'poppler': 'pdftoppm',
            'gs': 'gs',
            'ghostscript': 'gs',
            'djvutxt': 'djvutxt',
            'ddjvu': 'ddjvu',
            'djvulibre': 'djvutxt',  # Default to djvutxt for djvulibre
            'ebook-converter': 'ebook-converter',
            'ebook-convert': 'ebook-converter',
            'calibre': 'ebook-converter'
        }
        
        # Get the actual binary key
        binary_key = binary_map.get(binary_name, binary_name)
        
        # Return the path from the stored binary paths
        binary_paths = getattr(self, '_binary_paths', {})
        return binary_paths.get(binary_key)
    
    def _check_system_dependencies(self) -> Dict[str, bool]:
        """
        Check system dependencies with comprehensive binary detection.
        Stores binary paths for later use and returns a boolean compatibility dict.
        """
        logging.debug("Checking system dependencies for PDFEXtractor:")
        
        # Dictionary to store binary paths (not just boolean values)
        binary_paths = {
            'tesseract': None,       # OCR engine
            'pdftoppm': None,        # PDF to image conversion (poppler)
            'gs': None,              # Ghostscript
            'djvutxt': None,         # For DJVU text extraction
            'ddjvu': None,           # For DJVU conversion
            'ebook-converter': None  # Calibre converter
        }
        
        # Executable name variations by platform and binary
        executable_names = {
            'tesseract': ['tesseract', 'tesseract.exe'],
            'pdftoppm': ['pdftoppm', 'pdftoppm.exe'],
            'gs': ['gs', 'gswin64c', 'gswin64c.exe', 'gswin32c.exe'],
            'djvutxt': ['djvutxt', 'djvutxt.exe'],
            'ddjvu': ['ddjvu', 'ddjvu.exe'],
            'ebook-converter': ['ebook-converter', 'ebook-convert', 'ebook-converter.exe', 'ebook-convert.exe']
        }
        
        # Version check flags for different binaries
        version_flags = {
            'tesseract': '--version',
            'pdftoppm': '-v',
            'gs': '--version',
            'djvutxt': '',         # DjVuLibre tools don't have a version flag
            'ddjvu': '--help',     
            'ebook-converter': '-h'
        }
        
        # Common installation directories by platform
        system = platform.system()
        common_dirs = {}
        
        if system == 'Windows':
            common_dirs = {
                'tesseract': [
                    r'C:\Program Files\Tesseract-OCR',
                    r'C:\Program Files (x86)\Tesseract-OCR'
                ],
                'pdftoppm': [
                    r'C:\Program Files\poppler-24.08.0\Library\bin',
                    r'C:\Program Files\poppler-24.02.0\Library\bin',
                    r'C:\Program Files\poppler\bin',
                    r'C:\Program Files (x86)\poppler\bin',
                    # Additional paths users might have installed to
                    r'C:\poppler\bin'
                ],
                'gs': [
                    r'C:\Program Files\gs\gs10.04.0\bin',
                    r'C:\Program Files\gs\gs10.02.0\bin',
                    r'C:\Program Files (x86)\gs\gs10.04.0\bin',
                    r'C:\Program Files (x86)\gs\gs10.02.0\bin',
                    # Generic location
                    r'C:\Program Files\gs\bin'
                ],
                'ebook-converter': [
                    r'C:\Program Files\Calibre2',
                    r'C:\Program Files (x86)\Calibre2',
                    r'C:\Program Files\Calibre',
                    r'C:\Program Files (x86)\Calibre'
                ],
                'djvutxt': [
                    r'C:\Program Files\DjVuLibre',
                    r'C:\Program Files (x86)\DjVuLibre'
                ],
                'ddjvu': [
                    r'C:\Program Files\DjVuLibre',
                    r'C:\Program Files (x86)\DjVuLibre'
                ]
            }
        elif system == 'Darwin':  # macOS
            # macOS typical install locations including Homebrew and MacPorts
            common_dirs = {
                'tesseract': [
                    '/usr/local/bin',
                    '/opt/homebrew/bin',  # Apple Silicon homebrew
                    '/opt/local/bin'      # MacPorts
                ],
                'pdftoppm': [
                    '/usr/local/bin',
                    '/opt/homebrew/bin',
                    '/opt/local/bin'
                ],
                'gs': [
                    '/usr/local/bin',
                    '/opt/homebrew/bin',
                    '/opt/local/bin'
                ],
                'ebook-converter': [
                    '/Applications/calibre.app/Contents/MacOS',
                    '/opt/homebrew/bin',
                    '/usr/local/bin',
                    '~/bin'  # User's bin directory
                ],
                'djvutxt': [
                    '/usr/local/bin',
                    '/opt/homebrew/bin',
                    '/opt/local/bin'
                ],
                'ddjvu': [
                    '/usr/local/bin',
                    '/opt/homebrew/bin',
                    '/opt/local/bin'
                ]
            }
        else:  # Linux and others
            common_dirs = {
                'tesseract': [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/opt/bin'
                ],
                'pdftoppm': [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/opt/bin'
                ],
                'gs': [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/opt/bin'
                ],
                'ebook-converter': [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/opt/bin',
                    '~/bin'  # User's bin directory
                ],
                'djvutxt': [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/opt/bin'
                ],
                'ddjvu': [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/opt/bin'
                ]
            }
            
            # Handle potential user home directory paths
            for binary in common_dirs:
                expanded_paths = []
                for path in common_dirs[binary]:
                    if '~' in path:
                        expanded_paths.append(os.path.expanduser(path))
                    else:
                        expanded_paths.append(path)
                common_dirs[binary] = expanded_paths
        
        # First check PATH for each binary
        for binary, names in executable_names.items():
            for name in names:
                path = shutil.which(name)
                if path:
                    binary_paths[binary] = path
                    if self._debug:
                        logging.debug(f"Found {binary} in PATH: {path}")
                    break
        
        # For binaries not found in PATH, check common directories
        for binary, path in binary_paths.items():
            if path is None and binary in common_dirs:
                for directory in common_dirs[binary]:
                    if not os.path.exists(directory):
                        continue
                        
                    for name in executable_names[binary]:
                        full_path = os.path.join(directory, name)
                        if os.path.exists(full_path) and os.access(full_path, os.X_OK):
                            binary_paths[binary] = full_path
                            if self._debug:
                                logging.debug(f"Found {binary} in common directory: {full_path}")
                            break
                    if binary_paths[binary]:  # Stop checking if found
                        break
        
        # Verify binary versions if found
        for binary, path in binary_paths.items():
            if path:
                # Skip verification for Calibre on Mac as it might not be executable directly
                if binary == 'ebook-converter' and system == 'Darwin' and '/Applications/calibre.app' in path:
                    if self._debug:
                        logging.debug(f"Skipping version check for {binary} in Mac app bundle")
                    continue

                # Get the appropriate version flag
                version_flag = version_flags.get(binary, '--version')
                try:
                    # For binaries that might error on version check but still work
                    try:
                        result = subprocess.run(
                            [path, version_flag],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=2  # Prevent hanging
                        )
                        
                        # Extract version info from output
                        if result.stdout:
                            version_info = result.stdout.splitlines()[0]
                        elif result.stderr:
                            version_info = result.stderr.splitlines()[0]
                        else:
                            version_info = f"Available (no version info)"
                        
                        if self._debug:
                            logging.debug(f"Verified {binary}: {version_info}")
                    except subprocess.CalledProcessError as e:
                        # For binaries that return non-zero but might still work
                        if self._debug:
                            logging.debug(f"Version check for {binary} returned non-zero: {e.returncode}")
                            if e.stdout:
                                logging.debug(f"  stdout: {e.stdout.splitlines()[0] if e.stdout else ''}")
                            if e.stderr:
                                logging.debug(f"  stderr: {e.stderr.splitlines()[0] if e.stderr else ''}")

                        # If we get here, the binary exists but the version check failed
                        # We'll still consider it available
                        if self._debug:
                            logging.debug(f"Considering {binary} available despite version check failure")
                except Exception as e:
                    if self._debug:
                        logging.debug(f"Failed to verify {binary} version: {e}")
        
        # Log missing binaries with download information
        download_info = {
            'tesseract': "Tesseract not found. Download from: https://github.com/UB-Mannheim/tesseract/wiki",
            'pdftoppm': "Poppler (pdftoppm) not found. Download from: https://github.com/oschwartz10612/poppler-windows/releases/",
            'gs': "Ghostscript not found. Download from: https://ghostscript.com/releases/gsdnld.html",
            'ebook-converter': "Calibre not found. Download from: https://calibre-ebook.com/download",
            'djvutxt': "DjVuLibre not found. For Windows: https://sourceforge.net/projects/djvu/files/DjVuLibre_Windows/",
            'ddjvu': "DjVuLibre (ddjvu) not found. For Windows: https://sourceforge.net/projects/djvu/files/DjVuLibre_Windows/"
        }
        
        # Only show warnings for missing binaries once by tracking which warnings we've shown
        warned_about = getattr(self, '_warned_about_binaries', set())
        
        for binary, path in binary_paths.items():
            if not path and binary in download_info and binary not in warned_about:
                logging.info(download_info[binary])
                warned_about.add(binary)
        
        # Store which binaries we've warned about
        self._warned_about_binaries = warned_about
        
        # Store paths in instance for later use
        self._binary_paths = binary_paths
        
        # Create backward-compatible boolean results dictionary
        # Map typical names to the binaries
        binaries_bool = {
            'tesseract': bool(binary_paths['tesseract']),
            'poppler': bool(binary_paths['pdftoppm']),
            'ghostscript': bool(binary_paths['gs']),
            'djvulibre': bool(binary_paths['djvutxt']) or bool(binary_paths['ddjvu']),
            'calibre': bool(binary_paths['ebook-converter'])
        }
        
        return binaries_bool
    
    def _check_core_dependencies(self):
        """Check core text extraction dependencies and add to initialized methods"""
        # These are the most reliable methods, so check them first

        # Check for calibre (using binary_paths first, then PATH)
        # Do this first as it doesn't require imports
        if self._check_calibre_available():
            self._initialized_methods.add('calibre')
            if self._debug:
                logging.debug("Calibre ebook-converter available")
        
        # Check for PyMuPDF
        try:
            import fitz  # pymupdf
            self._initialized_methods.add('pymupdf')
            if self._debug:
                logging.debug("PyMuPDF (fitz) available")
        except ImportError:
            if self._debug:
                logging.debug("PyMuPDF not available")

        # Check for pdfplumber
        try:
            import pdfplumber
            self._initialized_methods.add('pdfplumber')
            if self._debug:
                logging.debug("pdfplumber available")
        except ImportError:
            if self._debug:
                logging.debug("pdfplumber not available")

        # Check for pypdf
        try:
            import pypdf
            self._initialized_methods.add('pypdf')
            if self._debug:
                logging.debug("pypdf available")
        except ImportError:
            if self._debug:
                logging.debug("pypdf not available")

        # Check for pdfminer
        try:
            from pdfminer import high_level
            self._initialized_methods.add('pdfminer')
            if self._debug:
                logging.debug("pdfminer available")
        except ImportError:
            if self._debug:
                logging.debug("pdfminer not available")
        
        # Log all initialized methods
        if self._debug:
            logging.debug(f"Initialized core methods: {', '.join(sorted(set(self._initialized_methods) & set(self.CORE_METHODS)))}")


    
    def _check_ocr_dependencies(self):
        """Check OCR-related dependencies separately"""
        logging.debug("Checking OCR dependencies for PDFExtract.")
        # OCR-related checks - these are more likely to cause issues
        try:
            import pytesseract
            import pdf2image
            
            # Verify tesseract works by checking the version
            try:
                version = pytesseract.get_tesseract_version()
                self._pytesseract = pytesseract
                self._pdf2image = pdf2image
                self._initialized_methods.add('tesseract')
                if self._debug:
                    logging.debug(f"Found Tesseract version: {version}")
            except Exception as e:
                if self._debug:
                    logging.debug(f"Tesseract check failed: {e}")
        except ImportError:
            if self._debug:
                logging.debug("pytesseract or pdf2image not available")

        # PaddleOCR check
        try:
            from paddleocr import PaddleOCR
            self._initialized_methods.add('paddleocr')
            if self._debug:
                logging.debug("PaddleOCR available")
        except ImportError:
            if self._debug:
                logging.debug("PaddleOCR not available")

        # Skip problematic OCR dependencies in normal operation
        # They'll be checked when actually needed
        if self._debug:
            try:
                self._safe_import('doctr')
                self._initialized_methods.add('doctr')
                logging.debug("doctr available")
            except Exception:
                pass

            try:
                self._safe_import('easyocr')
                self._initialized_methods.add('easyocr')
                logging.debug("easyocr available")
            except Exception:
                pass

            try:
                self._safe_import('kraken')
                self._initialized_methods.add('kraken')
                logging.debug("kraken available")
            except Exception:
                pass
    
    def _is_method_available(self, method: str) -> bool:
        """Check if extraction method is available with better logging"""
        # First check if it's already in initialized methods
        if method in self._initialized_methods:
            if self._debug:
                logging.debug(f"Method {method} is already initialized")
            return True

        # For Calibre, check on demand
        if method == 'calibre' and method not in self._initialized_methods:
            available = self._check_calibre_available()
            if available:
                self._initialized_methods.add('calibre')
                if self._debug:
                    logging.debug(f"Calibre checked and is available")
                return True
            if self._debug:
                logging.debug(f"Calibre checked and is NOT available")
            return False
        
        # Core methods
        if method in self.CORE_METHODS:
            try:
                if method == 'pymupdf':
                    import fitz
                    self._initialized_methods.add('pymupdf')
                    return True
                elif method == 'pdfplumber':
                    import pdfplumber
                    self._initialized_methods.add('pdfplumber')
                    return True
                elif method == 'pypdf':
                    import pypdf
                    self._initialized_methods.add('pypdf')
                    return True
                elif method == 'pdfminer':
                    from pdfminer import high_level
                    self._initialized_methods.add('pdfminer')
                    return True
            except ImportError:
                if self._debug:
                    logging.debug(f"Method {method} could not be imported")
                return False
        
        # For OCR methods
        if method in self.OCR_METHODS and method not in self._initialized_methods:
            try:
                if method == 'tesseract':
                    # Only check if tesseract binary exists
                    if not self._binaries.get('tesseract', False):
                        if self._debug:
                            logging.debug("Tesseract binary not found")
                        return False
                    
                    # Import required packages
                    import pytesseract
                    import pdf2image
                    
                    if pytesseract and pdf2image:
                        try:
                            version = pytesseract.get_tesseract_version()
                            self._initialized_methods.add('tesseract')
                            if self._debug:
                                logging.debug(f"Tesseract version: {version}")
                            return True
                        except Exception as e:
                            if self._debug:
                                logging.debug(f"Tesseract version check failed: {e}")
                            return False
                    return False
                    
                elif method == 'doctr':
                    # Check if doctr is available
                    if self._import_cache.is_available('doctr'):
                        self._initialized_methods.add('doctr')
                        return True
                    return False
                    
                elif method == 'paddleocr':
                    # Check if paddleocr is available
                    if self._import_cache.is_available('paddleocr'):
                        self._initialized_methods.add('paddleocr')
                        return True
                    return False
                
                elif method == 'easyocr':
                    # Check if easyocr is available
                    if self._import_cache.is_available('easyocr'):
                        self._initialized_methods.add('easyocr')
                        return True
                    return False
                    
                elif method == 'kraken':
                    # Skip if already in failed methods
                    if method in self._ocr_failed_methods:
                        return False
                    # Check if kraken is available
                    if self._import_cache.is_available('kraken'):
                        self._initialized_methods.add('kraken')
                        return True
                    return False
                    
                elif method == 'kraken_cli':
                    # Check if kraken CLI is available
                    kraken_bin = shutil.which('kraken')
                    if kraken_bin:
                        self._initialized_methods.add('kraken_cli')
                        return True
                    return False
                    
            except Exception as e:
                if self._debug:
                    logging.debug(f"Error checking OCR method {method}: {e}")
                return False
        
        # If we get here, the method is not available
        return False
        
    def set_password(self, password: str):
        """Set password for encrypted PDFs"""
        self._password = password

    @property
    def available_methods(self) -> Dict[str, bool]:
        """Lazy load available methods"""
        if self._available_methods is None:
            self._available_methods = {
                method: self._is_method_available(method)
                for method in self.TEXT_METHODS + self.TABLE_METHODS
            }
        return self._available_methods

    def extract_text(self, pdf_path: str, preferred_method: Optional[str] = None,
                ocr_method: Optional[str] = None, force_ocr: bool = False,
                progress_callback: Optional[Callable] = None, **kwargs) -> str:
        """
        Extract text with optimized fallback methods and improved error handling.
        
        Args:
            pdf_path: Path to PDF file
            preferred_method: Optional preferred extraction method
            ocr_method: Optional OCR method (only for OCR methods)
            force_ocr: Whether to force OCR even if text layer exists
            progress_callback: Optional callback for progress updates
            **kwargs: Additional extraction options
            
        Returns:
            str: Extracted text
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
        
        # Log clearly which method we're prioritizing
        if preferred_method:
            if self._debug:
                print(f"EXTRACT: Preferred extraction method: {preferred_method}")
        
        # Special case for calibre preference
        if preferred_method == 'calibre':
            if self._debug:
                print("EXTRACT: Calibre specifically requested, checking availability")
            if self._check_calibre_available():
                if self._debug:
                    print("EXTRACT: Calibre is available, trying it directly")
                try:
                    # Try calibre method directly
                    text = self.extract_with_calibre(pdf_path, progress_callback)
                    if text and len(text.strip()) > 100:  # Meaningful content check
                        if self._debug:
                            print(f"EXTRACT: Calibre succeeded with {len(text)} characters - returning immediately")
                        return text
                    else:
                        if self._debug:
                            print("EXTRACT: Calibre returned empty or too little text, trying fallbacks")
                except Exception as e:
                    if self._debug:
                        print(f"EXTRACT: Calibre extraction failed with error: {e}")
        
        # Start with core methods first, but respect preferred_method
        methods = list(self.CORE_METHODS)  # Make a copy
        
        # Define OCR methods, respecting ocr_method if provided
        ocr_methods = list(self.OCR_METHODS)  # Make a copy

        # If a preferred method is specified, ensure it's valid and initialized
        if preferred_method:
            # For calibre, always check directly (already tried above if it was 'calibre')
            if preferred_method != 'calibre':
                # Check if the method is available and initialize it if possible
                method_available = self._is_method_available(preferred_method)
                
                if method_available:
                    if self._debug:
                        print(f"EXTRACT: Method {preferred_method} is available")
                    
                    # If it's a core method, prioritize in core methods
                    if preferred_method in self.CORE_METHODS:
                        if preferred_method in methods:
                            methods.remove(preferred_method)
                        methods.insert(0, preferred_method)
                        if self._debug:
                            print(f"EXTRACT: Prioritizing core method: {preferred_method}")
                    # If it's an OCR method, prioritize in OCR methods
                    elif preferred_method in self.OCR_METHODS:
                        if preferred_method in ocr_methods:
                            ocr_methods.remove(preferred_method)
                        ocr_methods.insert(0, preferred_method)
                        if self._debug:
                            print(f"EXTRACT: Prioritizing OCR method: {preferred_method}")
                    else:
                        if self._debug:
                            print(f"EXTRACT: Method '{preferred_method}' is not recognized")
                else:
                    if self._debug:
                        print(f"EXTRACT: Method {preferred_method} is not available")

        # If force_ocr is True, skip non-OCR methods entirely
        if force_ocr:
            methods = []  # Skip standard extraction methods
            if self._debug:
                print("EXTRACT: Force OCR enabled - using only OCR methods")

        # If a specific OCR method is provided, prioritize it
        if ocr_method and ocr_method != 'auto':
            # Check if the OCR method is available
            ocr_available = self._is_method_available(ocr_method)
            
            if ocr_available:
                # Remove it from its current position if present
                if ocr_method in ocr_methods:
                    ocr_methods.remove(ocr_method)
                # Add it to the front of OCR methods
                ocr_methods.insert(0, ocr_method)
                if self._debug:
                    print(f"EXTRACT: Prioritizing OCR method: {ocr_method}")
            else:
                if self._debug:
                    print(f"EXTRACT: OCR method {ocr_method} is not available")
        
        # Log which methods we have initialized
        if self._debug:
            print(f"EXTRACT: Initialized methods: {', '.join(sorted(self._initialized_methods))}")
        
        # Log the methods we're going to try in order
        if self._debug:
            print(f"EXTRACT: Will try these core methods: {', '.join(methods)}")
            print(f"EXTRACT: Will try these OCR methods if needed: {', '.join(ocr_methods)}")
        
        text_parts = []
        current_method = None

        try:
            # Try core methods first
            for method in methods:
                # Skip calibre as we already tried it if it was preferred
                if method == 'calibre' and preferred_method == 'calibre':
                    if self._debug:
                        print("EXTRACT: Skipping calibre as we already tried it")
                    continue
                    
                # Skip methods that aren't initialized
                if method not in self._initialized_methods:
                    if self._debug:
                        print(f"EXTRACT: Skipping {method} - not initialized")
                    continue
                    
                # Skip methods that don't have an implementation
                if not hasattr(self, f'extract_with_{method}'):
                    if self._debug:
                        print(f"EXTRACT: Method {method} doesn't have an implementation - skipping")
                    continue
                    
                try:
                    current_method = method
                    if self._debug:
                        print(f"EXTRACT: Trying extraction with {method}...")
                    
                    # Signal start of extraction with this method
                    if progress_callback:
                        try:
                            progress_callback(0, method)
                        except Exception as e:
                            if self._debug:
                                print(f"EXTRACT: Progress callback error: {e}")
                    
                    extraction_func = getattr(self, f'extract_with_{method}')
                    
                    # Extract text with thorough error handling
                    try:
                        text = extraction_func(
                            pdf_path,
                            lambda n: progress_callback(n, method) if progress_callback else None
                        )
                    except KeyboardInterrupt:
                        if self._debug:
                            print("\nEXTRACT: Interrupted by user.")
                        raise
                    except Exception as extract_error:
                        if self._debug:
                            print(f"EXTRACT: Error during {method} extraction: {extract_error}")
                        # Try to ensure progress callback completion
                        if progress_callback:
                            try:
                                progress_callback(1, method)
                            except:
                                pass
                        continue  # Try next method
                    
                    if text and text.strip():
                        text_parts.append(text.strip())
                        quality = self._assess_text_quality(text)
                        if self._debug:
                            print(f"EXTRACT: Text quality with {method}: {quality:.2f}")
                        if quality > 0.7:  # Good enough quality
                            if self._debug:
                                print(f"EXTRACT: Got good quality text from {method}, stopping here")
                            break
                    else:
                        if self._debug:
                            print(f"EXTRACT: No text extracted with {method}")
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"EXTRACT: Error with {method}: {str(e)}")
                    continue
                finally:
                    # Ensure progress callback completion
                    if progress_callback and current_method == method:
                        try:
                            progress_callback(1, None)
                        except Exception as e:
                            print(f"EXTRACT: Progress callback completion error: {e}")

            # Only try OCR if we didn't get good quality text or force_ocr is enabled
            if force_ocr or (not text_parts and self._might_need_ocr(pdf_path)):
                if self._debug:
                    print(f"EXTRACT: {'Forcing OCR' if force_ocr else 'No good text extracted, trying OCR methods'}...")
                
                for method in ocr_methods:
                    # Skip methods that are in the failed methods set
                    if method in self._ocr_failed_methods:
                        if self._debug:
                            print(f"EXTRACT: Skipping {method} - previously failed")
                        continue
                    
                    # Pre-check if method is initialized or can be initialized
                    if not self._init_ocr(method):
                        if self._debug:
                            print(f"EXTRACT: Skipping {method} - initialization failed")
                        continue
                    
                    # Skip methods that don't have an implementation
                    if not hasattr(self, f'extract_with_{method}'):
                        if self._debug:
                            print(f"EXTRACT: Method {method} doesn't have an implementation - skipping")
                        continue
                    
                    try:
                        current_method = method
                        if self._debug:
                            print(f"EXTRACT: Trying OCR with {method}...")
                        
                        # Signal start of extraction with this method
                        if progress_callback:
                            try:
                                progress_callback(0, method)
                            except Exception as e:
                                if self._debug:
                                    print(f"EXTRACT: Progress callback error: {e}")
                        
                        # Verify tesseract is available if using tesseract
                        if method == 'tesseract':
                            tesseract_available = (
                                self._binaries.get('tesseract', False) and 
                                hasattr(self, '_pytesseract') and 
                                self._pytesseract is not None
                            )
                            if not tesseract_available:
                                if self._debug:
                                    print("EXTRACT: Tesseract not properly available, skipping")
                                self._ocr_failed_methods.add('tesseract')
                                continue
                        
                        extraction_func = getattr(self, f'extract_with_{method}')
                        
                        # Extract text with thorough error handling
                        try:
                            text = extraction_func(
                                pdf_path,
                                lambda n: progress_callback(n, method) if progress_callback else None
                            )
                        except KeyboardInterrupt:
                            print("\nEXTRACT: OCR interrupted by user.")
                            raise
                        except Exception as extract_error:
                            print(f"EXTRACT: Error during {method} OCR: {extract_error}")
                            # Mark this method as failed
                            self._ocr_failed_methods.add(method)
                            # Try to ensure progress callback completion
                            if progress_callback:
                                try:
                                    progress_callback(1, method)
                                except:
                                    pass
                            continue  # Try next OCR method
                        
                        if text and text.strip():
                            text_parts.append(text.strip())
                            if self._debug:
                                print(f"EXTRACT: Successfully extracted text using {method}")
                            break  # One successful OCR method is enough
                        else:
                            print(f"EXTRACT: No text extracted with {method}")
                            
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"EXTRACT: Error with {method}: {str(e)}")
                        self._ocr_failed_methods.add(method)
                        continue
                    finally:
                        # Ensure progress callback completion
                        if progress_callback and current_method == method:
                            try:
                                progress_callback(1, None)
                            except Exception as e:
                                print(f"EXTRACT: Progress callback completion error: {e}")

        except Exception as e:
            print(f"EXTRACT: Unexpected error in extraction: {str(e)}")
        finally:
            self._cleanup()

        # Check if we extracted any text
        if not text_parts:
            print("EXTRACT: No text extracted with any method")
            return ""
        else:
            total_chars = sum(len(part) for part in text_parts)
            if self._debug:
                print(f"EXTRACT: Successfully extracted {total_chars} characters using {current_method}")

        return "\n\n".join(text_parts).strip()

    def _might_need_ocr(self, pdf_path: str) -> bool:
        """Quick check if PDF might need OCR"""
        try:
            if 'pymupdf' in self._initialized_methods:
                import fitz
                doc = fitz.open(pdf_path)
                try:
                    # Check first 3 pages or all pages if less
                    pages_to_check = min(3, len(doc))
                    total_text = 0
                    
                    for i in range(pages_to_check):
                        text = doc[i].get_text()
                        total_text += len(text.strip())
                        
                    # If average text per page is very low, might need OCR
                    return (total_text / pages_to_check) < 100
                    
                finally:
                    doc.close()
        except Exception:
            pass
            
        return True  # Default to yes if we can't check

    def _assess_text_quality(self, text: str) -> float:
        """Assess extracted text quality"""
        if not text:
            return 0.0
            
        score = 0.0
        text = text.strip()
        
        # Basic text characteristics
        words = text.split()
        if not words:
            return 0.0
            
        # Check word lengths
        avg_word_len = sum(len(w) for w in words) / len(words)
        if 3 <= avg_word_len <= 10:
            score += 0.3
            
        # Check for reasonable text structure
        lines = text.split('\n')
        if lines:
            # Check line lengths
            avg_line_len = sum(len(l.strip()) for l in lines) / len(lines)
            if 30 <= avg_line_len <= 100:
                score += 0.3
                
        # Check for paragraph structure
        if '\n\n' in text:
            score += 0.2
            
        # Check character distribution
        alpha_count = sum(c.isalpha() for c in text)
        if len(text) > 0:
            alpha_ratio = alpha_count / len(text)
            if 0.6 <= alpha_ratio <= 0.9:
                score += 0.2
                
        return min(max(score, 0.0), 1.0)

    def _is_scanned_pdf(self, pdf_path: str) -> bool:
        """Quick check if PDF appears to be scanned"""
        try:
            # Try quick text extraction with pymupdf
            import fitz
            doc = fitz.open(pdf_path)
            first_page = doc[0]
            text = first_page.get_text()
            doc.close()
            
            # If first page has very little text, likely scanned
            return len(text.strip()) < 100
            
        except Exception:
            return False

    def _needs_further_processing(self, text: str) -> bool:
        """Check if text needs additional processing methods"""
        # Check text quality
        words = text.split()
        if len(words) < 100:  # Too short - try other methods
            return True
            
        # Check for common OCR/extraction artifacts
        artifacts = ['�', '□', '■', '○', '●', '¶']
        artifact_count = sum(text.count(a) for a in artifacts)
        if artifact_count > len(text) * 0.01:  # More than 1% artifacts
            return True
            
        # Check for layout issues
        lines = text.splitlines()
        if not lines:
            return True
            
        # Check for suspiciously short lines
        short_lines = sum(1 for line in lines if len(line.strip()) < 20)
        if short_lines > len(lines) * 0.5:  # More than 50% short lines
            return True
            
        # Check for reasonable paragraph structure
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:  # No clear paragraph breaks
            return True
            
        return False

    def _cleanup(self):
        """Clean up resources"""
        if self._current_doc:
            try:
                self._current_doc.close()
            except:
                pass
            self._current_doc = None
            
        import gc
        gc.collect()

    def _clear_memory(self):
        """Clear memory after processing"""
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
    def extract_with_pymupdf(self, pdf_path: str, progress_callback=None) -> str:
        try:
            fitz = self._import_cache.import_module('fitz')  # Use ImportCache
            text_parts = []
            
            doc = fitz.open(pdf_path)
            self._current_doc = doc
            
            if doc.needs_pass:
                if not self._password or not doc.authenticate(self._password):
                    raise ValueError("Invalid PDF password")
            
            total_pages = len(doc)
            with tqdm(total=total_pages, desc="PyMuPDF extraction", unit="pages") as pbar:
                for page_num in range(total_pages):
                    try:
                        page = doc[page_num]
                        # Try different extraction strategies
                        page_text = page.get_text("text", sort=True)
                        if not page_text.strip():
                            # Fallback to dict extraction for complex layouts
                            page_text = page.get_text("dict")
                            if isinstance(page_text, dict):
                                page_text = self._process_text_dict(page_text)
                        
                        if page_text.strip():
                            text_parts.append(page_text.strip())
                        
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(1)
                            
                    except Exception as e:
                        logging.debug(f"Page {page_num + 1} extraction failed: {e}")
                        continue
                    finally:
                        page = None
            
            return "\n\n".join(text_parts)
            
        finally:
            if self._current_doc:
                try:
                    self._current_doc.close()
                except:
                    pass
                self._current_doc = None

    def extract_with_calibre(self, pdf_path: str, progress_callback=None) -> str:
        """
        Extract text using Calibre's ebook-converter with tracked process for Ctrl+C handling
        """
        import subprocess
        import tempfile
        import os
        import shutil
        
        if self._debug:
            print("CALIBRE: Starting extraction")
        
        # Find calibre binary
        calibre_bin = None
        
        # Check binary_paths
        if hasattr(self, '_binary_paths') and self._binary_paths:
            calibre_bin = self._binary_paths.get('ebook-converter')
            if calibre_bin:
                if self._debug:
                    print(f"CALIBRE: Using binary from _binary_paths: {calibre_bin}")
        
        # Check PATH
        if not calibre_bin:
            for binary_name in ['ebook-converter', 'ebook-convert']:
                path = shutil.which(binary_name)
                if path:
                    calibre_bin = path
                    if self._debug:
                        print(f"CALIBRE: Found in PATH: {calibre_bin}")
                    break
        
        if not calibre_bin:
            if self._debug:
                print("CALIBRE: Binary not found!")
            return ""
        
        # Create temporary output file
        temp_output = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                temp_output = tmp.name
        
            if self._debug:
                print(f"CALIBRE: Running command: {calibre_bin} {pdf_path} {temp_output}")
            
            # Use the tracked process function instead of subprocess.run
            result = run_process(
                [calibre_bin, pdf_path, temp_output],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Check result
            if result.returncode != 0:
                print(f"CALIBRE ERROR: Command failed with code {result.returncode}")
                if self._debug:
                    print(f"CALIBRE STDERR: {result.stderr}")
                return ""
            
            # Read output file
            if os.path.exists(temp_output):
                with open(temp_output, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                    if self._debug:
                        print(f"CALIBRE: Successfully extracted {len(text)} characters")
                    return text
            else:
                print(f"CALIBRE ERROR: Output file not created: {temp_output}")
                return ""
        
        except Exception as e:
            print(f"CALIBRE ERROR: {str(e)}")
            return ""
        finally:
            # Clean up
            if temp_output and os.path.exists(temp_output):
                try:
                    os.unlink(temp_output)
                except:
                    pass

    def _process_text_dict(self, text_dict: Dict) -> str:
        """Process PyMuPDF dict format text"""
        text_parts = []
        try:
            for block in text_dict.get('blocks', []):
                if 'lines' in block:
                    for line in block['lines']:
                        line_text = ' '.join(
                            span.get('text', '') 
                            for span in line.get('spans', [])
                        )
                        if line_text.strip():
                            text_parts.append(line_text)
        except Exception as e:
            logging.debug(f"Text dict processing failed: {e}")
        return '\n'.join(text_parts)

    def extract_with_pdfplumber(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using pdfplumber with layout preservation and progress bar"""
        pdfplumber = self._import_cache.import_module('pdfplumber')
        text_parts = []
        
        try:
            with pdfplumber.open(pdf_path, password=self._password) as pdf:
                self._current_doc = pdf
                total_pages = len(pdf.pages)
                
                with tqdm(total=total_pages, desc="pdfplumber extraction", unit="pages") as pbar:
                    for page in pdf.pages:
                        try:
                            # Extract with layout settings
                            words = page.extract_words(
                                keep_blank_chars=True,
                                use_text_flow=True,
                                horizontal_ltr=True
                            )
                            
                            if words:
                                # Group words into lines
                                lines = self._group_words_into_lines(words)
                                text_parts.append('\n'.join(' '.join(line) for line in lines))
                            else:
                                # Fallback to basic extraction
                                text = page.extract_text()
                                if text.strip():
                                    text_parts.append(text.strip())
                            
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                                
                        except Exception as e:
                            logging.debug(f"Page extraction failed: {e}")
                            continue
            
            return '\n\n'.join(text_parts)
            
        finally:
            self._current_doc = None
            
    def _group_words_into_lines(self, words: List[Dict]) -> List[List[str]]:
        """Group words into lines based on positions"""
        if not words:
            return []
        
        # Sort by top position and x position
        words.sort(key=lambda w: (round(w['top']), w['x0']))
        
        lines = []
        current_line = []
        last_top = round(words[0]['top'])
        
        for word in words:
            current_top = round(word['top'])
            if abs(current_top - last_top) > 3:  # Line break threshold
                if current_line:
                    lines.append(current_line)
                current_line = []
                last_top = current_top
            current_line.append(word['text'])
        
        if current_line:
            lines.append(current_line)
        
        return lines

    def extract_with_pypdf(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using pypdf with encryption support and progress bar"""
        pypdf = self._import_cache.import_module('pypdf')
        
        # Disable debug logging from pypdf
        logging.getLogger('pypdf').setLevel(logging.WARNING)
        
        text_parts = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                self._current_doc = reader
                
                if reader.is_encrypted:
                    if not reader.decrypt(self._password or ""):
                        raise ValueError("PDF is encrypted and requires a valid password")
                
                total_pages = len(reader.pages)
                with tqdm(total=total_pages, desc="pypdf extraction", unit="pages") as pbar:
                    for page in reader.pages:
                        try:
                            text = page.extract_text()
                            if text.strip():
                                text_parts.append(text.strip())
                            
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                                
                        except Exception as e:
                            logging.debug(f"Page extraction failed: {e}")
                            continue
            
            return '\n\n'.join(text_parts)
            
        finally:
            self._current_doc = None

    def extract_with_pdfminer(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using pdfminer with layout analysis"""
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        from io import StringIO
        
        try:
            output = StringIO()
            with open(pdf_path, 'rb') as file:
                # Configure layout parameters
                laparams = LAParams(
                    line_margin=0.5,
                    word_margin=0.1,
                    char_margin=2.0,
                    boxes_flow=0.5,
                    detect_vertical=True
                )
                
                extract_text_to_fp(
                    file, output,
                    laparams=laparams,
                    password=self._password,
                    codec='utf-8'
                )
                
                text = output.getvalue()
                
                # Post-process the extracted text
                if text.strip():
                    lines = text.splitlines()
                    processed_lines = []
                    current_para = []
                    
                    for line in lines:
                        line = line.strip()
                        if not line and current_para:
                            processed_lines.append(' '.join(current_para))
                            current_para = []
                        elif line:
                            current_para.append(line)
                    
                    if current_para:
                        processed_lines.append(' '.join(current_para))
                    
                    return '\n\n'.join(processed_lines)
            
            return ""
            
        finally:
            if 'output' in locals():
                output.close()
    
    def extract_with_doctr(self, pdf_path: str, progress_callback=None) -> str:
        """
        Extract text using DocTR with enhanced model configuration and text block handling.
        Optimized for speed - stops after first successful extraction.
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback for progress updates
            
        Returns:
            str: Extracted text
        """
        if not self._init_ocr('doctr'):
            logging.error("DocTR initialization failed")
            return ""
            
        try:
            # Import required packages with better error handling
            try:
                pdf2image = self._import_cache.import_module('pdf2image')
                import numpy as np
                import io
                from PIL import Image
                import traceback
            except ImportError as e:
                logging.error(f"Failed to import required packages for DocTR: {e}")
                return ""
            
            # Get DocTR
            doctr = self._doctr
            
            text_parts = []
            
            # Configure PDF conversion settings for better OCR results
            conversion_settings = {
                'dpi': 300,              # Default DPI
                'thread_count': 1,       # Single thread for stability
                'grayscale': False,      # DocTR works better with color images
                'size': (None, None),    # Default to original size
                'use_cropbox': True,     # Use cropbox for better results
                'strict': False          # Continue even with errors
            }
            
            logging.info(f"Starting DocTR extraction for {pdf_path}")
            
            # Try multiple DPI settings if first attempt fails
            dpi_options = [300, 600]  # Reduced to just two options for speed
            
            for dpi_attempt, dpi in enumerate(dpi_options):
                if dpi_attempt > 0:
                    logging.info(f"Trying PDF conversion with increased DPI: {dpi}")
                    
                # Update DPI for this attempt
                conversion_settings['dpi'] = dpi
                
                # Convert PDF to images first
                with tqdm(desc=f"Converting PDF to images for DocTR (DPI: {dpi})", unit="page") as pbar:
                    try:
                        # Convert PDF pages to PIL images
                        images = pdf2image.convert_from_path(
                            pdf_path,
                            **conversion_settings
                        )
                        pbar.update(len(images))
                        logging.info(f"Successfully converted {len(images)} pages from PDF (DPI: {dpi})")
                    except Exception as e:
                        logging.error(f"PDF to image conversion failed at DPI {dpi}: {e}")
                        if dpi_attempt < len(dpi_options) - 1:
                            continue  # Try next DPI
                        else:
                            return ""  # All DPI options failed
                
                # Optimize model configurations - fastest first
                model_configs = [
                    # Default model - usually fastest and effective enough
                    {
                        "name": "Default model",
                        "det_arch": "db_resnet50",
                        "reco_arch": "crnn_vgg16_bn",
                        "assume_straight_pages": True,
                        "straighten_pages": False
                    },
                    # Only use if first model doesn't work
                    {
                        "name": "Simple model",
                        "det_arch": "db_resnet34",
                        "reco_arch": "crnn_mobilenet_v3_small",
                        "assume_straight_pages": True,
                        "straighten_pages": False
                    },
                    # Last resort - slowest but most accurate
                    {
                        "name": "Alternative detection model",
                        "det_arch": "linknet_resnet18",
                        "reco_arch": "crnn_vgg16_bn",
                        "assume_straight_pages": False,
                        "straighten_pages": True
                    }
                ]
                
                # Track if we've succeeded with any model
                extraction_success = False
                
                # Try each model configuration until we find one that works
                for model_idx, model_config in enumerate(model_configs):
                    # Skip slower models if we already have text
                    if extraction_success:
                        logging.info(f"Skipping {model_config['name']} since text was already extracted successfully")
                        continue
                    
                    model_text_parts = []  # Store text from this model configuration
                    
                    logging.info(f"Trying DocTR with {model_config['name']} configuration")
                    
                    # Initialize DocTR predictor with this configuration
                    try:
                        logging.info(f"Initializing DocTR predictor with {model_config['det_arch']} detection model")
                        self._doctr_predictor = doctr.models.ocr_predictor(
                            det_arch=model_config['det_arch'],
                            reco_arch=model_config['reco_arch'],
                            pretrained=True,
                            assume_straight_pages=model_config['assume_straight_pages'],
                            straighten_pages=model_config['straighten_pages']
                        )
                        logging.info("DocTR OCR predictor initialized with custom configuration")
                    except Exception as e:
                        logging.error(f"Failed to initialize DocTR predictor with custom config: {e}")
                        continue  # Try next configuration
                    
                    # Make fresh copy of images for this model
                    model_images = []
                    for img in images:
                        img_copy = img.copy()
                        model_images.append(img_copy)
                    
                    # Process each image with extensive error handling
                    with tqdm(total=len(model_images), desc="DocTR processing", unit="page") as pbar:
                        page_success_count = 0
                        
                        for i, image in enumerate(model_images):
                            try:
                                # Convert PIL image to numpy array
                                img_np = np.array(image)
                                
                                # Try different preprocessing techniques
                                processed_images = [
                                    img_np,  # Original image
                                    self._apply_contrast_enhancement(img_np),  # Enhanced contrast
                                    self._apply_binarization(img_np)  # Binary image
                                ]
                                
                                # Flag to track if we extracted text from this page
                                page_success = False
                                page_text = ""
                                
                                # Try each preprocessing technique
                                for img_variant_idx, img_variant in enumerate(processed_images):
                                    # Stop if we already got text from this page
                                    if page_success:
                                        break
                                        
                                    variant_name = ["Original", "Enhanced", "Binary"][img_variant_idx]
                                    logging.debug(f"Trying {variant_name} image for page {i}")
                                    
                                    try:
                                        # Process with DocTR
                                        result = self._doctr_predictor([img_variant])
                                        
                                        # Check if we got any results
                                        if result and hasattr(result, 'pages') and result.pages:
                                            page = result.pages[0]
                                            
                                            # Force block creation for difficult documents
                                            if not hasattr(page, 'blocks') or len(page.blocks) == 0:
                                                logging.debug(f"No blocks detected, applying forced block detection")
                                                
                                                # Attempt to extract text directly from the image using forced detection
                                                current_page_text = self._extract_text_with_forced_detection(img_variant, i)
                                                
                                                if current_page_text and current_page_text.strip():
                                                    page_text = current_page_text
                                                    page_success = True
                                                    break  # Success with this preprocessing technique
                                            else:
                                                # Normal extraction if blocks are detected
                                                current_page_text = self._extract_doctr_text_from_image(img_variant, i)
                                                
                                                if current_page_text and current_page_text.strip():
                                                    page_text = current_page_text
                                                    page_success = True
                                                    break  # Success with this preprocessing technique
                                    except Exception as variant_e:
                                        logging.debug(f"Error processing {variant_name} variant: {variant_e}")
                                
                                # Add page text if successful
                                if page_success and page_text:
                                    model_text_parts.append(page_text)
                                    page_success_count += 1
                                    logging.debug(f"Successfully extracted text from page {i}")
                                
                                # Update progress
                                pbar.update(1)
                                if progress_callback:
                                    progress_callback(1)
                            except Exception as e:
                                logging.error(f"DocTR failed on page {i}: {e}")
                                logging.debug(f"Traceback: {traceback.format_exc()}")
                            finally:
                                try:
                                    image.close()
                                except:
                                    pass
                    
                    logging.info(f"DocTR successful pages with {model_config['name']}: {page_success_count}/{len(model_images)}")
                    
                    # Check if this model configuration worked well enough
                    if page_success_count > 0:
                        extraction_success = True
                        text_parts = model_text_parts
                        logging.info(f"Successfully extracted text with {model_config['name']}. Skipping remaining models.")
                        break  # Exit model loop - we found a working model
                
                # If we got text with this DPI, stop trying more DPIs
                if extraction_success:
                    break
                    
            if text_parts:
                final_text = "\n\n".join(text_parts)
                logging.info(f"DocTR extraction completed with {len(final_text)} characters")
                return final_text
            else:
                logging.warning("DocTR: No text extracted from any page with any configuration")
                return ""
        
        except Exception as e:
            logging.error(f"DocTR extraction failed: {e}")
            logging.debug(f"Traceback: {traceback.format_exc()}")
            return ""
        finally:
            # Clean up resources
            if 'images' in locals() and images:
                for img in images:
                    try:
                        img.close()
                    except:
                        pass
            
            # Clear GPU memory
            self._clear_gpu_memory()
    
    def extract_with_doctr_tryall(self, pdf_path: str, progress_callback=None) -> str:
        """
        Extract text using DocTR with enhanced model configuration and text block handling.
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback for progress updates
            
        Returns:
            str: Extracted text
        """
        if not self._init_ocr('doctr'):
            logging.error("DocTR initialization failed")
            return ""
            
        try:
            # Import required packages with better error handling
            try:
                pdf2image = self._import_cache.import_module('pdf2image')
                import numpy as np
                import io
                from PIL import Image
                import traceback
            except ImportError as e:
                logging.error(f"Failed to import required packages for DocTR: {e}")
                return ""
            
            # Get DocTR
            doctr = self._doctr
            
            text_parts = []
            all_extracted_text = []  # Store text from all model configurations
            
            # Configure PDF conversion settings for better OCR results
            conversion_settings = {
                'dpi': 300,              # Default DPI
                'thread_count': 1,       # Single thread for stability
                'grayscale': False,      # DocTR works better with color images
                'size': (None, None),    # Default to original size
                'use_cropbox': True,     # Use cropbox for better results
                'strict': False          # Continue even with errors
            }
            
            logging.info(f"Starting DocTR extraction for {pdf_path}")
            
            # Try multiple DPI settings if first attempt fails
            dpi_options = [300, 400, 600]
            
            for dpi_attempt, dpi in enumerate(dpi_options):
                if dpi_attempt > 0:
                    logging.info(f"Trying PDF conversion with increased DPI: {dpi}")
                    
                # Update DPI for this attempt
                conversion_settings['dpi'] = dpi
                
                # Convert PDF to images first
                with tqdm(desc=f"Converting PDF to images for DocTR (DPI: {dpi})", unit="page") as pbar:
                    try:
                        # Convert PDF pages to PIL images
                        images = pdf2image.convert_from_path(
                            pdf_path,
                            **conversion_settings
                        )
                        pbar.update(len(images))
                        logging.info(f"Successfully converted {len(images)} pages from PDF (DPI: {dpi})")
                    except Exception as e:
                        logging.error(f"PDF to image conversion failed at DPI {dpi}: {e}")
                        if dpi_attempt < len(dpi_options) - 1:
                            continue  # Try next DPI
                        else:
                            return ""  # All DPI options failed
                
                # Make copies of all images for each model configuration
                # This prevents the "closed image" error when switching between models
                image_copies = []
                for img in images:
                    img_copy = img.copy()
                    image_copies.append(img_copy)
                
                # Try multiple model configurations for detection
                model_configs = [
                    # Default model
                    {
                        "name": "Default model",
                        "det_arch": "db_resnet50",
                        "reco_arch": "crnn_vgg16_bn",
                        "assume_straight_pages": True,
                        "straighten_pages": False
                    },
                    # Alternative model for harder documents
                    {
                        "name": "Alternative detection model",
                        "det_arch": "linknet_resnet18",
                        "reco_arch": "crnn_vgg16_bn",
                        "assume_straight_pages": False,
                        "straighten_pages": True
                    },
                    # Third model focused on handling unusual documents
                    {
                        "name": "Simple model",
                        "det_arch": "db_resnet34",
                        "reco_arch": "crnn_mobilenet_v3_small",
                        "assume_straight_pages": True,
                        "straighten_pages": False
                    }
                ]
                
                # Try each model configuration until we find one that works
                for model_idx, model_config in enumerate(model_configs):
                    model_text_parts = []  # Store text from this model configuration
                    
                    logging.info(f"Trying DocTR with {model_config['name']} configuration")
                    
                    # Initialize DocTR predictor with this configuration
                    try:
                        logging.info(f"Initializing DocTR predictor with {model_config['det_arch']} detection model")
                        self._doctr_predictor = doctr.models.ocr_predictor(
                            det_arch=model_config['det_arch'],
                            reco_arch=model_config['reco_arch'],
                            pretrained=True,
                            assume_straight_pages=model_config['assume_straight_pages'],
                            straighten_pages=model_config['straighten_pages']
                        )
                        logging.info("DocTR OCR predictor initialized with custom configuration")
                    except Exception as e:
                        logging.error(f"Failed to initialize DocTR predictor with custom config: {e}")
                        continue  # Try next configuration
                    
                    # Make fresh copy of images for this model
                    model_images = []
                    if model_idx == 0:
                        # For first model, use original copies
                        model_images = image_copies
                    else:
                        # For subsequent models, make new copies to avoid closed image errors
                        model_images = []
                        for img in images:
                            model_images.append(img.copy())
                    
                    # Process each image with extensive error handling
                    with tqdm(total=len(model_images), desc="DocTR processing", unit="page") as pbar:
                        page_success_count = 0
                        
                        for i, image in enumerate(model_images):
                            try:
                                # Convert PIL image to numpy array
                                img_np = np.array(image)
                                
                                # Try different preprocessing techniques
                                processed_images = [
                                    img_np,  # Original image
                                    self._apply_contrast_enhancement(img_np),  # Enhanced contrast
                                    self._apply_binarization(img_np)  # Binary image
                                ]
                                
                                # Flag to track if we extracted text from this page
                                page_success = False
                                page_text = ""
                                
                                # Try each preprocessing technique
                                for img_variant_idx, img_variant in enumerate(processed_images):
                                    variant_name = ["Original", "Enhanced", "Binary"][img_variant_idx]
                                    logging.debug(f"Trying {variant_name} image for page {i}")
                                    
                                    try:
                                        # Process with DocTR
                                        result = self._doctr_predictor([img_variant])
                                        
                                        # Check if we got any results
                                        if result and hasattr(result, 'pages') and result.pages:
                                            page = result.pages[0]
                                            
                                            # Force block creation for difficult documents
                                            if not hasattr(page, 'blocks') or len(page.blocks) == 0:
                                                logging.debug(f"No blocks detected, applying forced block detection")
                                                
                                                # Attempt to extract text directly from the image using forced detection
                                                current_page_text = self._extract_text_with_forced_detection(img_variant, i)
                                                
                                                if current_page_text and current_page_text.strip():
                                                    page_text = current_page_text
                                                    page_success = True
                                                    break  # Success with this preprocessing technique
                                            else:
                                                # Normal extraction if blocks are detected
                                                current_page_text = self._extract_doctr_text_from_image(img_variant, i)
                                                
                                                if current_page_text and current_page_text.strip():
                                                    page_text = current_page_text
                                                    page_success = True
                                                    break  # Success with this preprocessing technique
                                    except Exception as variant_e:
                                        logging.debug(f"Error processing {variant_name} variant: {variant_e}")
                                
                                # Add page text if successful
                                if page_success and page_text:
                                    model_text_parts.append(page_text)
                                    page_success_count += 1
                                    logging.debug(f"Successfully extracted text from page {i}")
                                
                                # Update progress
                                pbar.update(1)
                                if progress_callback:
                                    progress_callback(1)
                            except Exception as e:
                                logging.error(f"DocTR failed on page {i}: {e}")
                                logging.debug(f"Traceback: {traceback.format_exc()}")
                            finally:
                                pass  # Don't close images here - we'll clean them up later
                    
                    logging.info(f"DocTR successful pages with {model_config['name']}: {page_success_count}/{len(model_images)}")
                    
                    # Save text from this model configuration
                    if model_text_parts:
                        all_extracted_text.append({
                            'model': model_config['name'],
                            'text': "\n\n".join(model_text_parts),
                            'success_count': page_success_count,
                            'total_pages': len(model_images)
                        })
                
                # Clean up all image copies
                for img in image_copies:
                    try:
                        img.close()
                    except:
                        pass
                for img in model_images:
                    try:
                        img.close()
                    except:
                        pass
                
                # If we got text from any model with this DPI, stop trying more DPIs
                if all_extracted_text:
                    break
                    
            # Choose the best result from all model configurations
            if all_extracted_text:
                # Sort by success count (most successful pages first)
                all_extracted_text.sort(key=lambda x: x['success_count'], reverse=True)
                best_result = all_extracted_text[0]
                
                logging.info(f"Best DocTR result: {best_result['model']} with {best_result['success_count']}/{best_result['total_pages']} successful pages")
                return best_result['text']
            else:
                logging.warning("DocTR: No text extracted from any page with any configuration")
                return ""
        
        except Exception as e:
            logging.error(f"DocTR extraction failed: {e}")
            logging.debug(f"Traceback: {traceback.format_exc()}")
            return ""
        finally:
            # Clean up resources
            if 'images' in locals() and images:
                for img in images:
                    try:
                        img.close()
                    except:
                        pass
            
            # Clear GPU memory
            self._clear_gpu_memory()

    def _apply_contrast_enhancement(self, image_np):
        """Apply CLAHE contrast enhancement"""
        try:
            import cv2
            import numpy as np
            
            # Convert to LAB color space
            lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l_channel)
            
            # Merge enhanced L-channel with original A and B channels
            enhanced_lab = cv2.merge((enhanced_l, a, b))
            
            # Convert back to RGB
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            return enhanced_rgb
        except Exception as e:
            logging.debug(f"Contrast enhancement failed: {e}")
            return image_np

    def _apply_binarization(self, image_np):
        """Apply adaptive thresholding for binarization"""
        try:
            import cv2
            import numpy as np
            
            # Convert to grayscale
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
                
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to RGB for DocTR
            rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            
            return rgb
        except Exception as e:
            logging.debug(f"Binarization failed: {e}")
            return image_np

    def _extract_text_with_forced_detection(self, image_np, page_idx):
        """
        Extract text by forcing block detection when DocTR fails to detect blocks.
        This is a fallback method for difficult documents.
        """
        try:
            import cv2
            import numpy as np
            
            # 1. Prepare the image
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
                
            # 2. Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 3. Find contours - these will be potential text blocks
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 4. Filter contours by size
            height, width = gray.shape
            min_area = height * width * 0.0001  # Minimum area threshold
            max_area = height * width * 0.5     # Maximum area threshold
            
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    valid_contours.append(contour)
            
            logging.debug(f"Found {len(valid_contours)} potential text blocks after filtering")
            
            if not valid_contours:
                logging.debug("No valid contours found for forced detection")
                return ""
                
            # 5. Extract and process each potential text block
            text_parts = []
            
            for i, contour in enumerate(valid_contours):
                try:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Extract region
                    region = image_np[y:y+h, x:x+w]
                    
                    # Skip regions that are too small
                    if region.shape[0] < 10 or region.shape[1] < 10:
                        continue
                    
                    # Process the region with DocTR
                    result = self._doctr_predictor([region])
                    
                    if result and hasattr(result, 'pages') and result.pages:
                        page = result.pages[0]
                        
                        # Extract text from page
                        if hasattr(page, 'export') and callable(page.export):
                            export_data = page.export()
                            if isinstance(export_data, dict) and "text" in export_data:
                                block_text = export_data["text"].strip()
                                if block_text:
                                    text_parts.append(block_text)
                                    logging.debug(f"Extracted text from forced block {i}: {block_text[:30]}...")
                except Exception as e:
                    logging.debug(f"Error processing forced block {i}: {e}")
                    continue
            
            # 6. Combine all text
            if text_parts:
                return "\n".join(text_parts)
            
            return ""
            
        except Exception as e:
            logging.error(f"Forced text detection failed: {e}")
            return ""
        
    def extract_with_paddleocr(self, pdf_path: str, progress_callback=None) -> str:
        """
        Extract text using PaddleOCR with enhanced error handling for printed text.
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback for progress updates
            
        Returns:
            str: Extracted text
        """
        if not self._init_ocr('paddleocr'):
            logging.error("PaddleOCR initialization failed")
            return ""
            
        try:
            # Import required packages
            pdf2image = self._import_cache.import_module('pdf2image')
            from PIL import Image
            import numpy as np
            import io
            
            # Reference to PaddleOCR instance
            paddle_ocr = self._paddleocr
            text_parts = []
            images = None
            
            try:
                # Convert PDF to images with lower DPI for better compatibility
                with tqdm(desc="Converting PDF to images for PaddleOCR", unit="page") as pbar:
                    try:
                        images = pdf2image.convert_from_path(
                            pdf_path,
                            dpi=200,  # Lower DPI for better compatibility
                            thread_count=1,
                            grayscale=False,
                            fmt="jpeg"  # Use JPEG for better compatibility
                        )
                        pbar.update(len(images))
                        logging.info(f"Successfully converted {len(images)} pages for PaddleOCR")
                    except Exception as e:
                        logging.error(f"PDF to image conversion failed: {e}")
                        return ""
                    
                # Process each image with PaddleOCR
                with tqdm(total=len(images), desc="PaddleOCR processing", unit="page") as pbar:
                    for i, image in enumerate(images, 1):
                        try:
                            # Save image to memory buffer to ensure proper format
                            img_buffer = io.BytesIO()
                            image.save(img_buffer, format="JPEG")
                            img_buffer.seek(0)
                            
                            # Load from buffer to ensure clean image
                            pil_img = Image.open(img_buffer)
                            
                            # Debug info about image
                            logging.debug(f"Processing image for page {i}: size={pil_img.size}, mode={pil_img.mode}")
                            
                            # Process with PaddleOCR using the direct image object
                            try:
                                # First try with direct PIL image
                                result = paddle_ocr.ocr(pil_img, cls=True)
                                logging.debug(f"PaddleOCR result type: {type(result)}")
                                if result:
                                    logging.debug(f"Result length: {len(result)}")
                            except Exception as direct_error:
                                logging.debug(f"Direct PIL processing failed: {direct_error}")
                                # Fall back to numpy array
                                try:
                                    img_array = np.array(pil_img)
                                    result = paddle_ocr.ocr(img_array, cls=True)
                                except Exception as numpy_error:
                                    logging.error(f"Numpy array processing also failed: {numpy_error}")
                                    pbar.update(1)
                                    if progress_callback:
                                        progress_callback(1)
                                    continue
                            
                            # Extract text with detailed debugging
                            page_text = []
                            
                            # Handle the result with extensive error checking
                            if result is not None:
                                logging.debug(f"Result is not None, type: {type(result)}")
                                
                                # Check if result is list of pages
                                if isinstance(result, list):
                                    for idx, page_result in enumerate(result):
                                        logging.debug(f"Processing page result {idx}, type: {type(page_result)}")
                                        
                                        if page_result is not None and isinstance(page_result, list):
                                            for line_idx, line in enumerate(page_result):
                                                logging.debug(f"Processing line {line_idx}, type: {type(line)}")
                                                
                                                # Try different formats
                                                try:
                                                    # Format: [[points], (text, confidence)]
                                                    if (isinstance(line, list) and len(line) == 2 and 
                                                        isinstance(line[1], tuple) and len(line[1]) == 2):
                                                        
                                                        text, confidence = line[1]
                                                        logging.debug(f"Found text: '{text}', confidence: {confidence}")
                                                        
                                                        if confidence > 0.5:  # Use a lower threshold for better recall
                                                            page_text.append(text)
                                                            
                                                    # Format: [points, [text, confidence]]
                                                    elif (isinstance(line, list) and len(line) == 2 and 
                                                        isinstance(line[1], list) and len(line[1]) == 2):
                                                        
                                                        text, confidence = line[1]
                                                        logging.debug(f"Found text (alternate format): '{text}', confidence: {confidence}")
                                                        
                                                        if confidence > 0.5:
                                                            page_text.append(text)
                                                            
                                                    # Legacy format: [text, confidence]
                                                    elif (isinstance(line, list) and len(line) == 2 and 
                                                        isinstance(line[0], str) and isinstance(line[1], float)):
                                                        
                                                        text, confidence = line
                                                        logging.debug(f"Found text (legacy format): '{text}', confidence: {confidence}")
                                                        
                                                        if confidence > 0.5:
                                                            page_text.append(text)
                                                            
                                                    else:
                                                        logging.debug(f"Unknown result format: {line}")
                                                        
                                                except Exception as format_error:
                                                    logging.debug(f"Error extracting text from line: {format_error}")
                                                    continue
                            
                            # Join text with newlines and add to overall result
                            if page_text:
                                text_parts.append('\n'.join(page_text))
                                logging.debug(f"Successfully extracted {len(page_text)} text lines from page {i}")
                            else:
                                logging.warning(f"No text extracted from page {i} with English model")
                                
                                # Try with German language model as fallback
                                try:
                                    # Lazy initialize German model
                                    if not hasattr(self, '_paddleocr_german'):
                                        try:
                                            from paddleocr import PaddleOCR
                                            self._paddleocr_german = PaddleOCR(
                                                use_angle_cls=True,
                                                lang='german',
                                                use_gpu=False,
                                                show_log=False,
                                                ocr_version='PP-OCRv3'  # Use v3 for better German support
                                            )
                                            logging.info("German PaddleOCR model initialized")
                                        except Exception as ge:
                                            logging.warning(f"Failed to initialize German model: {ge}")
                                    
                                    # Use German model if available
                                    if hasattr(self, '_paddleocr_german'):
                                        try:
                                            german_result = self._paddleocr_german.ocr(pil_img, cls=True)
                                            german_text = self._extract_paddleocr_text(german_result, min_confidence=0.4)
                                            
                                            if german_text:
                                                text_parts.append('\n'.join(german_text))
                                                logging.debug(f"Extracted {len(german_text)} text lines with German model")
                                            else:
                                                logging.warning(f"No text extracted with German model either")
                                        except Exception as ge2:
                                            logging.error(f"German model processing failed: {ge2}")
                                except Exception as ge3:
                                    logging.error(f"Error in German fallback: {ge3}")
                            
                            # Update progress
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                            
                            # Clean up resources
                            img_buffer.close()
                                
                        except Exception as e:
                            logging.error(f"PaddleOCR failed on page {i}: {e}")
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                        finally:
                            # Clean up resources
                            try:
                                if image:
                                    image.close()
                            except:
                                pass
                                
                # Return combined text
                if text_parts:
                    return '\n\n'.join(text_parts)
                else:
                    logging.warning("No text extracted with PaddleOCR")
                    return ""
                    
            except Exception as e:
                logging.error(f"PaddleOCR processing error: {e}")
                return ""
                
        except Exception as e:
            logging.error(f"PaddleOCR extraction failed: {e}")
            return ""

    def _extract_paddleocr_text(self, result, min_confidence=0.5):
        """Helper method to extract text from PaddleOCR result with flexible format handling"""
        extracted_text = []
        
        try:
            if result is None:
                return extracted_text
                
            # Handle result as list of pages
            if isinstance(result, list):
                for page_idx, page_result in enumerate(result):
                    if page_result is None:
                        continue
                        
                    if isinstance(page_result, list):
                        for line in page_result:
                            try:
                                # Try multiple possible formats
                                
                                # Format: [[points], (text, confidence)]
                                if (isinstance(line, list) and len(line) == 2 and 
                                    isinstance(line[1], tuple) and len(line[1]) == 2):
                                    
                                    text, confidence = line[1]
                                    if text and confidence > min_confidence:
                                        extracted_text.append(text)
                                        
                                # Format: [points, [text, confidence]]
                                elif (isinstance(line, list) and len(line) == 2 and 
                                    isinstance(line[1], list) and len(line[1]) == 2):
                                    
                                    text, confidence = line[1]
                                    if text and confidence > min_confidence:
                                        extracted_text.append(text)
                                        
                                # Legacy format: [text, confidence]
                                elif (isinstance(line, list) and len(line) == 2 and 
                                    isinstance(line[0], str) and isinstance(line[1], float)):
                                    
                                    text, confidence = line
                                    if text and confidence > min_confidence:
                                        extracted_text.append(text)
                            except Exception:
                                # Skip this line if extraction fails
                                continue
        except Exception:
            # Return whatever we managed to extract so far
            pass
            
        return extracted_text

    def extract_with_kraken_cli(self, pdf_path: str, progress_callback=None) -> str:
        """
        Extract text using the Kraken CLI instead of the Python API.
        This avoids TensorFlow compatibility issues.
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback for progress updates
            
        Returns:
            str: Extracted text
        """
        # Check if kraken CLI is available
        kraken_bin = shutil.which('kraken')
        if not kraken_bin:
            logging.warning("kraken command not found in PATH. Please install kraken CLI.")
            self._ocr_failed_methods.add('kraken_cli')
            return ""
        
        try:
            # Import pdf2image to convert PDF to images
            pdf2image = self._import_cache.import_module('pdf2image')
            import tempfile
            import os
            import subprocess
            import platform
            
            # Create a temporary directory to store images and results
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert PDF to images
                logging.info(f"Converting PDF to images for Kraken CLI processing")
                with tqdm(desc="Converting PDF to images", unit="page") as pbar:
                    try:
                        images = pdf2image.convert_from_path(
                            pdf_path,
                            dpi=300,
                            thread_count=1,
                            grayscale=True,
                            output_folder=temp_dir,
                            fmt="png"
                        )
                        pbar.update(len(images))
                        logging.info(f"Successfully converted {len(images)} pages to images")
                    except Exception as e:
                        logging.error(f"PDF to image conversion failed: {e}")
                        return ""
                
                # Find all converted images in the temp directory
                image_paths = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) 
                                if f.endswith('.png')])
                
                if not image_paths:
                    logging.error("No images were created from the PDF")
                    return ""
                
                # Process each image with Kraken CLI
                text_parts = []
                
                # Check if we're on Windows to handle command line differences
                is_windows = platform.system() == 'Windows'
                
                with tqdm(total=len(image_paths), desc="Kraken CLI processing", unit="page") as pbar:
                    for i, img_path in enumerate(image_paths, 1):
                        try:
                            # Create output file paths
                            txt_output = os.path.join(temp_dir, f"page_{i}.txt")
                            
                            # Prepare command - use one-step process following the docs
                            # "kraken -i image.tif image.txt segment -bl ocr -m model.mlmodel"
                            ocr_cmd = [
                                kraken_bin,
                                "-i", img_path,
                                txt_output,
                                "segment", "-bl", 
                                "ocr"
                            ]
                            
                            # Check if we have a model to use
                            model_path = None
                            # First check user's home directory for default model
                            model_locations = [
                                os.path.expanduser("~/.local/share/kraken/10.5281_zenodo.10592716.mlmodel"),
                                os.path.expanduser("~/.kraken/10.5281_zenodo.10592716.mlmodel"),
                                # Add Windows locations
                                os.path.join(os.path.expanduser("~"), "AppData", "Local", "kraken", "10.5281_zenodo.10592716.mlmodel"),
                                # Add specific path if you know where kraken stores models
                            ]
                            
                            for loc in model_locations:
                                if os.path.exists(loc):
                                    model_path = loc
                                    break
                                    
                            if model_path:
                                ocr_cmd.extend(["-m", model_path])
                            else:
                                # Try to download the model first if it doesn't exist
                                try:
                                    download_cmd = [kraken_bin, "get", "10.5281/zenodo.10592716"]
                                    logging.info("Downloading default Kraken model...")
                                    
                                    # On Windows, use different shell settings
                                    if is_windows:
                                        download_process = subprocess.run(
                                            download_cmd,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            shell=True  # Use shell on Windows
                                        )
                                    else:
                                        download_process = subprocess.run(
                                            download_cmd,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True
                                        )
                                    
                                    if download_process.returncode == 0:
                                        # Check again for the model
                                        for loc in model_locations:
                                            if os.path.exists(loc):
                                                model_path = loc
                                                ocr_cmd.extend(["-m", model_path])
                                                break
                                    else:
                                        logging.warning(f"Failed to download model: {download_process.stderr}")
                                except Exception as download_error:
                                    logging.warning(f"Error downloading model: {download_error}")
                            
                            # Run OCR command with appropriate shell settings for platform
                            logging.debug(f"Running kraken OCR command: {' '.join(ocr_cmd)}")
                            try:
                                if is_windows:
                                    ocr_process = subprocess.run(
                                        ocr_cmd,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        check=False,  # Don't raise exception on error
                                        shell=True    # Use shell on Windows
                                    )
                                else:
                                    ocr_process = subprocess.run(
                                        ocr_cmd,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        check=False  # Don't raise exception on error
                                    )
                                
                                if ocr_process.returncode != 0:
                                    logging.error(f"Kraken OCR failed: {ocr_process.stderr}")
                                    
                                    # Try an alternative approach based on the docs
                                    logging.info("Trying alternative Kraken command syntax...")
                                    
                                    alt_cmd = [
                                        kraken_bin,
                                        "-i", img_path,
                                        txt_output,
                                        "binarize", "segment", "ocr"
                                    ]
                                    
                                    if model_path:
                                        alt_cmd.extend(["-m", model_path])
                                    
                                    if is_windows:
                                        alt_process = subprocess.run(
                                            alt_cmd,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            shell=True
                                        )
                                    else:
                                        alt_process = subprocess.run(
                                            alt_cmd,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True
                                        )
                                    
                                    if alt_process.returncode != 0:
                                        logging.error(f"Alternative Kraken command failed: {alt_process.stderr}")
                                        continue
                                
                            except Exception as proc_error:
                                logging.error(f"Error running Kraken command: {proc_error}")
                                continue
                            
                            # Read the OCR result if the file exists
                            if os.path.exists(txt_output):
                                try:
                                    with open(txt_output, 'r', encoding='utf-8') as f:
                                        page_text = f.read().strip()
                                        if page_text:
                                            text_parts.append(page_text)
                                            logging.debug(f"Successfully extracted text from page {i}")
                                        else:
                                            logging.warning(f"No text extracted from page {i}")
                                except Exception as read_error:
                                    logging.error(f"Error reading OCR result: {read_error}")
                            else:
                                logging.warning(f"Output file not created: {txt_output}")
                            
                            # Update progress
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                                
                        except Exception as e:
                            logging.error(f"Error processing page {i}: {e}")
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                
                # Combine the text from all pages
                if text_parts:
                    return '\n\n'.join(text_parts)
                else:
                    logging.warning("No text extracted with Kraken CLI")
                    return ""
                    
        except Exception as e:
            logging.error(f"Kraken CLI extraction failed: {e}")
            # Add to failed methods so we don't try again
            self._ocr_failed_methods.add('kraken_cli')
            return ""


    def extract_with_kraken(self, pdf_path: str, progress_callback=None) -> str:
        """
        Extract text using Kraken following the structure of the documentation.
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback for progress updates
            
        Returns:
            str: Extracted text
        """
        if not self._init_ocr('kraken'):
            logging.warning("Kraken initialization failed, skipping Kraken OCR")
            return ""
        
        # Import minimal dependencies
        pdf2image = self._import_cache.import_module('pdf2image')
        
        text_parts = []
        images = None
            
        try:
            # Convert PDF to images
            with tqdm(desc="Converting PDF to images for Kraken", unit="page") as pbar:
                images = pdf2image.convert_from_path(
                    pdf_path,
                    dpi=300,
                    thread_count=1,
                    grayscale=True
                )
                pbar.update(len(images))
            
            # Process images
            with tqdm(total=len(images), desc="Kraken processing", unit="page") as pbar:
                # Import kraken
                try:
                    import kraken
                    from kraken import binarization, pageseg, rpred
                    from kraken.lib import models
                except Exception as kraken_import_error:
                    logging.error(f"Failed to import Kraken modules: {kraken_import_error}")
                    # Add to failed methods so we don't try again
                    self._ocr_failed_methods.add('kraken')
                    return ""
                
                # Try to load a model first - exact steps from documentation
                model = None
                try:
                    # Try direct load using the documented approach
                    model_path = None
                    
                    # Check if there's a get_default_model function
                    if hasattr(rpred, 'get_default_model'):
                        try:
                            model_path = rpred.get_default_model()
                            logging.debug(f"Found default model path: {model_path}")
                        except Exception as path_error:
                            logging.debug(f"Error getting default model path: {path_error}")
                    
                    # If we found a model path, try to load it
                    if model_path:
                        try:
                            # Using documented approach for model loading
                            model = models.load_any(model_path)
                            logging.info(f"Successfully loaded model: {type(model)}")
                        except Exception as model_error:
                            logging.debug(f"Error loading model: {model_error}")
                except Exception as e:
                    logging.debug(f"Error in model loading: {e}")
                    
                # Process each image
                for i, image in enumerate(images, 1):
                    try:
                        # Guard against TensorFlow errors
                        try:
                            # Step 1: Binarize the image - directly following the docs
                            bw_im = binarization.nlbin(image)
                            logging.debug(f"Binarization successful on page {i}")
                            
                            # Step 2: Segment the image - directly following the docs
                            seg = pageseg.segment(bw_im)
                            logging.debug(f"Segmentation successful: {type(seg)}")
                            
                            # Get number of lines detected
                            if hasattr(seg, 'lines'):
                                lines = seg.lines
                                line_count = len(lines)
                                logging.debug(f"Found {line_count} lines on page {i}")
                            else:
                                line_count = 0
                                logging.warning(f"No text lines found on page {i}")
                            
                            # Step 3: Recognition - directly following the docs
                            if line_count > 0:
                                try:
                                    # Exactly as in the documentation
                                    pred_it = rpred(
                                        network=model,  # Can be None as the docs suggest default model is used
                                        im=bw_im,
                                        bounds=seg  # Using the segmentation result directly
                                    )
                                    
                                    # Process prediction records
                                    page_text_parts = []
                                    
                                    # Process results following docs example
                                    try:
                                        for record in pred_it:
                                            if hasattr(record, 'prediction') and record.prediction:
                                                page_text_parts.append(record.prediction)
                                                logging.debug(f"Extracted: {record.prediction[:30]}...")
                                    except Exception as pred_error:
                                        logging.debug(f"Error processing predictions: {pred_error}")
                                    
                                    # Add extracted text from this page
                                    if page_text_parts:
                                        page_text = "\n".join(page_text_parts)
                                        text_parts.append(page_text)
                                        logging.debug(f"Successfully extracted text from page {i}: {len(page_text)} chars")
                                    else:
                                        logging.warning(f"No text extracted from page {i} despite {line_count} lines")
                                        # Add placeholder if no text extracted
                                        text_parts.append(f"[Page {i}: Found {line_count} text regions but extraction failed]")
                                except Exception as recog_error:
                                    # Check for TensorFlow errors
                                    if "tensorflow" in str(recog_error).lower() or "register_load_context_function" in str(recog_error):
                                        logging.error(f"TensorFlow error with Kraken OCR: {recog_error}")
                                        # Add to failed methods so we don't try again
                                        self._ocr_failed_methods.add('kraken')
                                        return ""
                                    
                                    logging.debug(f"Recognition error on page {i}: {recog_error}")
                                    # Add placeholder if recognition failed
                                    text_parts.append(f"[Page {i}: Found {line_count} text regions but recognition failed: {recog_error}]")
                            else:
                                # Add placeholder if no lines found
                                text_parts.append(f"[Page {i}: No text regions found]")
                        
                        except Exception as tensorflow_error:
                            # Check for TensorFlow errors
                            if "tensorflow" in str(tensorflow_error).lower() or "register_load_context_function" in str(tensorflow_error):
                                logging.error(f"TensorFlow error in Kraken OCR: {tensorflow_error}")
                                # Add to failed methods so we don't try again
                                self._ocr_failed_methods.add('kraken')
                                return ""
                            raise  # Re-raise other errors to be caught by the outer try block
                        
                        # Update progress
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(1)
                            
                    except Exception as e:
                        logging.error(f"Kraken processing failed on page {i}: {e}")
                        # Add error placeholder
                        text_parts.append(f"[Page {i}: Processing error: {e}]")
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(1)
                    finally:
                        # Clean up
                        try:
                            image.close()
                        except:
                            pass
            
            # Return joined text if any was extracted
            if text_parts:
                return '\n\n'.join(text_parts)
            else:
                logging.warning("No text extracted with Kraken")
                return ""
                
        except Exception as e:
            # Check for TensorFlow-specific errors
            if "tensorflow" in str(e).lower() or "register_load_context_function" in str(e):
                logging.error(f"TensorFlow compatibility issue with Kraken: {e}")
                # Add to failed methods so we don't try again
                self._ocr_failed_methods.add('kraken')
                return ""
                
            logging.error(f"Kraken extraction failed: {e}")
            return ""
        finally:
            # Clean up
            if 'images' in locals() and images:
                for img in images:
                    try:
                        img.close()
                    except:
                        pass

    def extract_with_easyocr(self, pdf_path: str, progress_callback=None) -> str:
        """
        Extract text using EasyOCR with proper image handling.
        
        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback for progress updates
            
        Returns:
            str: Extracted text
        """
        if not self._init_ocr('easyocr'):
            return ""
        
        try:
            # Import required dependencies here to ensure they're available
            pdf2image = self._import_cache.import_module('pdf2image')
            import numpy as np
            
            text_parts = []
            images = None
            
            try:
                # Convert PDF to images
                with tqdm(desc="Converting PDF to images for EasyOCR", unit="page") as pbar:
                    images = pdf2image.convert_from_path(
                        pdf_path,
                        dpi=300,
                        thread_count=1,  # Single thread for stability
                        grayscale=False   # EasyOCR works better with color images
                    )
                    pbar.update(len(images))
                
                # Initialize reader if needed
                easyocr = self._easyocr
                reader = None
                
                # Initialize reader with English as default language
                try:
                    if not hasattr(self, '_reader') or self._reader is None:
                        import torch
                        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                        self._reader = reader
                    else:
                        reader = self._reader
                except Exception as init_error:
                    logging.error(f"EasyOCR reader initialization failed: {init_error}")
                    return ""
                
                # Process pages
                with tqdm(total=len(images), desc="EasyOCR processing", unit="page") as pbar:
                    for i, image in enumerate(images, 1):
                        try:
                            # Convert PIL image to numpy array (this is what EasyOCR expects)
                            img_array = np.array(image)
                            
                            # Process page using EasyOCR
                            results = reader.readtext(
                                img_array,
                                detail=0,  # Just get the text
                                paragraph=True  # Combine text into paragraphs
                            )
                            
                            # Add extracted text
                            if results:
                                text_parts.append('\n'.join(results))
                            else:
                                text_parts.append(f"[EasyOCR found no text on page {i}]")
                            
                            # Update progress
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                        except Exception as e:
                            logging.error(f"EasyOCR failed on page {i}: {e}")
                        finally:
                            # Make sure to close the image
                            try:
                                image.close()
                            except:
                                pass
                            
            finally:
                # Clean up resources
                if images:
                    for img in images:
                        try:
                            img.close()
                        except:
                            pass
                
                # Clear GPU memory
                self._clear_gpu_memory()
            
            return '\n\n'.join(text_parts)
        
        except Exception as e:
            logging.error(f"EasyOCR extraction failed: {e}")
            return ""


    def _init_ocr(self, method: str) -> bool:
        """
        Initialize OCR engine with correct dependency checks and API usage.
        
        Args:
            method: OCR method to initialize
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if method not in self._ocr_initialized:
            # Skip if already determined to be unavailable
            if method in self._ocr_failed_methods:
                return False
            
            # Check if we're in the main thread
            in_main_thread = threading.current_thread() is threading.main_thread()
            
            try:
                if method == 'tesseract':
                    # Check basic dependencies for tesseract
                    try:
                        # Check binary availability first
                        if not shutil.which('tesseract'):
                            if self._debug:
                                logging.debug("Tesseract binary not found in PATH")
                            self._ocr_initialized[method] = False
                            return False
                        
                        # Import dependencies
                        import pytesseract
                        import pdf2image
                        
                        # Store the imported modules for later use
                        self._pytesseract = pytesseract
                        self._pdf2image = pdf2image
                        
                        # Verify Tesseract installation
                        try:
                            version = pytesseract.get_tesseract_version()
                            if self._debug:
                                logging.debug(f"Found Tesseract version: {version}")
                            self._ocr_initialized[method] = True
                            return True
                        except Exception as e:
                            if self._debug:
                                logging.debug(f"Tesseract verification failed: {e}")
                            self._ocr_initialized[method] = False
                            return False
                    except Exception as e:
                        if self._debug:
                            logging.debug(f"Tesseract initialization error: {e}")
                        self._ocr_initialized[method] = False
                        return False
                    
                elif method == 'paddleocr':
                    try:
                        # Check if paddleocr is available
                        if not self._import_cache.is_available('paddleocr'):
                            if self._debug:
                                logging.debug("PaddleOCR package not available")
                            self._ocr_initialized[method] = False
                            return False
                        
                        # Import paddleocr
                        from paddleocr import PaddleOCR
                        
                        # Initialize the OCR engine with English and German languages
                        self._paddleocr = PaddleOCR(use_angle_cls=True, lang='en', 
                                                    ocr_version='PP-OCRv4')
                        
                        # Store in cache
                        self._ocr_initialized[method] = True
                        logging.info("PaddleOCR initialized successfully")
                        return True
                        
                    except ImportError as e:
                        logging.warning(f"PaddleOCR import error: {e}")
                        self._ocr_initialized[method] = False
                        return False
                    except Exception as e:
                        logging.warning(f"PaddleOCR initialization error: {e}")
                        self._ocr_initialized[method] = False
                        return False
                        
                elif method == 'doctr':
                    
                    try:
                        # Check if doctr is available with better error reporting
                        if not self._import_cache.is_available('doctr'):
                            logging.warning("DocTR package not available")
                            self._ocr_initialized[method] = False
                            return False
                        
                        # Import doctr and verify required modules
                        import doctr
                        
                        # Log doctr version
                        try:
                            logging.info(f"DocTR version: {doctr.__version__}")
                        except:
                            logging.info("DocTR installed but version unknown")
                        
                        # Check for essential modules
                        required_modules = ['io', 'models', 'utils']
                        missing_modules = []
                        
                        for module_name in required_modules:
                            if not hasattr(doctr, module_name):
                                missing_modules.append(module_name)
                        
                        if missing_modules:
                            logging.warning(f"DocTR missing required modules: {', '.join(missing_modules)}")
                            self._ocr_initialized[method] = False
                            return False
                        
                        # Check for required PyTorch installation
                        try:
                            import torch
                            logging.info(f"PyTorch version: {torch.__version__}")
                            
                            # Check CUDA availability - not required but good to know
                            if torch.cuda.is_available():
                                logging.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                            else:
                                logging.info("CUDA not available, using CPU")
                                
                        except ImportError:
                            logging.warning("PyTorch not available - DocTR might not work correctly")
                        
                        # Store module references
                        self._doctr = doctr
                        
                        # Log success
                        logging.info("DocTR initialized successfully")
                        
                        # Store in cache
                        self._ocr_initialized[method] = True
                        return True
                        
                    except ImportError as e:
                        logging.warning(f"DocTR import error: {e}")
                        self._ocr_initialized[method] = False
                        self._ocr_failed_methods.add(method)
                        return False
                    except Exception as e:
                        logging.warning(f"DocTR initialization error: {e}")
                        self._ocr_initialized[method] = False
                        self._ocr_failed_methods.add(method)
                        return False
                    
                elif method == 'kraken_cli':
                    # Check if the kraken CLI is available
                    kraken_bin = shutil.which('kraken')
                    if not kraken_bin:
                        if self._debug:
                            logging.debug("Kraken CLI not found in PATH")
                        self._ocr_initialized[method] = False
                        return False
                    
                    # Check if pdf2image is available (needed to convert PDFs to images)
                    try:
                        import pdf2image
                        self._pdf2image = pdf2image
                        
                        # Successfully initialized
                        self._ocr_initialized[method] = True
                        if self._debug:
                            logging.debug(f"Found Kraken CLI at: {kraken_bin}")
                        return True
                    except ImportError:
                        if self._debug:
                            logging.debug("pdf2image not available, required for Kraken CLI")
                        self._ocr_initialized[method] = False
                        return False
                        
                elif method == 'kraken':
                    # thorough check for kraken availability
                    try:
                        # First verify the package is importable
                        if not self._import_cache.is_available('kraken'):
                            if self._debug:
                                logging.debug("Kraken package not available")
                            self._ocr_initialized[method] = False
                            return False
                        
                        # Check for TensorFlow compatibility before importing kraken
                        try:
                            import tensorflow as tf
                            # Check for the specific attribute that's causing the error
                            if not hasattr(tf._api.v2.compat.v2.__internal__, 'register_load_context_function'):
                                logging.warning("Detected incompatible TensorFlow version for Kraken OCR")
                                self._ocr_initialized[method] = False
                                self._ocr_failed_methods.add(method)
                                return False
                        except (ImportError, AttributeError) as tf_error:
                            # TensorFlow not available or has compatibility issues
                            logging.debug(f"TensorFlow compatibility check failed: {tf_error}")
                            # Continue with Kraken import attempt - it might work with other backends
                        
                        # Import kraken explicitly
                        import kraken
                        
                        # Check if specific modules are available
                        required_modules = []
                        
                        # Check which module structure is used based on version
                        try:
                            # Try importing kraken.binarization
                            from kraken import binarization
                            required_modules.append('binarization')
                        except ImportError:
                            pass
                            
                        try:
                            # Try importing kraken.pageseg
                            from kraken import pageseg
                            required_modules.append('pageseg')
                        except ImportError:
                            pass
                            
                        try:
                            # Try importing kraken.recognition
                            from kraken import recognition
                            required_modules.append('recognition')
                        except ImportError:
                            # Try older module structure
                            try:
                                from kraken import rpred
                                required_modules.append('rpred')
                            except ImportError:
                                pass
                        
                        # If no modules found, we can't use kraken
                        if not required_modules:
                            logging.debug("No usable Kraken modules found")
                            self._ocr_initialized[method] = False
                            return False
                        
                        # Log which module structure we found
                        logging.debug(f"Found Kraken modules: {', '.join(required_modules)}")
                        
                        # Store the kraken module and the required submodules
                        self._kraken = kraken
                        
                        # Store individual module references
                        if 'binarization' in required_modules:
                            self._kraken_binarization = binarization
                        if 'pageseg' in required_modules:
                            self._kraken_pageseg = pageseg
                        if 'recognition' in required_modules:
                            self._kraken_recognition = recognition
                        elif 'rpred' in required_modules:
                            self._kraken_rpred = rpred
                        
                        # Mark as initialized
                        self._ocr_initialized[method] = True
                        return True
                        
                    except ImportError as e:
                        if self._debug:
                            logging.debug(f"Kraken import error: {e}")
                        self._ocr_initialized[method] = False
                        self._ocr_failed_methods.add(method)
                        return False
                    except Exception as e:
                        # Catch TensorFlow-specific errors
                        if "tensorflow" in str(e).lower() or "register_load_context_function" in str(e):
                            logging.warning(f"TensorFlow compatibility issue with Kraken: {e}")
                            self._ocr_initialized[method] = False
                            self._ocr_failed_methods.add(method)
                            return False
                        if self._debug:
                            logging.debug(f"Unexpected error initializing Kraken: {e}")
                        self._ocr_initialized[method] = False
                        self._ocr_failed_methods.add(method)
                        return False
                        
                elif method == 'easyocr':
                    if not self._import_cache.is_available('easyocr'):
                        if self._debug:
                            logging.debug("EasyOCR not available")
                        self._ocr_initialized[method] = False
                        return False
                        
                    # Check if torch is available
                    if not self._import_cache.is_available('torch'):
                        if self._debug:
                            logging.debug("PyTorch not available for EasyOCR")
                        self._ocr_initialized[method] = False
                        return False
                    
                    try:
                        # Import easyocr according to docs
                        import easyocr
                        import torch  # Explicitly import torch here
                        
                        # Store module reference
                        self._easyocr = easyocr
                        
                        # Mark as initialized with deferred reader creation
                        self._ocr_initialized[method] = True
                        self._easyocr_reader_loaded = False
                        return True
                            
                    except ImportError as e:
                        if self._debug:
                            logging.debug(f"EasyOCR import error: {e}")
                        self._ocr_initialized[method] = False
                        return False
                        
            except Exception as e:
                # Catch all other errors
                if self._debug:
                    logging.debug(f"Failed to initialize {method}: {e}")
                self._ocr_initialized[method] = False
                self._ocr_failed_methods.add(method)
        
        # Return cached result
        return self._ocr_initialized.get(method, False)

    def _get_poppler_path(self):
        """
        Get the path to the poppler binaries directory
        
        Returns:
            str or None: Directory containing poppler binaries (pdftoppm, etc.) or None if not found
        """
        # If we have direct path to pdftoppm
        if hasattr(self, '_binary_paths') and self._binary_paths.get('pdftoppm'):
            return os.path.dirname(self._binary_paths['pdftoppm'])
        return None

    def extract_with_tesseract(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using Tesseract OCR without signal handlers in worker threads"""
        # First verify tesseract is actually available
        if 'tesseract' not in self._initialized_methods:
            logging.error("Tesseract not available in initialized methods")
            return ""
            
        # Use the stored module references
        pytesseract = self._pytesseract
        pdf2image = self._pdf2image
        
        text_parts = []
        images = None
        processes = []  # Track any launched subprocesses
        
        # Check if we're in the main thread - signal handlers only work there
        in_main_thread = threading.current_thread() is threading.main_thread()
        
        try:
            # Only set up signal handlers if we're in the main thread
            original_sigint_handler = None
            original_sigterm_handler = None
            
            if in_main_thread:
                original_sigint_handler = signal.getsignal(signal.SIGINT)
                original_sigterm_handler = signal.getsignal(signal.SIGTERM)
                
                def custom_signal_handler(signum, frame):
                    """Custom signal handler to terminate tesseract processes"""
                    logging.info(f"Received signal {signum}, cleaning up tesseract processes...")
                    
                    # Try to terminate any running tesseract processes
                    for proc in processes:
                        if proc and proc.poll() is None:  # If process exists and is still running
                            try:
                                proc.terminate()
                                logging.info(f"Terminated tesseract process {proc.pid}")
                            except Exception as e:
                                logging.error(f"Failed to terminate process: {e}")
                    
                    # Restore original signal handlers
                    signal.signal(signal.SIGINT, original_sigint_handler)
                    signal.signal(signal.SIGTERM, original_sigterm_handler)
                    
                    # Raise KeyboardInterrupt to properly exit
                    raise KeyboardInterrupt("OCR interrupted by user")
                
                # Set custom signal handlers
                signal.signal(signal.SIGINT, custom_signal_handler)
                signal.signal(signal.SIGTERM, custom_signal_handler)

            # Get poppler path for pdf2image
            poppler_path = self._get_poppler_path()
            
            # Convert PDF to images with progress bar
            with tqdm(desc="Converting PDF to images for tesseract", unit="page") as pbar:
                try:
                    # Use thread_count=1 for better interrupt handling

                    # Pass poppler_path if we found it
                    conversion_args = {
                        'dpi': 300,
                        'thread_count': 1,  # Use single thread for stability and interrupt handling
                        'grayscale': True,
                        'size': (None, 2000),  # Limit height for memory
                        'use_cropbox': True,   # Use cropbox for extraction
                        'strict': False        # Continue even with errors
                    }
                    
                    # Add poppler_path if available
                    if poppler_path:
                        conversion_args['poppler_path'] = poppler_path
                        if self._debug:
                            logging.debug(f"Using poppler path: {poppler_path}")
                    
                    # Add password if provided
                    if self._password:
                        conversion_args['userpw'] = self._password
                    
                    images = pdf2image.convert_from_path(
                        pdf_path,
                        **conversion_args
                    )
                    pbar.update(len(images))
                except KeyboardInterrupt:
                    logging.info("PDF conversion interrupted by user")
                    raise
                except Exception as e:
                    logging.error(f"PDF to image conversion failed: {e}")
                    # Try alternative conversion options
                    try:
                        logging.info("Trying alternative conversion method...")
                        images = pdf2image.convert_from_path(
                            pdf_path,
                            dpi=150,  # Lower DPI
                            thread_count=1,
                            grayscale=True,
                            first_page=1,
                            last_page=None,
                            strict=False
                        )
                        pbar.update(len(images))
                    except KeyboardInterrupt:
                        logging.info("Alternative conversion interrupted by user")
                        raise
                    except Exception as e2:
                        logging.error(f"Alternative conversion also failed: {e2}")
                        return ""
            
            # Process images with OCR
            with tqdm(total=len(images), desc="OCR Processing with tesseract", unit="page") as pbar:
                for i, image in enumerate(images, 1):
                    try:
                        # Get tesseract process info for better interrupt handling
                        # First check if we're using pytesseract.pytesseract.run_tesseract
                        original_run_tesseract = None
                        tesseract_process = None
                        
                        # Monkey patch pytesseract to capture the subprocess only if in main thread
                        if in_main_thread and hasattr(pytesseract, 'pytesseract') and hasattr(pytesseract.pytesseract, 'run_tesseract'):
                            original_run_tesseract = pytesseract.pytesseract.run_tesseract
                            
                            def patched_run_tesseract(input_filename, output_filename_base, extension, lang, config=''):
                                cmd = [
                                    pytesseract.pytesseract.tesseract_cmd,
                                    input_filename,
                                    output_filename_base,
                                    *(['-l', lang] if lang else []),
                                    *shlex.split(config),
                                ]
                                proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
                                processes.append(proc)  # Add to our list of processes
                                status = proc.wait()
                                error_string = proc.stderr.read().decode('utf-8').strip()
                                processes.remove(proc)  # Remove from our process list
                                return status, error_string
                            
                            pytesseract.pytesseract.run_tesseract = patched_run_tesseract
                        
                        # OCR with optimized settings
                        try:
                            text = pytesseract.image_to_string(
                                image,
                                config='--oem 3 --psm 3 -l eng',  # Specify language and PSM mode
                                # try deu+eng instead
                            )
                        finally:
                            # Restore original function if we patched it
                            if in_main_thread and original_run_tesseract:
                                pytesseract.pytesseract.run_tesseract = original_run_tesseract
                        
                        if text.strip():
                            text_parts.append(text.strip())
                        
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(1)
                            
                    except KeyboardInterrupt:
                        logging.info(f"OCR interrupted on page {i}")
                        raise
                    except Exception as e:
                        logging.error(f"OCR failed on page {i}: {e}")
                        continue
                    finally:
                        if image:
                            try:
                                image.close()
                            except:
                                pass
        except KeyboardInterrupt:
            logging.info("Tesseract OCR process interrupted by user")
            # Just let it propagate
            raise
        except Exception as e:
            logging.error(f"Tesseract OCR extraction failed: {e}")
            return ""
        finally:
            # Restore original signal handlers if we're in the main thread
            if in_main_thread and original_sigint_handler and original_sigterm_handler:
                signal.signal(signal.SIGINT, original_sigint_handler)
                signal.signal(signal.SIGTERM, original_sigterm_handler)
            
            # Clean up resources
            if images:
                for img in images:
                    try:
                        img.close()
                    except:
                        pass
            
            # Terminate any remaining processes
            for proc in processes:
                if proc and proc.poll() is None:
                    try:
                        proc.terminate()
                    except:
                        pass
        
        # Return combined text if any was extracted
        if text_parts:
            return '\n\n'.join(text_parts)
        else:
            logging.warning("No text extracted using Tesseract OCR")
            return ""


    def _preprocess_image(self, image) -> 'PIL.Image':
        """Optimize image for OCR"""
        try:
            PIL = self._import_cache.import_module('PIL')

            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = PIL.ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Apply adaptive thresholding
            threshold = self._adaptive_threshold(image)
            image = image.point(lambda x: 255 if x > threshold else 0)
            
            return image
        except Exception as e:
            logging.debug(f"Image preprocessing failed: {e}")
            return image

    def _adaptive_threshold(self, image, window_size=41, constant=2):
        """Calculate adaptive threshold for image"""
        try:
            import numpy as np
            img_array = np.array(image)
            mean = np.mean(img_array)
            std = np.std(img_array)
            return int(mean - constant * std)
        except:
            return 127
            
    def _configure_torch_security(self):
        """Configure PyTorch security settings"""
        try:
            import torch
            
            # Configure secure loading
            torch.backends.cudnn.benchmark = True  # Performance optimization
            torch.set_float32_matmul_precision('medium')  # Balance of speed and precision
            
            # Set security-related configurations
            torch.set_warn_always(False)
            
            # Use safer default tensor type - updated API calls
            torch.set_default_tensor_type(torch.FloatTensor)
            torch.set_default_dtype(torch.float32)
            
            # Configure device
            if torch.cuda.is_available():
                try:
                    torch.cuda.init()
                    device = torch.cuda.current_device()
                    logging.info(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
                except Exception as e:
                    logging.warning(f"CUDA initialization failed: {e}")
                    
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logging.info("Using MPS (Metal Performance Shaders) device")
            else:
                logging.info("Using CPU device")
                
        except Exception as e:
            logging.debug(f"PyTorch security configuration failed: {e}")
            # Continue without PyTorch security settings
            pass

    def _clear_gpu_memory(self):
        """Clear GPU memory after processing"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except:
                pass

    
def setup_logging(verbosity: int = 0):
    """Set up logging configuration with TQDM compatibility."""
    levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    level = levels.get(verbosity, logging.DEBUG)
    
    # Configure a logger that works well with tqdm
    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up new handlers
    root_logger.setLevel(level)
    
    format_str = '%(levelname)s: %(message)s' if verbosity > 0 else '%(message)s'
    
    # Console handler that uses tqdm.write
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(logging.Formatter(format_str))
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler('pdf_extraction.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
    root_logger.addHandler(file_handler)
    
    # Suppress common warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Suppress detailed logs from specific libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('pdf2image').setLevel(logging.WARNING)
    logging.getLogger('pytesseract').setLevel(logging.WARNING)
    logging.getLogger('pdfminer').setLevel(logging.WARNING)
    logging.getLogger('pypdf').setLevel(logging.WARNING)
    logging.getLogger('camelot').setLevel(logging.WARNING)
    logging.getLogger('pymupdf').setLevel(logging.WARNING)
    
    if verbosity < 2:  # Unless in debug mode
        logging.getLogger('pypdf').setLevel(logging.ERROR)


class DocumentProcessor:
    """Main document processing coordinator"""
    
    def __init__(self, debug: bool = False):
        self.manager = ExtractionManager(debug=debug)
        self._debug = debug
        # (Optional) Initialize table extractor once if needed
        self._table_extractor = TableExtractor(ImportCache())
        
    def process_files(self, input_files: List[str], 
                output_dir: Optional[str] = None,
                method: Optional[str] = None,
                ocr_method: Optional[str] = None,
                password: Optional[str] = None,
                extract_tables: bool = False,  
                force_ocr: bool = False,
                max_workers: int = None,
                noskip: bool = False,
                sort: bool = False,                
                rename_script_path: str = None,    
                llm_provider = None,  
                temperature: float = 0.7,     
                max_tokens: int = 250,        
                **kwargs) -> Dict[str, Any]:
        """Process multiple files with interrupt handling and optional sorting"""
        results = {}
        failed = []
        skipped = []
        
        # Ensure output directory exists
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine number of workers - fewer workers when sorting to avoid Ollama overload
        if sort:
            # When sorting is enabled, use fewer workers to prevent Ollama API overload
            max_workers = max_workers or min(4, os.cpu_count() or 1)  # Use at most 4 threads for sorting
        else:
            max_workers = max_workers or min(len(input_files), (os.cpu_count() or 1))
        
        # ProcessPoolExecutor doesn't work well with our thread_local OpenAI clients
        # So we stick with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            with tqdm(total=len(input_files), desc="Processing files", unit="file") as pbar:
                for input_file in input_files:
                    logging.debug(f"Processing {input_file}... with {llm_provider}")
                    if shutdown_flag.is_set():
                        logging.info("Shutdown flag detected. Not submitting more jobs.")
                        break
                        
                    future = executor.submit(
                        self._process_single_file,
                        input_file,
                        output_dir,
                        method,
                        ocr_method,  # note: we must be very specific about the correct position
                        password,    # because there are no named parameters for future executions!
                        extract_tables,
                        force_ocr,
                        noskip,
                        sort,                     
                        rename_script_path,       
                        counters if 'counters' in locals() else None,
                        llm_provider,
                        temperature,
                        max_tokens,
                        **kwargs
                    )
                    futures[future] = input_file
                
                for future in as_completed(futures):
                    input_file = futures[future]
                    try:
                        result = future.result()
                        results[input_file] = result
                        if result.get('skipped', False):
                            skipped.append(input_file)
                        elif not result['success']:
                            failed.append((input_file, result.get('error', 'Unknown error')))
                            if self._debug:
                                logging.error(f"Failed to process {input_file}: {result.get('error')}")
                    except Exception as e:
                        failed.append((input_file, str(e)))
                        if self._debug:
                            logging.error(f"Failed to process {input_file}: {e}")
                    finally:
                        pbar.update(1)
                        
                    # Check shutdown flag periodically
                    if shutdown_flag.is_set() and not future.done():
                        future.cancel()
        
        # Print summary 
        if True: # or change to: self._debug
            successful = len([r for r in results.values() if r['success']])
            logging.info(f"\nProcessing Summary:")
            logging.info(f"Total files: {len(input_files)}")
            logging.info(f"Successful: {successful}")
            logging.info(f"Skipped: {len(skipped)}")
            logging.info(f"Failed: {len(failed)}")
            
            if skipped: # and self._debug:
                logging.info("\nSkipped files (output already exists):")
                for file in skipped:
                    logging.info(f"  {file}")
            
            if failed: # and self._debug:
                logging.info("\nFailed files:")
                for file, error in failed:
                    logging.info(f"  {file}: {error}")
        
        return {
            'results': results,
            'failed': failed,
            'skipped': skipped
        }
    
    def _extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract PDF metadata using PyPDF"""
        metadata = {}
        try:
            from pypdf import PdfReader
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                doc_info = reader.metadata
                if doc_info:
                    for key, value in doc_info.items():
                        metadata[key] = value
        except Exception as e:
            logging.warning(f"Failed to extract PDF metadata: {e}")
        return metadata

    def _extract_epub_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract EPUB metadata using ebooklib"""
        metadata = {}
        try:
            from ebooklib import epub
            book = epub.read_epub(file_path)
            titles = book.get_metadata('DC', 'title')
            creators = book.get_metadata('DC', 'creator')
            metadata['title'] = titles[0][0] if titles else None
            metadata['authors'] = [item[0] for item in creators] if creators else None
        except Exception as e:
            logging.warning(f"Failed to extract EPUB metadata: {e}")
        return metadata
    
    def _get_unique_output_path(self, input_file: str, output_dir: Optional[str] = None, noskip: bool = False) -> str:
        """
        Generate output path for a given input file without preserving subdirectory structure
        
        Args:
            input_file: Path or filename of input file
            output_dir: Optional output directory
            noskip: Whether to generate unique filenames for existing files
            
        Returns:
            Output path (without creating unique name if noskip=False)
        """
        # Extract just the basename from the input file, discarding any directory structure
        input_basename = os.path.basename(input_file)
        
        # Convert input file to absolute path if it's a full path
        if os.path.dirname(input_file):
            input_path = Path(input_file).resolve()
            input_basename = input_path.name
        
        # Use current directory if none specified, ensure it's a Path
        output_dir = Path(output_dir or '.').resolve()
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get base name without extension
        base_name = Path(input_basename).stem
        
        # Create basic output path
        output_name = f"{base_name}.txt"
        output_path = output_dir / output_name
        
        # If noskip is True, ensure we have a unique filename to avoid overwriting
        if noskip and output_path.exists():
            counter = 1
            while True:
                output_name = f"{base_name}_{counter}.txt"
                output_path = output_dir / output_name
                
                if not output_path.exists():
                    if self._debug:
                        logging.debug(f"Using unique output path: {output_path}")
                    break
                
                counter += 1
        
        return str(output_path)
    
    def _process_metadata_for_sorting(self, input_file, text, llm_provider, rename_script_path, output_path=None):
        """
        Extract and process metadata for sorting/renaming files.
        """
        result = {
            'success': False,
            'metadata': None,
            'renamed': False,
            'error': None
        }
        
        try:
            logging.debug(f"Processing metadata for file: {input_file}")
            
            # Get metadata using the appropriate provider
            is_openai_client = hasattr(llm_provider, 'chat') and hasattr(llm_provider.chat, 'completions')
            
            if is_openai_client:
                # Using OpenAI client for Ollama
                logging.debug(f"Using OpenAI client for Ollama to extract metadata")
                metadata_content = send_to_ollama_server(text, input_file, llm_provider)
            else:
                # Using an LLM provider instance
                logging.debug(f"Using LLM provider instance to extract metadata")
                metadata_content = send_to_llm(
                    text=text, 
                    filename=input_file, 
                    provider=llm_provider
                )
            
            logging.debug(f"Metadata content received: {metadata_content[:100]}...")
            
            if not metadata_content:
                result['error'] = "Failed to get metadata from LLM provider"
                logging.warning(f"No metadata content received for {input_file}")
                return result
            
            # Parse metadata with improved parser
            metadata = parse_metadata(metadata_content, verbose=True)  # Enable verbose mode
            if not metadata:
                result['error'] = "Failed to parse metadata"
                logging.warning(f"Failed to parse metadata for {input_file}")
                return result
                
            # Get basic metadata fields
            author = metadata['author']
            title = metadata['title']
            year = metadata['year']
            language = metadata.get('language', 'en')
            
            # Validate and fix year
            year = validate_and_fix_year(year, os.path.basename(input_file), text[:5000])
            
            # Process and validate author name
            if not author or author.lower() in ["unknown", "unknownauthor", "n a"]:
                result['error'] = "Missing or invalid author name"
                return result
                
            # Process author with appropriate provider
            if is_openai_client:
                corrected_author = sort_author_names(
                    author_names=author,
                    provider=llm_provider
                )
            else:
                corrected_author = sort_author_names(
                    author_names=author,
                    provider=llm_provider
                )
                
            if not corrected_author or corrected_author == "UnknownAuthor":
                result['error'] = "Failed to format author name correctly"
                return result
                
            # Validate title
            if not title or title.lower() in ["unknown", "title", ""]:
                result['error'] = "Missing or invalid title"
                return result
                
            # Create target paths with sanitized names
            first_author = sanitize_filename(corrected_author)
            sanitized_title = sanitize_filename(title)
            
            # Create target directory path using the base output directory, not preserving subdirectory structure
            # Use the output_dir (if specified) or the current directory as the base
            if output_path:
                base_dir = os.path.dirname(output_path)
                if not base_dir or base_dir == '.':
                    base_dir = os.path.abspath('.')
            else:
                base_dir = os.path.abspath('.')
                
            # Create the target directory directly in the base directory
            target_dir = os.path.join(base_dir, first_author)
            
            # Create new filename with year and title
            file_extension = os.path.splitext(input_file)[1].lower()
            new_filename = f"{year} {sanitized_title}{file_extension}"
            
            # Add language code suffix for non-English files if detected
            if language and language.lower() not in ['en', 'eng', 'english', 'unknown']:
                # Extract just the base extension without dot
                base_ext = file_extension[1:] if file_extension.startswith('.') else file_extension
                # Replace the file extension with language code + extension
                new_filename = f"{year} {sanitized_title}_{language}.{base_ext}"
            
            logging.debug(f"New path/filename will be: {target_dir}/{new_filename}")
            
            # Add rename command to the script
            add_rename_command(
                rename_script_path,
                source_path=input_file,
                target_dir=target_dir,
                new_filename=new_filename,
                output_dir=os.path.dirname(output_path) if output_path else None
            )
            
            result['success'] = True
            result['metadata'] = {
                'author': corrected_author,
                'title': title,
                'year': year,
                'language': language
            }
            result['renamed'] = True
            
        except Exception as e:
            result['error'] = str(e)
            logging.error(f"Error processing metadata: {e}")
            
        return result

    def _handle_sorting(self, input_file, text, llm_provider, rename_script_path, output_path, counters):
        """
        Handle sorting operations for a single file.
        
        Args:
            input_file: Path to the input file
            text: Extracted text content
            llm_provider: LLM provider instance or OpenAI client
            rename_script_path: Path to the rename script
            output_path: Path to the output text file
            counters: Dictionary of counters
            
        Returns:
            dict: Dictionary with result information
        """
        try:
            # Process metadata for sorting
            sort_result = self._process_metadata_for_sorting(
                input_file=input_file,
                text=text,
                llm_provider=llm_provider,
                rename_script_path=rename_script_path,
                output_path=output_path
            )
            
            if sort_result['success']:
                counters['sorted'] += 1
                return {
                    'success': True,
                    'metadata': sort_result['metadata']
                }
            else:
                # Log the error
                error_msg = sort_result.get('error', "Unknown error during sorting")
                logging.warning(f"Failed to sort file {input_file}: {error_msg}")
                
                # Add to unparseables list
                with file_lock:
                    with open("unparseables.lst", "a") as unparseable_file:
                        unparseable_file.write(f"{input_file} - {error_msg}\n")
                        unparseable_file.flush()
                        
                counters['sort_failed'] += 1
                return {
                    'success': False,
                    'error': error_msg
                }
                
        except Exception as e:
            # Catch any unhandled exceptions
            error_msg = f"Error sorting file: {str(e)}"
            logging.error(error_msg)
            
            # Add to unparseables list
            with file_lock:
                with open("unparseables.lst", "a") as unparseable_file:
                    unparseable_file.write(f"{input_file} - {error_msg}\n")
                    unparseable_file.flush()
                    
            counters['sort_failed'] += 1
            return {
                'success': False,
                'error': error_msg
            }

    def _process_single_file(self, input_file: str,
                output_dir: Optional[str] = None,
                method: Optional[str] = None,
                ocr_method: Optional[str] = None,
                password: Optional[str] = None,
                extract_tables: bool = False,
                force_ocr: bool = False,
                noskip: bool = False,
                sort: bool = False,
                rename_script_path: str = None,
                counters: Dict[str, int] = None,
                llm_provider = None,     
                temperature: float = 0.7,
                max_tokens: int = 250,
                **kwargs) -> Dict[str, Any]:
        """
        Process a single document, with optimized skipping logic for sorting
        
        Args:
            input_file: Path to input file
            output_dir: Optional output directory
            method: Preferred extraction method
            ocr_method: Optional OCR method
            password: Password for encrypted documents
            extract_tables: Whether to extract tables
            noskip: Whether to process even if output exists
            sort: Whether to sort files based on content
            rename_script_path: Path to write rename commands
            counters: Dictionary for tracking statistics
            llm_provider: Provider for LLM communication
            temperature: Temperature setting for LLM
            max_tokens: Maximum tokens for LLM
            **kwargs: Additional extraction options
            
        Returns:
            Dict with processing results
        """
        logging.debug(f"Attempting processing of single file: {input_file} with {llm_provider}")
        
        # Initialize result dictionary
        result = {
            'success': False,
            'text': '',
            'tables': [],
            'metadata': {},
            'input_file': input_file,
            'skipped': False
        }
        
        # Initialize counters if not provided
        if counters is None:
            counters = {
                'total': 0,
                'processed': 0,
                'skipped': 0,
                'sorted': 0,
                'sort_failed': 0,
                'failed': 0
            }
        
        # Check for shutdown flag
        if shutdown_flag.is_set():
            logging.debug(f"Shutdown flag detected. Skipping {input_file}")
            result['error'] = "Processing aborted due to shutdown signal"
            return result
        
        try:
            # Extract just the filename without path for output
            input_basename = os.path.basename(input_file)
            
            input_stem = os.path.splitext(input_basename)[0]  # Filename without extension
            
            # Use base output directory, creating it if needed
            base_output_dir = os.path.abspath(output_dir or '.')
            os.makedirs(base_output_dir, exist_ok=True)
            
            # Generate the output text file path
            basic_output_path = os.path.join(base_output_dir, f"{input_stem}.txt")
            
            # Check if we should skip this file
            should_skip = False
            if os.path.exists(basic_output_path) and not noskip:
                # If not sorting, skip existing files
                if not sort:
                    should_skip = True
                else:
                    # When sorting, skip only if file is already in rename script
                    if rename_script_path and os.path.exists(rename_script_path):
                        try:
                            # Check if input file path is in rename script
                            with open(rename_script_path, 'r') as script_file:
                                script_content = script_file.read()
                                if input_file in script_content:
                                    should_skip = True
                                    logging.debug(f"File {input_file} already in rename script, skipping")
                        except Exception as e:
                            logging.error(f"Error checking rename script: {e}")
                            # Continue processing if we can't check the rename script
            
            
            if should_skip:
                if self._debug:
                    logging.info(f"Skipping {input_file} - output file exists and already processed")
                result['success'] = True
                result['skipped'] = True
                result['output_path'] = basic_output_path
                counters['skipped'] += 1
                return result
            
            # Create unique output path if needed (for noskip option)
            output_path = basic_output_path
            
            if noskip and os.path.exists(basic_output_path):
                counter = 1
                while True:
                    output_path = os.path.join(base_output_dir, f"{input_stem}_{counter}.txt")
                    
                    if not os.path.exists(output_path):
                        break
                    counter += 1
            
            # Determine if we need to extract text or can use existing file
            text = ""
            reused_text = False

            
            
            if sort and os.path.exists(basic_output_path):
                # Reuse existing text file for sorting
                try:
                    with open(basic_output_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    reused_text = True
                    logging.debug(f"Reusing existing text from {basic_output_path} for sorting")
                except Exception as e:
                    logging.error(f"Error reading existing text file: {e}")
                    # Will fall back to extraction
                    
            # Extract text if we couldn't reuse existing
            if not reused_text:
                logging.debug(f"Going to extract {input_file} -> {output_path}")
                    
                text = self.manager.extract(
                    input_file,
                    output_path=None,  # We'll handle saving ourselves
                    method=method,
                    ocr_method=ocr_method,
                    password=password,
                    extract_tables=extract_tables,
                    force_ocr=force_ocr, 
                    **kwargs
                )

            
            
            if text:
                result['text'] = text
                result['success'] = True
                counters['processed'] += 1
                
                # Only save the text to file if we extracted it (not if we reused existing)
                if not reused_text:
                    try:
                        # Ensure the output directory exists
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # Write the text file
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(text)
                        result['output_path'] = output_path
                        
                        if self._debug:
                            logging.info(f"Saved text to {output_path}")
                        else:
                            logging.debug(f"Saved text to {output_path}")
                            
                    except Exception as e:
                        logging.error(f"Failed to write output file {output_path}: {e}")
                        result['success'] = False
                        result['error'] = str(e)
                        counters['failed'] += 1
                        return result
                else:
                    result['output_path'] = basic_output_path
                
                logging.debug(f"Working on {input_basename} => {output_path}: {sort}, {llm_provider}, {rename_script_path} ...")   
                
                # Handle sorting if enabled
                try:
                    # First, check what type of provider we have
                    is_openai_client = False
                    if llm_provider is None:
                        # Fall back to local Ollama
                        openai_client = get_openai_client()
                        is_openai_client = True
                        metadata_content = send_to_ollama_server(text, input_file, openai_client)
                    else:
                        # Use the provided LLM provider
                        logging.debug(f"Sending to llm {llm_provider}.")
                        metadata_content = send_to_llm(
                            text=text, 
                            filename=input_file, 
                            provider=llm_provider
                        )
                    
                    if metadata_content:
                        # Parse metadata with improved parser
                        metadata = parse_metadata(metadata_content)
                        if metadata:
                            # Process author names
                            author = metadata['author']
                            logging.debug(f"extracted author: {author}")
                            
                            # Use appropriate method for author name sorting
                            if is_openai_client:
                                corrected_author = sort_author_names(author, openai_client)
                            else:
                                corrected_author = corrected_author = sort_author_names(
                                    author_names=author,
                                    provider=llm_provider,
                                    temperature=temperature,
                                    max_tokens=max_tokens
                                )
                            
                            logging.debug(f"corrected author: {corrected_author}")
                            metadata['author'] = corrected_author
                                
                            # Get file details
                            title = metadata['title']
                            year = metadata.get('year', 'Unknown')
                            
                            # Validate and fix year with new helper function
                            year = validate_and_fix_year(year, os.path.basename(input_file), text[:5000])
                            
                            # Get language if available
                            language = metadata.get('language', 'en')
                            
                            # Validate essential metadata
                            if not corrected_author or corrected_author == "UnknownAuthor" or not title:
                                logging.warning(f"Missing author or title for {input_file}. Skipping rename.")
                                with file_lock:
                                    with open("unparseables.lst", "a") as unparseable_file:
                                        unparseable_file.write(f"{input_file} - Missing metadata: Author='{corrected_author}', Title='{title}'\n")
                                        unparseable_file.flush()
                                counters['sort_failed'] += 1
                            else:
                                # Create target paths with sanitized names
                                first_author = sanitize_filename(corrected_author)
                                sanitized_title = sanitize_filename(title)
                                # Simply use the author name as the target directory - add_rename_command will handle the full path
                                target_dir = first_author  # Just the author name, not a full path

                                logging.debug(f"Outputting to {target_dir}.")

                                file_extension = os.path.splitext(input_file)[1].lower()
                                
                                # Create filename with appropriate formatting
                                # Handle non-English files with language code
                                if language and language.lower() not in ['en', 'eng', 'english', 'unknown']:
                                    # Extract just the base extension without dot
                                    base_ext = file_extension[1:] if file_extension.startswith('.') else file_extension
                                    # Add language code before extension
                                    new_filename = f"{year} {sanitized_title}_{language}.{base_ext}"
                                else:
                                    new_filename = f"{year} {sanitized_title}.{file_extension}"
                                
                                logging.debug(f"New path/filename will be: {target_dir}/{new_filename}")
                                
                                # Add rename command with improved function
                                add_rename_command(
                                    rename_script_path,
                                    source_path=input_file,
                                    target_dir=target_dir,
                                    new_filename=new_filename,
                                    output_dir=os.path.dirname(output_path) if output_path else None
                                )
                                
                                result['metadata'] = metadata
                                counters['sorted'] += 1
                        else:
                            logging.warning(f"Failed to parse metadata for {input_file}")
                            with file_lock:
                                with open("unparseables.lst", "a") as unparseable_file:
                                    unparseable_file.write(f"{input_file} - Failed to parse metadata format: {metadata_content[:100]}...\n")
                                    unparseable_file.flush()
                            counters['sort_failed'] += 1
                    else:
                        logging.warning(f"Failed to get metadata from Ollama server for {input_file}")
                        with file_lock:
                            with open("unparseables.lst", "a") as unparseable_file:
                                unparseable_file.write(f"{input_file} - Failed to get metadata from Ollama server\n")
                                unparseable_file.flush()
                        counters['sort_failed'] += 1
                except Exception as sort_e:
                    logging.error(f"Error sorting file {input_file}: {sort_e}")
                    with file_lock:
                        with open("unparseables.lst", "a") as unparseable_file:
                            unparseable_file.write(f"{input_file} - Error during sorting: {str(sort_e)}\n")
                            unparseable_file.flush()
                    counters['sort_failed'] += 1
                
                # Extract tables if requested (only for PDFs)
                if extract_tables and input_file.lower().endswith('.pdf'):
                    try:
                        tables = self._table_extractor.extract_tables(input_file)
                        result['tables'] = [table.df.to_dict() for table in tables]
                        if self._debug:
                            logging.info(f"Extracted {len(tables)} tables from {input_file}")
                    except Exception as te:
                        logging.error(f"Table extraction failed for {input_file}: {te}")
                        result['tables'] = []
                
                # Extract file metadata
                file_metadata = self._extract_metadata(input_file)
                if file_metadata:
                    result['metadata'].update(file_metadata)
                    
            else:
                logging.error(f"Failed to extract text from {input_file}")
                result['error'] = "No text extracted"
                with file_lock:
                    with open("unparseables.lst", "a") as unparseable_file:
                        unparseable_file.write(f"{input_file} - No text extracted\n")
                        unparseable_file.flush()
                counters['failed'] += 1
                
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logging.error(error_msg)
            result['error'] = error_msg
            result['success'] = False
            counters['failed'] += 1
            
            with file_lock:
                with open("unparseables.lst", "a") as unparseable_file:
                    unparseable_file.write(f"{input_file} - Processing error: {str(e)}\n")
                    unparseable_file.flush()
        
        return result
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract document metadata"""
        metadata = {}
        try:
            file_info = Path(file_path)
            metadata.update({
                'filename': file_info.name,
                'size': file_info.stat().st_size,
                'modified': file_info.stat().st_mtime
            })
            
            # Extract document-specific metadata
            if file_info.suffix.lower() == '.pdf':
                metadata.update(self._extract_pdf_metadata(file_path))
            elif file_info.suffix.lower() == '.epub':
                metadata.update(self._extract_epub_metadata(file_path))
                
        except Exception as e:
            if self._debug:
                logging.error(f"Metadata extraction failed: {e}")
                
        return metadata


# Signal handler to set the shutdown flag
# Keep the global signal handler for setting the shutdown_flag
# Keep track of any OCR subprocesses
ocr_processes = []
active_processes = []  # Track all active subprocesses
extraction_in_progress = threading.Event()  # Flag to indicate extraction is in progress

def signal_handler(signum, frame):
    """
    Enhanced signal handler for SIGINT, SIGTERM, and other interrupt signals.
    Sets the shutdown flag and immediately terminates any running processes.
    """
    global active_processes, extraction_in_progress
    
    if not shutdown_flag.is_set():  # Only log once
        signal_name = {
            signal.SIGINT: "SIGINT (Ctrl+C)",
            signal.SIGTERM: "SIGTERM",
            signal.SIGTSTP: "SIGTSTP (Ctrl+Z)"
        }.get(signum, f"Signal {signum}")
        
        logging.info(f"Received {signal_name}. Initiating forceful shutdown...")
        
        # Set shutdown flag
        shutdown_flag.set()
        
        # Immediately terminate all tracked processes
        for proc in list(active_processes):
            if proc and proc.poll() is None:  # If process exists and is still running
                try:
                    logging.info(f"Terminating process {proc.pid}")
                    proc.terminate()
                    
                    # Give it a short time to terminate gracefully
                    for _ in range(5):  # Wait up to 0.5 seconds
                        if proc.poll() is not None:
                            break
                        time.sleep(0.1)
                    
                    # If still running, force kill
                    if proc.poll() is None:
                        if platform.system() != 'Windows':
                            logging.info(f"Forcefully killing process {proc.pid}")
                            os.kill(proc.pid, signal.SIGKILL)
                        else:
                            logging.info(f"Forcefully terminating process {proc.pid}")
                            proc.kill()
                except Exception as e:
                    logging.error(f"Failed to terminate process: {e}")
        
        # Clean up active_processes list
        active_processes.clear()
        
        # Set extraction_in_progress to false
        extraction_in_progress.clear()

# Register enhanced signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if platform.system() != 'Windows':
    signal.signal(signal.SIGTSTP, signal_handler)

def run_process(cmd, **kwargs):
    """
    Run a subprocess and track it for proper Ctrl+C handling.
    
    Args:
        cmd: Command to run (list of strings)
        **kwargs: Additional arguments for subprocess.run
        
    Returns:
        subprocess.CompletedProcess: The completed process object
    """
    global active_processes
    
    # Set extraction in progress
    extraction_in_progress.set()
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=kwargs.get('stdout', subprocess.PIPE),
        stderr=kwargs.get('stderr', subprocess.PIPE),
        text=kwargs.get('text', True)
    )
    
    # Add to active processes list
    active_processes.append(process)
    
    try:
        # Wait for process to complete
        stdout, stderr = process.communicate()
        
        # Create a CompletedProcess object similar to subprocess.run
        result = subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr
        )
        
        return result
    finally:
        # Remove from active processes list
        if process in active_processes:
            active_processes.remove(process)
        
        # If this was the last process, clear the extraction flag
        if not active_processes:
            extraction_in_progress.clear()

def is_file_in_rename_script(rename_script_path, input_file):
    """
    Check if a file path is already mentioned in the rename script(s).
    Checks both bash and batch scripts if needed.
    
    Args:
        rename_script_path: Path to the rename script
        input_file: Input file path to check
        
    Returns:
        bool: True if file is already in any rename script, False otherwise
    """
    # Determine if we need to check both scripts
    is_windows = platform.system() == 'Windows'
    script_ext = os.path.splitext(rename_script_path)[1].lower()
    
    scripts_to_check = [rename_script_path]
    
    # Add batch script if on Windows and main script is not .bat
    if is_windows and script_ext != '.bat':
        batch_script_path = os.path.splitext(rename_script_path)[0] + '.bat'
        if os.path.exists(batch_script_path):
            scripts_to_check.append(batch_script_path)
    
    # Check each script
    for script_path in scripts_to_check:
        if not os.path.exists(script_path):
            continue
            
        try:
            with open(script_path, 'r') as script_file:
                content = script_file.read()
                
                # For Windows script, check both with forward and backslashes
                if script_path.endswith('.bat'):
                    unix_path = input_file
                    win_path = input_file.replace('/', '\\')
                    if unix_path in content or win_path in content:
                        return True
                else:
                    if input_file in content:
                        return True
        except Exception as e:
            logging.error(f"Error checking rename script {script_path}: {e}")
    
    # If we get here, the file isn't in any script
    return False
    
def sanitize_filename(name):
    """Sanitize a filename to ensure safe filesystem operations"""
    unsafe_chars = {
        '\\': '-',  # backslash to hyphen
        '/': '-',   # forward slash to hyphen
        ':': '-',   # colon to hyphen
        '*': '',    # remove asterisk
        '?': '',    # remove question mark
        '"': '',    # remove double quote
        "'": "",    # remove single quote
        '<': '',    # remove less than
        '>': '',    # remove greater than
        '|': '-',   # pipe to hyphen
        ';': '',    # remove semicolon
        '`': '',    # remove backtick
        '$': '',    # remove dollar sign
        '&': 'and', # ampersand to 'and'
        '!': '',    # remove exclamation
        '#': '',    # remove hash
        '=': '',    # remove equals sign
    }
    
    for char, replacement in unsafe_chars.items():
        name = name.replace(char, replacement)
    
    # Clean up whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Handle periods to prevent double dots when adding extension
    # Replace multiple consecutive dots with a single dot
    name = re.sub(r'\.{2,}', '.', name)
    
    # Remove dot at the end of the name to prevent double dots with extension
    name = name.rstrip('.')
    
    # Handle spacing and initials
    if " " in name:
        name = re.sub(r'(\w)\.', r'\1', name)
        parts = name.split()
        new_parts = []
        current_initials = ""
        for part in parts:
            if len(part) == 1:
                current_initials += part
            else:
                if current_initials:
                    new_parts.append(current_initials)
                    current_initials = ""
                new_parts.append(part)
        if current_initials:
            new_parts.append(current_initials)
        name = " ".join(new_parts)
    
    return name.strip().replace('/', '')

def clean_author_name(author_name):
    """
    Clean author name by removing titles, extra spaces, and punctuation.
    """
    # Remove common titles
    author_name = re.sub(r'\b(Dr|Prof|Mr|Mrs|Ms|Rev|Sir)\.?\s+', '', author_name, flags=re.IGNORECASE)
    
    # Remove XML entities
    author_name = re.sub(r'&[a-zA-Z]+;', ' ', author_name)
    
    # Remove special characters but keep diacritics for non-English names
    author_name = re.sub(r'[^\w\s.\'-áéíóúàèìòùäëïöüÄËÏÖÜâêîôûÂÊÎÔÛñÑçÇ]', ' ', author_name, flags=re.UNICODE)
    
    # Clean up extra spaces
    author_name = re.sub(r'\s+', ' ', author_name).strip()
    
    return author_name

def valid_author_name(author_name):
    """
    Check if the author name is valid.
    Flexible validation to handle various name formats.
    """
    # Quick sanity check for empty input
    if not author_name or len(author_name.strip()) < 3:
        return False
        
    # Split into parts
    parts = author_name.strip().split()
    
    # Name should have at least two parts
    if len(parts) < 2:
        return False
        
    # Check for placeholder values
    lower_name = author_name.lower()
    if "unknown" in lower_name and len(parts) == 2 and parts[1].lower() == "unknown":
        # Special case: Allow "Lastname Unknown" as a valid format
        return True
        
    if any(placeholder in lower_name for placeholder in 
           ["unknownauthor", "lastname", "surname", "firstname", "author", "n a"]):
        return False
        
    # Check for reasonable character content
    # Allow letters, spaces, periods, hyphens, apostrophes and accented characters
    if not re.match(r'^[\w\s.\'-áéíóúàèìòùäëïöüÄËÏÖÜâêîôûÂÊÎÔÛñÑçÇ]+$', author_name, re.UNICODE):
        return False
        
    # Each part should be reasonably sized
    if any(len(part) < 1 or len(part) > 20 for part in parts):
        return False
        
    return True

def validate_and_fix_year(year, filename=None, text_sample=None):
    """
    Validate and fix the detected year.
    
    Args:
        year: The year to validate
        filename: Optional filename to extract year from if not valid
        text_sample: Optional text sample to look for year in if not in filename
        
    Returns:
        str: A valid year or "UnknownYear"
    """
    current_year = datetime.now().year
    
    # If year is valid (4 digits between 1500 and current year + 1)
    if year and re.match(r'^\d{4}$', year) and 1500 <= int(year) <= current_year + 1:
        return year
        
    # Try to extract from filename if provided
    if filename:
        filename_year = extract_year_from_filename(filename)
        if filename_year:
            return filename_year
            
    # Try to extract from text if provided
    if text_sample:
        # Look for copyright pattern
        copyright_match = re.search(r'copyright\s+©?\s*(\d{4})', text_sample, re.IGNORECASE)
        if copyright_match:
            return copyright_match.group(1)
            
        # Look for publication pattern
        pub_match = re.search(r'published\s+in\s+(\d{4})', text_sample, re.IGNORECASE)
        if pub_match:
            return pub_match.group(1)
            
        # Look for any 4-digit year in a reasonable range
        year_matches = re.findall(r'\b(1[5-9]\d\d|20[0-2]\d)\b', text_sample)
        if year_matches:
            # Take the most recent year that's not in the future
            valid_years = [y for y in year_matches if int(y) <= current_year]
            if valid_years:
                return max(valid_years)
    
    # Default if no valid year found
    return "UnknownYear"

def execute_rename_commands(script_path):
    """Execute the generated rename commands script"""
    try:
        # Ensure the file is closed before executing
        with open(script_path, 'r') as script_file:
            pass  # Just to ensure it's accessible and can be opened
        subprocess.run(['bash', script_path], check=True)
        logging.info(f"Successfully executed rename commands from {script_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing rename commands: {e}. Please check the rename script for issues.")
    except FileNotFoundError:
        logging.error(f"Rename script '{script_path}' not found.")
    except PermissionError:
        logging.error(f"Permission denied while executing the rename script '{script_path}'. Ensure it is executable.")
    except Exception as e:
        logging.error(f"Unexpected error during rename command execution: {e}")

def parse_metadata(content, verbose=False):
    """
    Parse metadata content returned by the Ollama server supporting multiple formats.
    Properly handles author names with commas.
    
    Returns:
        dict or None: Dictionary containing author, year, title, and language
    """
    logging.debug("parsing metadata...")
    
    # Remove XML declarations which might interfere with parsing
    content = re.sub(r'<\?xml[^>]+\?>', '', content)
    
    # Fix common tag issues
    content = content.replace("<TITLE", "<TITLE>").replace("<AUTHOR", "<AUTHOR>")
    content = content.replace("<YEAR", "<YEAR>").replace("<LANGUAGE", "<LANGUAGE>")
    
    # Try multiple tag formats for each field
    title_patterns = [
        r'<TITLE>(.*?)</TITLE>',
        r'<Title>(.*?)</Title>',
        r'<title>(.*?)</title>',
        r'"(.*?)"'  # For cases where title is just in quotes
    ]
    
    year_patterns = [
        r'<YEAR>(\d{4})</YEAR>',
        r'<Year>(\d{4})</Year>',
        r'<year>(\d{4})</year>'
    ]
    
    author_patterns = [
        r'<AUTHOR>(.*?)</AUTHOR>',
        r'<Author>(.*?)</Author>',
        r'<author>(.*?)</author>'
    ]
    
    language_patterns = [
        r'<LANGUAGE>(.*?)</LANGUAGE>',
        r'<Language>(.*?)</Language>',
        r'<language>(.*?)</language>'
    ]
    
    # Try to extract title
    title = None
    for pattern in title_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            title = match.group(1).strip()
            if title:
                break
    
    # Try to extract year
    year = "Unknown"
    for pattern in year_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            year = match.group(1).strip()
            if year:
                break
    
    # Try to extract author
    author = None
    for pattern in author_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            author = match.group(1).strip()
            if author:
                break
    
    # Try to extract language
    language = "en"
    for pattern in language_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            language = match.group(1).strip().lower()
            if language:
                break
    
    # Don't split on commas in author names - they're likely "Lastname, Firstname" format
    # Instead, handle multiple authors separated by semicolons
    if author and ';' in author:
        # Split on semicolons and keep only the first author
        author = author.split(';')[0].strip()
    
    # Further clean author name
    if author:
        # Remove XML entities
        author = re.sub(r'&[a-zA-Z]+;', ' ', author)
        # Remove placeholder text
        author = re.sub(r'\b(Lastname|Firstname|Surname)\b', '', author, flags=re.IGNORECASE)
        author = re.sub(r'\s+', ' ', author).strip()
    
    # Check if author is a template/placeholder
    if author and author.lower() in ["lastname firstname", "surname firstname"]:
        author = None
    
    # Validate extracted data
    if not title:
        logging.warning(f"No match for title in content")
        return None
    if not author:
        logging.warning(f"No match for author in content")
        return None
    
    # Sanitize filenames
    title = sanitize_filename(title) if title else "unknown"
    author = sanitize_filename(author) if author else "unknown"
    year = sanitize_filename(year)
    language = sanitize_filename(language)
    
    # Check for placeholder values
    if any(placeholder in (title.lower(), author.lower()) 
           for placeholder in ["unknown", "unknownauthor", "n a", ""]):
        logging.warning("Warning: Found 'unknown', 'n a', or empty strings in metadata.")
        return None
        
    return {'author': author, 'year': year, 'title': title, 'language': language}

def send_to_ollama_server(text, filename, openai_client, max_attempts=5, verbose=False):
    """
    Query the Ollama server to extract author, year, title, and language with exponential backoff.
    
    Returns:
        str: The formatted metadata response
    """
    logging.debug("preparing sending to ollama...")
    base_retry_wait = 2  # Base wait time in seconds
    attempt = 1
    
    # Prepare different prompt templates to try if earlier ones fail
    prompt_templates = [
        # First attempt - simple structured format
        (
            f"Extract the main author name (Lastname Surname), "
            f"year of publication, title, and language from the following text, considering the filename '{os.path.basename(filename)}' "
            f"which may contain clues. I need the output **only** in the following format with no additional text or explanations: \n"
            f"<TITLE>The publication title</TITLE>\n<YEAR>2023</YEAR>\n<AUTHOR>Lastname Surname</AUTHOR>\n<LANGUAGE>en</LANGUAGE>\n\n"
        ),
        # Second attempt - emphasize exact format
        (
            f"I need to extract metadata from a document. Please give me ONLY these four tags with the information, and nothing else:\n"
            f"<TITLE>The exact title</TITLE>\n<YEAR>The publication year (4 digits)</YEAR>\n<AUTHOR>The main author's name (LastName FirstName)</AUTHOR>\n<LANGUAGE>The language code</LANGUAGE>\n\n"
            f"Document filename: {os.path.basename(filename)}\n"
        ),
        # Third attempt - even more explicit
        (
            f"You are a metadata extraction tool. Extract these fields from the text:\n"
            f"1. TITLE (the full publication title)\n"
            f"2. YEAR (the 4-digit publication year, use 'Unknown' if not found)\n"
            f"3. AUTHOR (the main author's last name and first name)\n"
            f"4. LANGUAGE (the 2-letter language code, e.g., 'en', 'de', 'fr')\n\n"
            f"Format your response EXACTLY like this with no other text:\n"
            f"<TITLE>The title</TITLE>\n<YEAR>2023</YEAR>\n<AUTHOR>Smith John</AUTHOR>\n<LANGUAGE>en</LANGUAGE>\n\n"
        )
    ]
    
    # Try different prompt templates if we encounter format issues
    while attempt <= max_attempts and not shutdown_flag.is_set():
        # Choose prompt template based on attempt number
        template_index = min(attempt - 1, len(prompt_templates) - 1)
        prompt_template = prompt_templates[template_index]
        
        logging.debug(f"Consulting Ollama server on file: {filename} (Attempt: {attempt}, Template: {template_index + 1})")
        
        # Build the final prompt with text sample
        prompt = prompt_template + f"Here is the document text:\n{text[:3000]}"  # Limit text to avoid token limits
        messages = [{"role": "user", "content": prompt}]
        
        with ollama_semaphore:
            try:
                response = openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=0.5,  # Reduced temperature for more consistent formatting
                    max_tokens=250,
                    messages=messages,
                    timeout=120  # 2 minute timeout
                )
                
                output = response.choices[0].message.content.strip()
                logging.debug(f"Metadata content received from server: {output}")
                
                # Use the new more flexible parser
                metadata = parse_metadata(output, verbose=verbose)
                if metadata:
                    return output
                else:
                    logging.warning(f"Unexpected response format from Ollama server: {output}")
                    # Less aggressive backoff for format issues - might not be server's fault
                    if attempt < max_attempts:
                        wait_time = base_retry_wait * (1.5 ** (attempt - 1))  # Gentler exponential backoff
                        logging.info(f"Retrying with different prompt in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                        attempt += 1
                        continue
                    return output
                
            except Exception as e:
                if "rate_limit" in str(e).lower() or "timeout" in str(e).lower():
                    # Use exponential backoff for rate limiting/timeouts
                    wait_time = base_retry_wait * (2 ** (attempt - 1))  # Exponential backoff
                    logging.info(f"Rate limit or timeout encountered. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                else:
                    logging.error(f"Error communicating with Ollama server for {filename}: {e}")
                    if attempt < max_attempts:
                        wait_time = base_retry_wait * (1.5 ** (attempt - 1))  # Gentler exponential backoff
                        logging.info(f"Retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                        attempt += 1
                        continue
                return ""
                
    logging.error(f"Maximum retry attempts reached for sending to Ollama server.")
    return ""

def initialize_rename_scripts(rename_script_path):
    """Initialize both bash and batch rename scripts if needed"""
    is_windows = platform.system() == 'Windows'
    
    # Get script extension
    script_ext = os.path.splitext(rename_script_path)[1].lower()
    create_bash = script_ext in ['', '.sh']
    create_batch = is_windows or script_ext == '.bat'
    
    # Derive the batch script path if needed
    batch_script_path = rename_script_path
    if create_batch and script_ext != '.bat':
        batch_script_path = os.path.splitext(rename_script_path)[0] + '.bat'
    
    # Initialize bash script
    if create_bash:
        with open(rename_script_path, "w") as bash_file:
            bash_file.write("#!/bin/bash\n")
            #bash_file.write('set -e\n\n')
            bash_file.flush()
        
        try:
            os.chmod(rename_script_path, 0o755)  # Make executable
            logging.debug(f"Made bash script executable: {rename_script_path}")
        except Exception as e:
            logging.warning(f"Could not set executable permission on {rename_script_path}: {e}")
    
    # Initialize batch script
    if create_batch:
        with open(batch_script_path, "w") as batch_file:
            batch_file.write("@echo off\n")
            batch_file.write("setlocal enabledelayedexpansion\n\n")
            batch_file.write("rem Rename script for Windows\n\n")
            batch_file.flush()
    
    return {
        'bash_script': rename_script_path if create_bash else None,
        'batch_script': batch_script_path if create_batch else None
    }

def sort_author_names(author_names, provider, temperature: float = 0.3, 
                    max_tokens: int = 100, max_attempts: int = 5, 
                    verbose: bool = False):
    """
    Format author names into 'Lastname Firstname' format using LLM with backoff.
    
    Args:
        author_names: Author names to format
        provider: LLM provider instance or OpenAI client
        temperature: Temperature setting for generation
        max_tokens: Maximum tokens to generate
        max_attempts: Maximum retry attempts
        verbose: Whether to print debug info
    """
    if provider is None or isinstance(provider, (int, float)):
        logging.warning(f"Invalid provider passed to sort_author_names: {provider}")
        return author_names
    
    if not author_names or author_names.strip() in ["Unknown", "UnknownAuthor", "n a", ""]:
        return "UnknownAuthor"
    
    # First clean the input and handle commas properly
    author_names = author_names.replace('</AUTHOR>', '').replace('<AUTHOR>', '')
    
    # Check for comma format (likely "Lastname, Firstname")
    if ',' in author_names:
        # This is already likely in "Lastname, Firstname" format
        # Just remove the comma to get "Lastname Firstname"
        formatted_name = author_names.replace(',', ' ').strip()
        # Clean up extra spaces
        formatted_name = re.sub(r'\s+', ' ', formatted_name)
        
        # Validate the name has at least two parts
        if ' ' in formatted_name:
            logging.debug(f"Processed comma-formatted name: {author_names} -> {formatted_name}")
            return formatted_name
    
    # Remove placeholder text that might appear in responses
    formatted_author_names = re.sub(r'\b(Lastname|Firstname|Surname)\b', '', author_names, flags=re.IGNORECASE)
    formatted_author_names = re.sub(r'\s+', ' ', formatted_author_names).strip()
    
    # Handle multiple authors separated by different delimiters (but not commas)
    if any(delimiter in formatted_author_names for delimiter in [';', '&', ' and ']):
        # Split on these delimiters and keep only the first author
        for delimiter in [';', '&', ' and ']:
            if delimiter in formatted_author_names:
                formatted_author_names = formatted_author_names.split(delimiter)[0].strip()
                break
    
    # If after cleaning we have nothing, return unknown
    if not formatted_author_names or len(formatted_author_names.strip()) < 3:
        return "UnknownAuthor"
    
    # Check if name is already in "Lastname Firstname" format after basic processing
    parts = formatted_author_names.split()
    if len(parts) >= 2:
        # We have at least two parts, might be good enough
        return formatted_author_names
    
    # Determine if we're using OpenAI client for Ollama or an LLM provider
    is_openai_client = hasattr(provider, 'chat') and hasattr(provider.chat, 'completions')
    
    # Use LLM to get the correct format with retries
    base_retry_wait = 2  # Base wait time in seconds
    for attempt in range(1, max_attempts + 1):
        if verbose:
            logging.debug(f"Attempt {attempt} to sort author name: {formatted_author_names}")
        
        # Prepare different prompt templates to try if earlier ones fail
        if attempt == 1:
            prompt = (
                f"You will be given an author name that you must put into the format 'Lastname Firstname'. "
                f"If it appears to be just a last name without a first name, return it as is. "
                f"Examples:\n"
                f"- 'Michael Mustermann' → 'Mustermann Michael'\n"
                f"- 'Mustermann Michael' → 'Mustermann Michael' (already correct)\n"
                f"- 'Jean-Paul Sartre' → 'Sartre Jean-Paul'\n"
                f"- 'van Gogh Vincent' → 'van Gogh Vincent' (already correct)\n"
                f"- 'Butz' → 'Butz' (single name, return as is)\n"
                f"Respond ONLY with: <AUTHOR>Lastname Firstname</AUTHOR> or just <AUTHOR>Lastname</AUTHOR> for single names.\n\n"
                f"Author name: {formatted_author_names}"
            )
        else:
            # Try a different approach on subsequent attempts
            prompt = (
                f"I need the author name '{formatted_author_names}' formatted as 'Lastname Firstname(s)'. "
                f"If it's just a single name (like a last name), return it as is without adding anything. "
                f"Don't add any explanation, just return the correctly formatted name in this exact format: "
                f"<AUTHOR>Lastname Firstname</AUTHOR> or <AUTHOR>Lastname</AUTHOR> for single names."
            )
            
        messages = [{"role": "user", "content": prompt}]
        
        # Use the semaphore to limit concurrent requests
        with llm_semaphore:
            try:
                if not hasattr(provider, 'chat_completion'):
                    logging.warning(f"Provider {provider} missing chat_completion method")
                    return author_names
                
                if is_openai_client:
                    # Using OpenAI client for Ollama
                    response = provider.chat.completions.create(
                        model=MODEL_NAME,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        messages=messages
                    )
                    reformatted_name = response.choices[0].message.content.strip()
                else:
                    # Using an LLM provider instance
                    response = provider.chat_completion(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    reformatted_name = response["content"]
                
                # Check for tag issues and fix them
                if "<AUTHOR>" not in reformatted_name:
                    reformatted_name = f"<AUTHOR>{reformatted_name}</AUTHOR>"
                if "</AUTHOR>" not in reformatted_name and not reformatted_name.endswith("</AUTHOR>"):
                    reformatted_name = reformatted_name.replace("<AUTHOR>", "<AUTHOR>") + "</AUTHOR>"
                
                name_match = re.search(r'<AUTHOR>(.*?)</AUTHOR>', reformatted_name)
                if name_match:
                    ordered_name = name_match.group(1).strip()
                    # Final cleanup - remove any placeholder text that might appear
                    ordered_name = re.sub(r'\b(Lastname|Firstname|Surname)\b', '', ordered_name, flags=re.IGNORECASE)
                    ordered_name = clean_author_name(ordered_name)
                    logging.debug(f"Ordered name after cleaning: '{ordered_name}'")
                    
                    # Return it even if it's a single word - we won't add "Unknown"
                    if ordered_name and len(ordered_name) >= 2:
                        return ordered_name
                else:
                    # Try to extract the name without tags if tags are malformed
                    cleaned_response = reformatted_name.replace("</AUTHOR>", "").replace("<AUTHOR>", "").strip()
                    if cleaned_response and cleaned_response not in ['Lastname Firstname', 'Unknown']:
                        ordered_name = clean_author_name(cleaned_response)
                        if ordered_name and len(ordered_name) >= 2:
                            return ordered_name
                            
                    logging.warning(f"Failed to extract a valid name from: '{reformatted_name}', retrying...")
                
            except Exception as e:
                if "rate_limit" in str(e).lower() or "timeout" in str(e).lower():
                    wait_time = base_retry_wait * (2 ** (attempt - 1))
                    logging.info(f"Rate limit or timeout encountered. Retrying in {wait_time:.2f} seconds...")
                else:
                    logging.error(f"Error querying LLM for author names: {e}")
                    wait_time = base_retry_wait * (1.5 ** (attempt - 1))
                    logging.info(f"Retrying in {wait_time:.2f} seconds...")
                
                if attempt < max_attempts:
                    time.sleep(wait_time)
                    continue
                
                # Last resort - just return the original name, even if it's a single word
                return formatted_author_names
        
        # Wait a little between attempts even if no error occurred
        if attempt < max_attempts:
            time.sleep(1)  # Small pause between attempts
                
    logging.error(f"Maximum retry attempts reached for sorting author name: {formatted_author_names}")
    
    # Last resort - just return the original name
    return formatted_author_names

def extract_metadata(text, filename, llm_provider):
    """
    Extract metadata using the provided LLM provider
    
    Args:
        text: Text to analyze
        filename: Filename for context
        llm_provider: Either an OpenAI client for Ollama or an LLMProvider instance
    
    Returns:
        str: The metadata response
    """
    if isinstance(llm_provider, (int, float)):
        logging.error(f"Invalid llm_provider parameter: {llm_provider} (type: {type(llm_provider)})")
        return ""
    
    # Check if we're using OpenAI client for Ollama or an LLM provider
    is_openai_client = hasattr(llm_provider, 'chat') and hasattr(llm_provider.chat, 'completions')
    
    if is_openai_client:
        # Using OpenAI client for Ollama
        return send_to_ollama_server(text, filename, llm_provider)
    else:
        # Using an LLM provider instance
        return send_to_llm(text, filename, llm_provider)
    
def extract_year_from_filename(filename):
    """
    Try to extract a year (4 digits between 1500-2030) from a filename.
    
    Args:
        filename: The filename to check
        
    Returns:
        str: The year if found, None otherwise
    """
    # Look for a 4-digit number within the reasonable range for publication years
    matches = re.findall(r'\b(1[5-9]\d\d|20[0-2]\d)\b', filename)
    
    if matches:
        # Return the first reasonable match
        return matches[0]
    
    return None

def add_rename_command(rename_script_path, source_path, target_dir, new_filename, output_dir=None, debug=False):
    """
    Add mkdir and mv commands to the rename script.
    Also moves the corresponding .txt file if it exists.
    Generates appropriate commands for both bash and Windows batch scripts.
    
    Parameters:
        rename_script_path: Path to the rename script file
        source_path: Source file path
        target_dir: Target directory path - will have any commas removed
        new_filename: New filename
        output_dir: Optional output directory for text files
        debug: Whether to print verbose debug information
    
    Returns:
        dict: Information about the generated scripts and paths
    """
    import os
    import re
    import platform
    import logging

    # Sanitize both the target directory and filename
    target_dir = sanitize_filename(target_dir.replace(',', ''))  # remove also commas
    
    # Sanitize the new filename
    new_filename = sanitize_filename(new_filename)

    # Make sure target_dir is a simple relative path, not containing the base output dir multiple times
    # Check if target_dir contains the path separator multiple times
    if target_dir.count(os.path.sep) > 0:
        # Extract just the last component (the author name)
        target_dir = os.path.basename(target_dir)
        
    # Now create the full absolute paths
    base_dir = output_dir or os.path.dirname(source_path)
    full_target_dir = os.path.join(os.path.abspath(base_dir), target_dir)

    # Figure out the original extension
    orig_ext = os.path.splitext(source_path)[1].lower()  # e.g. ".pdf" or ".epub"
    if debug:
        logging.debug(f"Original file extension: {orig_ext}")

    # Create a clean version of the original extension without the dot
    clean_ext = orig_ext.lstrip('.')
    
    # Handle cases where extension might be embedded in the filename
    # Look for pattern where extension is at the end or followed by another extension
    endings_to_fix = [
        f"{clean_ext}",           # Handles "Titlepdf"
        f"{clean_ext}{orig_ext}", # Handles "Titlepdf.pdf"
        f"de{clean_ext}",         # Handles common patterns like "Staatdepdf"
        f"in{clean_ext}",         # Handles common patterns like "Bookinpdf"
    ]
    
    for ending in endings_to_fix:
        if new_filename.lower().endswith(ending.lower()):
            # Replace the problematic ending
            new_filename = new_filename[:-len(ending)]
            if debug:
                logging.debug(f"Removed ending '{ending}', now: {new_filename}")
            break
            
    # Also check for the extension embedded elsewhere in the filename
    # We need to be careful not to remove legitimate words that might contain these letters
    # So we specifically look for extension patterns with word boundaries
    ext_pattern = r'\b' + clean_ext + r'\b'
    if re.search(ext_pattern, new_filename, re.IGNORECASE):
        new_filename = re.sub(ext_pattern, '', new_filename, flags=re.IGNORECASE)
        if debug:
            logging.debug(f"Removed embedded extension, now: {new_filename}")
    
    # Clean up any extra spaces
    new_filename = re.sub(r'\s+', ' ', new_filename).strip()
    
    # Make sure no double dots when adding extension
    new_filename = new_filename.rstrip('.')
    
    # If the new name doesn't already end with the real extension, append it
    if not new_filename.lower().endswith(orig_ext.lower()):
        new_filename += orig_ext
        if debug:
            logging.debug(f"Added extension: {new_filename}")
    
    # Final check for consecutive dots and remove them
    new_filename = re.sub(r'\.{2,}', '.', new_filename)
    
    # Determine if we're on Windows
    is_windows = platform.system() == 'Windows'
    
    # Get script extension to determine if we need to create both scripts
    script_ext = os.path.splitext(rename_script_path)[1].lower()
    create_bash = script_ext in ['', '.sh']
    create_batch = is_windows or script_ext == '.bat'
    
    # Derive the batch script path if needed
    batch_script_path = rename_script_path
    if create_batch and script_ext != '.bat':
        batch_script_path = os.path.splitext(rename_script_path)[0] + '.bat'
    
    # Prepare paths for bash script - use the cleaned paths
    escaped_source_path = escape_special_chars(os.path.abspath(source_path))
    escaped_target_dir = escape_special_chars(full_target_dir)
    escaped_target_path = escape_special_chars(os.path.join(full_target_dir, new_filename))
    
    # Prepare paths for Windows batch script
    # Convert forward slashes to backslashes for Windows paths
    win_source = source_path.replace('/', '\\')
    win_target_dir_path = target_dir.replace('/', '\\')
    win_target_full = os.path.join(target_dir, new_filename).replace('/', '\\')
    
    # Add quotes for Windows paths
    win_source_path = '"' + win_source + '"'
    win_target_dir = '"' + win_target_dir_path + '"'
    win_target_path = '"' + win_target_full + '"'

    # Determine the corresponding text file paths
    if output_dir:
        # If output_dir is specified, text files are in that directory
        txt_source_path = os.path.join(output_dir, os.path.splitext(os.path.basename(source_path))[0] + ".txt")
    else:
        # Otherwise, text files are in the same directory as the source files
        txt_source_path = os.path.splitext(source_path)[0] + ".txt"
        
    # Target text file will be in the target directory with related name
    txt_new_filename = os.path.splitext(new_filename)[0] + ".txt"
    
    # Make sure no double dots in text filename
    txt_new_filename = re.sub(r'\.{2,}', '.', txt_new_filename)
    
    txt_target_path = os.path.join(target_dir, txt_new_filename)
    
    # Escape for bash - make sure we use proper absolute paths
    escaped_txt_source_path = escape_special_chars(os.path.abspath(txt_source_path))
    escaped_txt_target_path = escape_special_chars(os.path.join(os.path.abspath(target_dir), txt_new_filename))
    
    # Prepare for Windows batch
    win_txt_source = txt_source_path.replace('/', '\\')
    win_txt_target = txt_target_path.replace('/', '\\')
    
    # Add quotes for Windows text paths
    win_txt_source_path = '"' + win_txt_source + '"'
    win_txt_target_path = '"' + win_txt_target + '"'
    
    if debug:
        logging.debug(f"Source path: {source_path}")
        logging.debug(f"Target directory: {target_dir}")
        logging.debug(f"New filename: {new_filename}")
        logging.debug(f"Full target path: {os.path.join(target_dir, new_filename)}")
        if txt_source_path:
            logging.debug(f"Text source path: {txt_source_path}")
            logging.debug(f"Text target path: {txt_target_path}")
    
    # Write to the bash script if needed
    if create_bash:
        with file_lock:
            with open(rename_script_path, "a") as bash_file:
                # Create the target directory
                bash_file.write(f"mkdir -p {escaped_target_dir}\n")
                
                # Move the original file
                bash_file.write(f"mv {escaped_source_path} {escaped_target_path}\n")
                
                # Check if the corresponding text file exists and move it too
                bash_file.write(f"# Also move the text file if it exists\n")
                bash_file.write(f"if [ -f {escaped_txt_source_path} ]; then\n")
                bash_file.write(f"  mv {escaped_txt_source_path} {escaped_txt_target_path}\n")
                bash_file.write(f"fi\n\n")
                bash_file.flush()
                
        if debug:
            logging.debug(f"Added commands to bash script: {rename_script_path}")
    
    # Write to the Windows batch script if needed
    if create_batch:
        with file_lock:
            with open(batch_script_path, "a") as batch_file:
                # Create the target directory (mkdir in Windows automatically creates parent dirs)
                batch_file.write(f"if not exist {win_target_dir} mkdir {win_target_dir}\n")
                
                # Move the original file
                batch_file.write(f"move {win_source_path} {win_target_path}\n")
                
                # Check if the corresponding text file exists and move it too
                batch_file.write(f"rem Also move the text file if it exists\n")
                batch_file.write(f"if exist {win_txt_source_path} (\n")
                batch_file.write(f"  move {win_txt_source_path} {win_txt_target_path}\n")
                batch_file.write(f")\n\n")
                batch_file.flush()
                
        if debug:
            logging.debug(f"Added commands to batch script: {batch_script_path}")
    
    logging.debug(f"Added rename command: {source_path} -> {os.path.join(target_dir, new_filename)}")
    if os.path.exists(txt_source_path):
        logging.debug(f"Will also move text file: {txt_source_path} -> {txt_target_path}")
    
    return {
        'bash_script': rename_script_path if create_bash else None,
        'batch_script': batch_script_path if create_batch else None,
        'target_dir': target_dir,
        'new_path': os.path.join(target_dir, new_filename)
    }

def detect_language(text, min_text_length=100, max_sample_length=1000, verbose=False):
    """
    Detect language of text using multiple fallback methods.
    
    Args:
        text: Text to analyze
        min_text_length: Minimum text length to attempt detection
        max_sample_length: Maximum sample length to use
        verbose: Whether to print debug information
    
    Returns:
        str: ISO 639-1 two-letter language code (e.g., 'en', 'de', 'fr') or None if detection fails
    """
    if not text or len(text.strip()) < min_text_length:
        if verbose:
            print(f"Text too short for reliable language detection ({len(text) if text else 0} chars)")
        return None
    
    # Use a sample of text to avoid processing very large documents
    sample = text[:max_sample_length].strip()
    
    # Try langdetect first (fastest and most reliable)
    try:
        from langdetect import detect, DetectorFactory, LangDetectException
        # Make detection deterministic
        DetectorFactory.seed = 0
        try:
            return detect(sample)
        except LangDetectException as e:
            if verbose:
                print(f"langdetect failed: {str(e)}")
    except ImportError:
        if verbose:
            print("langdetect not available")
    
    # Try langid second (good balance of speed and accuracy)
    try:
        import langid
        lang, _ = langid.classify(sample)
        return lang
    except ImportError:
        if verbose:
            print("langid not available")
    
    # Try cld3 third (neural network-based, good for short text)
    try:
        import cld3
        prediction = cld3.get_language(sample)
        if prediction.is_reliable:
            return prediction.language
    except ImportError:
        if verbose:
            print("cld3 not available")
    
    # Final fallback: check for common language patterns
    return _fallback_detect_language(sample)

def _fallback_detect_language(text):
    """
    Very basic fallback language detection based on character patterns.
    Only detects a few common languages with distinctive characters.
    
    Args:
        text: Text to analyze
        
    Returns:
        str: ISO 639-1 language code or None
    """
    # Count characters in specific Unicode ranges
    counts = {
        'en': 0,  # English/Latin
        'zh': 0,  # Chinese
        'ja': 0,  # Japanese
        'ko': 0,  # Korean
        'ru': 0,  # Russian/Cyrillic
        'ar': 0,  # Arabic
        'de': 0,  # German
    }
    
    # Common German words that help distinguish from English
    german_words = ['der', 'die', 'das', 'und', 'ist', 'von', 'zu', 'mit', 'sich', 'auf', 'für', 'ein', 'eine']
    german_word_count = sum(1 for word in text.lower().split() if word in german_words)
    
    for char in text:
        # Chinese characters
        if '\u4e00' <= char <= '\u9fff':
            counts['zh'] += 1
        # Japanese-specific characters (Hiragana & Katakana)
        elif '\u3040' <= char <= '\u30ff':
            counts['ja'] += 1
        # Korean Hangul
        elif '\uac00' <= char <= '\ud7a3':
            counts['ko'] += 1
        # Russian/Cyrillic
        elif '\u0400' <= char <= '\u04ff':
            counts['ru'] += 1
        # Arabic
        elif '\u0600' <= char <= '\u06ff':
            counts['ar'] += 1
        # Latin alphabet (English, German, etc.)
        elif 'a' <= char.lower() <= 'z':
            counts['en'] += 1
    
    # Adjust for German if we see German-specific words
    if german_word_count > 5 and counts['en'] > 0:
        return 'de'
    
    # Return the language with the most character matches
    if any(counts.values()):
        # Get the language with highest count, excluding English initially
        non_latin = {k: v for k, v in counts.items() if k != 'en'}
        if non_latin and max(non_latin.values()) > 0:
            return max(non_latin.items(), key=lambda x: x[1])[0]
        # If no non-Latin characters detected, return English
        return 'en'
    
    return None

def escape_special_chars(filename):
    """Safely escape special characters in filenames for shell commands."""
    try:
        import shlex
        # Make sure we're handling paths correctly
        if filename and os.path.exists(os.path.dirname(filename)):
            # Real path that exists - escape it properly for bash
            return shlex.quote(filename)
        else:
            # Path might not exist yet - still escape it safely
            return shlex.quote(filename)
    except ImportError:
        # Fallback handling if shlex is not available
        sanitized = sanitize_filename(filename)
        
        # Escape remaining shell metacharacters
        chars_to_escape = r'!$"`\\'
        for char in chars_to_escape:
            sanitized = sanitized.replace(char, '\\' + char)
            
        # Properly handle single quotes
        if "'" in sanitized:
            sanitized = sanitized.replace("'", "'\\''")
            
        return f"'{sanitized}'"
    
def get_openai_client():
    """Initialize or return thread-local OpenAI client for Ollama"""
    if not hasattr(thread_local, "client"):
        try:
            # Import within the function to ensure it's available
            from openai import OpenAI
            
            # Check if Ollama is accessible
            ollama_url = "http://localhost:11434/v1/"
            logging.debug(f"Initializing OpenAI client with base URL: {ollama_url}")
            
            thread_local.client = OpenAI(
                base_url=ollama_url,
                api_key="ollama"
            )
            logging.debug("OpenAI client initialized successfully for thread.")
        except ImportError as e:
            logging.critical(f"OpenAI client not available: {e}")
            logging.critical("Please install the OpenAI client: pip install openai")
            raise
        except Exception as e:
            logging.critical(f"Failed to initialize OpenAI client in thread: {e}")
            raise
    return thread_local.client

def main():
    import glob
    
    """Command-line interface entry point"""
    parser = argparse.ArgumentParser(
        description="Document Text Extraction Tool for PDF, EPUB, DJVU, MOBI, TXT, HTML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              %(prog)s input.pdf
              %(prog)s -o output_dir/ *.pdf
              %(prog)s --method=djvulibre input.djvu
              %(prog)s --method=mobi input.mobi
              %(prog)s --method=bs4 input.html
              %(prog)s "*.pdf *.djvu *.epub"  # Process multiple file types
              %(prog)s -t -j output.json *.pdf
              %(prog)s --noskip input.pdf  # Process even if output exists
              %(prog)s --sort *.pdf  # Sort and rename files based on content
              %(prog)s --sort --execute-rename *.pdf  # Sort and immediately execute rename commands
        """)
    )
    
    parser.add_argument(
        '--sort',
        action='store_true',
        help="Sort and rename files based on content analysis from LLM"
    )

    parser.add_argument(
        '--force-ocr',
        action='store_true',
        help="Force OCR processing even if text layer extraction would work"
    )

    # arguments for renaming control
    parser.add_argument(
        '--execute-rename',
        action='store_true',
        help="Automatically execute the generated rename commands"
    )

    parser.add_argument(
        '--rename-script',
        default="rename_commands.sh",
        help="Path to write the rename commands (default: rename_commands.sh)"
    )
    
    parser.add_argument(
        'files',
        nargs='*', # changed from +
        help="Input files to process (supports wildcards and multiple file type patterns)"
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        help="Output directory for extracted text files (default: current directory)",
        default='.'  # Default to current directory
    )
    
    parser.add_argument(
        '-m', '--method',
        help=("Preferred extraction method. PDF: pymupdf, pdfplumber, pypdf, pdfminer; "
              "EPUB: ebooklib, bs4, zipfile; DJVU: djvulibre, pdf_conversion, ocr; "
              "MOBI: mobi, kindleunpack, calibre, zipfile; TXT: direct, charset_detection, encoding_detection; "
              "HTML: bs4, html2text, lxml, regex")
    )

    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help="Process files recursively through subdirectories"
    )
    
    parser.add_argument(
        '-p', '--password',
        help="Password for encrypted documents"
    )

    parser.add_argument(
        '--ocr-method',
        choices=['auto', 'tesseract', 'paddleocr', 'doctr', 'easyocr', 'kraken', 'kraken_cli'],
        default='auto',
        help="Preferred OCR method when text extraction is needed"
    )
    
    parser.add_argument(
        '-t', '--tables',
        action='store_true',
        help="Extract tables (PDF only)"
    )
    
    parser.add_argument(
        '-j', '--json',
        help="Save results to JSON file"
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        help="Maximum number of worker threads"
    )
    
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help="Enable debug logging"
    )
    
    parser.add_argument(
        '--noskip',
        action='store_true',
        help="Process files even if output text file already exists"
    )

    # Add to the argument parser in main()
    parser.add_argument(
        '--llm-provider',
        choices=['ollama', 'groq', 'cohere', 'openai', 'glhf', 'huggingface'],
        default='ollama',
        help="LLM provider to use for metadata extraction (default: ollama)"
    )

    parser.add_argument(
        '--llm-model',
        help="Model name to use with the LLM provider"
    )

    parser.add_argument(
        '--api-key',
        help="API key for cloud LLM providers (Groq, Poe)"
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5,
        help="Temperature setting for LLM generation (0.0-1.0)"
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=250,
        help="Maximum tokens in LLM response"
    )

    # Add filter for specific file types
    parser.add_argument(
        '--file-types',
        help="Only process specified file types (comma-separated, e.g., 'pdf,epub,djvu')"
    )

    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    try:
        # Build list of supported extensions (ensure they're all lowercase for comparison)
        supported_extensions = [ext.lower() for ext in ExtractionManager.SUPPORTED_EXTENSIONS.keys()]
        
        # Debug print to see what extensions are supported
        logging.debug(f"Supported extensions: {supported_extensions}")
        
        # Filter extensions if file-types argument is provided
        filtered_extensions = None
        if args.file_types:
            filtered_types = [f".{ext.strip().lower()}" for ext in args.file_types.split(',')]
            filtered_extensions = [ext for ext in filtered_types if ext in supported_extensions]
            if not filtered_extensions:
                logging.error(f"No valid file types specified in: {args.file_types}")
                logging.info(f"Supported file types: {', '.join(ext.lstrip('.') for ext in supported_extensions)}")
                return 1
        
        # Expand file patterns
        input_files = []
        
        # Handle case when no files provided - process all supported types in current directory
        if not args.files:
            logging.info("No files specified, processing all supported file types in current directory")
            for ext in filtered_extensions or supported_extensions:
                pattern = f"*{ext}"
                matched = glob.glob(pattern)
                if matched:
                    input_files.extend(matched)
            
            if not input_files:
                logging.error("No supported files found in current directory")
                return 1
        else:
            # Process each pattern which might include multiple space-separated patterns
            for pattern_group in args.files:
                # Debug which pattern we're processing
                logging.debug(f"Processing pattern group: {pattern_group}")
                
                # Handle the case where the shell has already expanded the glob pattern
                if os.path.isfile(pattern_group):
                    file_ext = os.path.splitext(pattern_group)[1].lower()
                    if file_ext in supported_extensions:
                        input_files.append(pattern_group)
                        if args.debug:
                            logging.debug(f"Direct file match: {pattern_group}")
                    continue
                
                # Split by space to handle multiple patterns like "*.pdf *.epub"
                for pattern in pattern_group.split():
                    if args.debug:
                        logging.debug(f"Processing pattern: {pattern}")
                    
                    # Use case-insensitive pattern matching for extensions
                    if '*.' in pattern:
                        base_pattern = pattern.split('*.')[0]
                        ext = pattern.split('*.')[1].lower()
                        
                        # Create case-insensitive patterns for common extension variants
                        variants = [ext, ext.upper(), ext.capitalize()]
                        for variant in variants:
                            case_pattern = f"{base_pattern}*.{variant}"
                            if args.debug:
                                logging.debug(f"Trying case variant: {case_pattern}")
                            matched = glob.glob(case_pattern)
                            for matched_file in matched:
                                if os.path.isfile(matched_file):
                                    file_ext = os.path.splitext(matched_file)[1].lower()
                                    if file_ext in supported_extensions:
                                        input_files.append(matched_file)
                                        if args.debug:
                                            logging.debug(f"Matched file: {matched_file}")
                            
                    if args.recursive:
                        # If the pattern is an existing directory, walk it recursively
                        if os.path.isdir(pattern):
                            for root, _, files in os.walk(pattern):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    file_ext = os.path.splitext(file_path)[1].lower()
                                    # Only add supported file types
                                    if filtered_extensions is None and file_ext in supported_extensions:
                                        input_files.append(file_path)
                                    elif filtered_extensions is not None and file_ext in filtered_extensions:
                                        input_files.append(file_path)
                        # Check if the pattern is an existing file (handles spaces in filenames)
                        elif os.path.isfile(pattern):
                            file_ext = os.path.splitext(pattern)[1].lower()
                            if filtered_extensions is None and file_ext in supported_extensions:
                                input_files.append(pattern)
                            elif filtered_extensions is not None and file_ext in filtered_extensions:
                                input_files.append(pattern)
                        else:
                            # Pattern globbing for wildcards with recursive option
                            matched_files = glob.glob(pattern, recursive=True)
                            for matched_file in matched_files:
                                if os.path.isfile(matched_file):
                                    file_ext = os.path.splitext(matched_file)[1].lower()
                                    if filtered_extensions is None and file_ext in supported_extensions:
                                        input_files.append(matched_file)
                                    elif filtered_extensions is not None and file_ext in filtered_extensions:
                                        input_files.append(matched_file)
                    else:
                        # Non-recursive mode
                        # Check if the pattern is an existing file (handles spaces in filenames)
                        if os.path.isfile(pattern):
                            file_ext = os.path.splitext(pattern)[1].lower()
                            if filtered_extensions is None and file_ext in supported_extensions:
                                input_files.append(pattern)
                            elif filtered_extensions is not None and file_ext in filtered_extensions:
                                input_files.append(pattern)
                        else:
                            matched_files = glob.glob(pattern)
                            for matched_file in matched_files:
                                if os.path.isfile(matched_file):
                                    file_ext = os.path.splitext(matched_file)[1].lower()
                                    if filtered_extensions is None and file_ext in supported_extensions:
                                        input_files.append(matched_file)
                                    elif filtered_extensions is not None and file_ext in filtered_extensions:
                                        input_files.append(matched_file)
        
        # Remove duplicates while preserving order
        input_files = list(dict.fromkeys(input_files))
        
        if args.debug:
            logging.debug(f"Files found after filtering: {len(input_files)}")
            for file in input_files:
                logging.debug(f"  {file}")
        
        if not input_files:
            logging.error("No supported input files found")
            return 1
            
        # Initialize processor
        processor = DocumentProcessor(debug=args.debug)
        
        # Initialize LLM provider and rename script if sorting is enabled
        llm_provider = None
        rename_script_path = None
        
        if args.sort:
            try:
                logging.debug(f"checking for llm provider: {args.llm_provider}")
                if args.llm_provider == "ollama":
                    # For Ollama, we keep the existing behavior
                    try:
                        from openai import OpenAI
                        OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
                        logging.debug("OpenAI client for Ollama is available")
                    except ImportError:
                        logging.error("OpenAI client not available. Install with 'pip install openai'")
                        logging.error("Proceeding without sorting functionality")
                        args.sort = False
                    except Exception as e:
                        logging.error(f"Error initializing OpenAI client: {e}")
                        logging.error("Proceeding without sorting functionality")
                        args.sort = False
                else:
                    # For other providers, use the new provider factory
                    try:
                        llm_provider = get_llm_provider(
                            provider_type=args.llm_provider,
                            model_name=args.llm_model,
                            api_key=args.api_key
                        )
                        logging.info(f"Initialized {args.llm_provider} LLM provider")
                    except ImportError as e:
                        logging.error(f"Required libraries not available for {args.llm_provider}: {e}")
                        logging.error("Proceeding without sorting functionality")
                        args.sort = False
                    except Exception as e:
                        logging.error(f"Error initializing LLM provider: {e}")
                        logging.error("Proceeding without sorting functionality")
                        args.sort = False
                
                if args.sort:  # Check again in case we disabled it due to errors
                    # Initialize rename script
                    rename_script_path = args.rename_script
                    script_paths = initialize_rename_scripts(rename_script_path)
            except Exception as e:
                logging.error(f"Error initializing sorting functionality: {e}")
                args.sort = False
                llm_provider = None
                rename_script_path = None
                
        if args.sort:  # Check again in case we disabled it due to errors
                # Initialize rename script
                rename_script_path = args.rename_script
                script_paths = initialize_rename_scripts(rename_script_path)
        
        try:
            # Display summary of files to process
            logging.info(f"Processing {len(input_files)} files")
            
            # Group files by extension for stats
            extension_counts = {}
            for file in input_files:
                ext = os.path.splitext(file)[1].lower()
                extension_counts[ext] = extension_counts.get(ext, 0) + 1
                
            for ext, count in sorted(extension_counts.items()):
                logging.info(f"  {ext} files: {count}")

            logging.debug(f"initiating process files for {llm_provider}")

            
            # Process files with periodic shutdown checks
            results = processor.process_files(
                input_files,
                output_dir=args.output_dir,
                method=args.method,
                ocr_method=args.ocr_method,
                password=args.password,
                extract_tables=args.tables,
                force_ocr=args.force_ocr,
                max_workers=args.workers,
                noskip=args.noskip,
                sort=args.sort,
                rename_script_path=rename_script_path,
                llm_provider=llm_provider,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            
            # Handle results
            if args.json:
                import json
                with open(args.json, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Handle rename script if sorting was enabled
            if args.sort and args.execute_rename:
                is_windows = platform.system() == 'Windows'
                
                if is_windows:
                    batch_script_path = os.path.splitext(rename_script_path)[0] + '.bat'
                    if os.path.exists(batch_script_path):
                        logging.info("Executing rename commands using batch script...")
                        try:
                            subprocess.run(['cmd', '/c', batch_script_path], check=True)
                            logging.info("Successfully executed rename commands")
                        except subprocess.CalledProcessError as e:
                            logging.error(f"Error executing rename commands: {e}")
                    else:
                        logging.error(f"Batch script {batch_script_path} not found")
                else:
                    # Unix execution
                    logging.info("Executing rename commands...")
                    execute_rename_commands(rename_script_path)
            elif args.sort:
                # Just print instructions
                if platform.system() == 'Windows':
                    batch_script_path = os.path.splitext(rename_script_path)[0] + '.bat'
                    logging.info(f"Rename commands written to {rename_script_path} and {batch_script_path}")
                    logging.info(f"Review and execute manually with: bash {rename_script_path}")
                    logging.info(f"  or on Windows: {batch_script_path}")
                else:
                    logging.info(f"Rename commands written to {rename_script_path}")
                    logging.info(f"Review and execute manually with: bash {rename_script_path}")
            
            # Update return code logic to include skipped files in the summary
            successful = len(results.get('results', {})) - len(results.get('failed', []))
            skipped = len(results.get('skipped', []))
            
            logging.info(f"Summary: {successful} succeeded, {skipped} skipped, {len(results.get('failed', []))} failed")
            
            return 0 if not results.get('failed') else 1
            
        except KeyboardInterrupt:
            if shutdown_flag.is_set():
                # Proper handling after graceful shutdown flag was set
                if args.sort and rename_script_path:
                    try:
                        os.chmod(rename_script_path, 0o755)
                        logging.info(f"Made rename script executable: {rename_script_path}")
                        logging.info(f"Review and execute manually with: bash {rename_script_path}")
                    except Exception as e:
                        logging.error(f"Error setting permissions on rename script: {e}")
                logging.info("Operation gracefully terminated after interrupt")
            else:
                # Direct KeyboardInterrupt without going through signal handler
                print("\nOperation cancelled by user")
                if args.sort and rename_script_path:
                    try:
                        os.chmod(rename_script_path, 0o755)
                        logging.info(f"Made rename script executable: {rename_script_path}")
                        logging.info(f"You can still use the partial rename script: bash {rename_script_path}")
                    except Exception:
                        pass
            return 130
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
