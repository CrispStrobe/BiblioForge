# llm_providers.py
import os
import logging
import threading
import time
from typing import Optional, List, Dict, Any, Union
import re # For sort_author_names
from pathlib import Path # For LlamaCPPProvider model path
import json # For pretty printing debug output
from tqdm import tqdm 
import sys
import json # For pretty-printing

# Attempt to import necessary HTTP client libraries and LLM clients
try:
    import requests
except ImportError:
    requests = None 
try:
    from openai import OpenAI, Timeout as OpenAITimeout, APIConnectionError as OpenAIAPIConnectionError
    import httpx # Used by OpenAI client, good for custom timeouts
except ImportError:
    OpenAI = None 
    OpenAITimeout = None
    OpenAIAPIConnectionError = None
    httpx = None
try:
    from huggingface_hub import InferenceClient, hf_hub_download
except ImportError:
    InferenceClient = None
    hf_hub_download = None # Needed for LlamaCPPProvider
try:
    import cohere 
except ImportError:
    cohere = None
try:
    from groq import Groq
except ImportError:
    Groq = None
try:
    import ollama # Official Ollama library
except ImportError:
    ollama = None
try:
    from llama_cpp import Llama, LlamaGrammar # For LlamaCPPProvider
except ImportError:
    Llama = None
    LlamaGrammar = None

# Import shutdown_flag from utils.
try:
    from utils import shutdown_flag # Assuming utils.py is in the same directory or PYTHONPATH
except ImportError:
    logging.critical("llm_providers.py: CRITICAL - Could not import shutdown_flag from utils. Graceful shutdown might not work.")
    shutdown_flag = threading.Event()


# Thread-local storage for LLM clients that benefit from it
thread_local = threading.local()
# Semaphore for LLM calls to limit concurrency
llm_semaphore = threading.Semaphore(4) 

# Default model names
DEFAULT_OLLAMA_MODEL_NAME = "cas/llama-3.2-3b-instruct:latest"
DEFAULT_LLAMACPP_REPO_ID = "TheBloke/phi-2-GGUF"
DEFAULT_LLAMACPP_FILENAME = "phi-2.Q4_K_M.gguf" 
DEFAULT_LOCAL_OPENAI_MODEL = "local-model" # Often model is selected in the local server UI

# Cache directory for LlamaCPP models
LLAMACPP_MODELS_CACHE_DIR = Path.home() / ".cache" / "biblioforge_llamacpp_models" # Changed from "gguf"
LLAMACPP_MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

class LLMProvider:
    """Base class for LLM providers."""
    def __init__(self, model_name: str, api_key: Optional[str] = None, debug: bool = False): # Added debug
        self.model_name = model_name
        self.api_key = api_key
        self._debug = debug # Store debug flag, making it available to all subclasses
        if not hasattr(self, '_initialized_event'): 
            self._initialized_event = threading.Event()

    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: Optional[int] = 500,
                        timeout_seconds: int = 120) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method.")

class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None, debug: bool = False):
        super().__init__(model_name, api_key, debug=debug) # Pass debug to super
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Either pass as api_key or set OPENAI_API_KEY.")
        if OpenAI is None or httpx is None:
            raise ImportError("OpenAI client library or httpx not installed. pip install openai httpx.")
        self._init_client()
    
    def _init_client(self):
        # Ensures client is initialized once per thread with specific name
        client_attr_name = f"openai_client_{self.model_name.replace('/', '_').replace(':', '_')}"
        if not hasattr(thread_local, client_attr_name):
            if self._debug: logging.debug(f"OpenAIProvider: Initializing new OpenAI client for thread {threading.get_ident()} and model {self.model_name}")
            custom_timeouts = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
            setattr(thread_local, client_attr_name, OpenAI(api_key=self.api_key, timeout=custom_timeouts))
        return getattr(thread_local, client_attr_name)
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: Optional[int] = 500,
                        timeout_seconds: int = 120) -> Dict[str, Any]:
        client = self._init_client()
        with llm_semaphore:
            try:
                if self._debug: logging.debug(f"OpenAIProvider: Sending request to {self.model_name} with timeout {timeout_seconds}s")
                response = client.chat.completions.create(
                    model=self.model_name, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, 
                    timeout=float(timeout_seconds) 
                )
                return {
                    "id": getattr(response, "id", "openai-unknown-id"),
                    "content": response.choices[0].message.content.strip() if response.choices and response.choices[0].message else "",
                    "finish_reason": response.choices[0].finish_reason if response.choices else "unknown",
                    "model": getattr(response, 'model', self.model_name)
                }
            except (OpenAIAPIConnectionError, OpenAITimeout) as e: # Specific OpenAI errors
                logging.error(f"OpenAIProvider ({self.model_name}) connection/timeout error: {e}")
                raise
            except Exception as e:
                logging.error(f"OpenAIProvider ({self.model_name}) request failed: {e}", exc_info=self._debug)
                raise

class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API provider."""
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        if InferenceClient is None:
            raise ImportError("huggingface_hub not installed. pip install huggingface_hub.")

    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: int = 500,
                        timeout_seconds: int = 120) -> Dict[str, Any]:
        prompt = self._format_messages_for_hf(messages)
        # InferenceClient timeout is for the HTTP request itself.
        client = InferenceClient(token=self.api_key, timeout=timeout_seconds) if self.api_key else InferenceClient(timeout=timeout_seconds)
        
        with llm_semaphore:
            try:
                response_text = client.text_generation(
                    prompt, model=self.model_name, max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0.0 else 0.1, # HF temp > 0
                    top_p=0.95 if temperature > 0.0 else None,
                    repetition_penalty=1.1, do_sample=True if temperature > 0.0 else False
                )
                return {
                    "id": "hf-inference-" + str(time.time()), # Generate a pseudo-id
                    "content": str(response_text).strip(),
                    "finish_reason": "stop", 
                    "model": self.model_name
                }
            except Exception as e:
                logging.error(f"HuggingFaceProvider request for {self.model_name} failed: {e}")
                raise
    
    def _format_messages_for_hf(self, messages: List[Dict[str, str]]) -> str:
        formatted_prompt = ""
        system_msgs = [msg for msg in messages if msg["role"] == "system"]
        if system_msgs:
            formatted_prompt = f"<s>[INST] {system_msgs[0]['content']} [/INST]</s>\n\n"
        
        user_assistant_msgs = [m for m in messages if m["role"] != "system"]
        for i, msg in enumerate(user_assistant_msgs):
            if msg["role"] == "user":
                formatted_prompt += f"<s>[INST] {msg['content']} [/INST]"
            elif msg["role"] == "assistant":
                formatted_prompt += f" {msg['content']}</s>"
                # Add newline if there's another user message following
                if i < len(user_assistant_msgs) - 1 and user_assistant_msgs[i+1]["role"] == "user":
                    formatted_prompt += "\n" 
        
        # If the last message was a user message and it's not ending the prompt correctly
        if messages and messages[-1]["role"] == "user" and not formatted_prompt.endswith("</INST>"):
             formatted_prompt += " " # Expecting assistant response
        return formatted_prompt

class CohereProvider(LLMProvider):
    """Cohere API provider."""
    def __init__(self, model_name: str = "command-r", api_key: Optional[str] = None): # Default to command-r
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key: raise ValueError("Cohere API key required.")
        if cohere is None: raise ImportError("Cohere client not installed. pip install cohere.")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: int = 500,
                        timeout_seconds: int = 120) -> Dict[str, Any]:
        with llm_semaphore:
            try:
                client = cohere.Client(api_key=self.api_key, client_name="biblioforge", timeout=timeout_seconds)
                
                system_prompt_content = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
                
                cohere_chat_history = []
                user_assistant_msgs = [msg for msg in messages if msg["role"] in ("user", "assistant")]

                # Build chat history for Cohere API
                for msg in user_assistant_msgs[:-1]: # All messages except the last user message
                    role_cohere = "USER" if msg["role"] == "user" else "CHATBOT"
                    cohere_chat_history.append({"role": role_cohere, "message": msg["content"]})
                
                current_user_message_content = ""
                if user_assistant_msgs and user_assistant_msgs[-1]["role"] == "user":
                    current_user_message_content = user_assistant_msgs[-1]["content"]
                else: # Last message must be from user for cohere.chat
                    logging.warning("Cohere API expects the last message to be from USER. Attempting to use system prompt or will fail.")
                    if system_prompt_content and not user_assistant_msgs : # Only system prompt exists
                        current_user_message_content = system_prompt_content
                        system_prompt_content = None # It's now the main message
                    else: # Cannot form a valid request
                        return {"id": "cohere-no-user-query", "content": "", "finish_reason": "ERROR", "model": self.model_name}

                response = client.chat(
                    model=self.model_name,
                    message=current_user_message_content,
                    chat_history=cohere_chat_history if cohere_chat_history else None,
                    preamble=system_prompt_content,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return {
                    "id": getattr(response, "response_id", "cohere-chat-id"),
                    "content": response.text,
                    "finish_reason": str(response.finish_reason).upper() if hasattr(response, "finish_reason") and response.finish_reason else "UNKNOWN",
                    "model": self.model_name 
                }
            except Exception as e:
                logging.error(f"CohereProvider request for model {self.model_name} failed: {e}")
                raise

class GLHFProvider(LLMProvider):
    """GLHF API provider."""
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("GLHF_API_KEY")
        if not self.api_key: raise ValueError("GLHF API key required.")
        if OpenAI is None: raise ImportError("OpenAI client not installed (for GLHFProvider).")
        self._init_client()
    
    def _init_client(self):
        if not hasattr(thread_local, "glhf_client"):
            thread_local.glhf_client = OpenAI(api_key=self.api_key, base_url="https://glhf.chat/api/openai/v1")
        return thread_local.glhf_client
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: int = 500,
                        timeout_seconds: int = 120) -> Dict[str, Any]:
        client = self._init_client()
        model_id = f"hf:{self.model_name}" if not self.model_name.startswith("hf:") else self.model_name
        with llm_semaphore:
            try:
                response_chunks = []
                completion_response = client.chat.completions.create( # Store the full response
                    stream=True, model=model_id, messages=messages,
                    temperature=temperature, max_tokens=max_tokens, timeout=timeout_seconds
                )
                final_choice_from_stream = None
                for chunk in completion_response:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                        response_chunks.append(chunk.choices[0].delta.content)
                    if chunk.choices and chunk.choices[0].finish_reason:
                        final_choice_from_stream = chunk.choices[0]

                full_response_text = "".join(response_chunks)
                finish_reason_val = final_choice_from_stream.finish_reason if final_choice_from_stream else "stop"
                
                # Try to get an ID from the first chunk of the completion if available, or generate one
                # This is a bit of a guess as stream objects might not have a top-level ID.
                response_id = "glhf-stream-" + str(time.time())
                # If the completion object itself (before iteration) has an ID
                if hasattr(completion_response, 'id') and completion_response.id:
                    response_id = completion_response.id
                elif final_choice_from_stream and hasattr(final_choice_from_stream, 'id') and final_choice_from_stream.id: # Unlikely for stream delta
                    response_id = final_choice_from_stream.id


                return {
                    "id": response_id, 
                    "content": full_response_text, "finish_reason": finish_reason_val, "model": self.model_name
                }
            except Exception as e:
                logging.error(f"GLHFProvider request for {self.model_name} failed: {e}")
                raise

class GroqProvider(LLMProvider):
    """Groq LLM provider."""
    def __init__(self, model_name: str = "llama3-70b-8192", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key: raise ValueError("Groq API key required.")
        if Groq is None: raise ImportError("Groq client not installed. pip install groq.")
        self._init_client()
    
    def _init_client(self):
        if not hasattr(thread_local, "groq_client"):
            # Groq client takes timeout at initialization
            thread_local.groq_client = Groq(api_key=self.api_key, timeout=20.0) # Default timeout for Groq client
        return thread_local.groq_client
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: int = 500,
                        timeout_seconds: int = 120) -> Dict[str, Any]: # timeout_seconds is for API consistency, not directly used by Groq create
        client = self._init_client()
        with llm_semaphore:
            try:
                response = client.chat.completions.create(
                    model=self.model_name, messages=messages,
                    temperature=temperature, max_tokens=max_tokens
                )
                return {
                    "id": getattr(response, "id", "groq-unknown-id"),
                    "content": response.choices[0].message.content.strip() if response.choices and response.choices[0].message else "",
                    "finish_reason": response.choices[0].finish_reason if response.choices else "unknown",
                    "model": getattr(response, 'model', self.model_name)
                }
            except Exception as e:
                logging.error(f"GroqProvider request for {self.model_name} failed: {e}")
                raise

class PoeProvider(LLMProvider):
    """Poe.com LLM provider."""
    def __init__(self, model_name: str = "claude-3-opus", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("POE_API_KEY")
        if not self.api_key: raise ValueError("Poe API key required.")
        if requests is None: raise ImportError("requests library not installed (for PoeProvider).")
        self.api_url = os.environ.get("POE_API_URL", "https://api.poe.com/v1/chat/completions") # Example endpoint

    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: int = 500,
                        timeout_seconds: int = 120) -> Dict[str, Any]:
        system_prompt = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        user_assistant_msgs = [{"role": msg["role"], "content": msg["content"]} for msg in messages if msg["role"] in ("user", "assistant")]
        
        payload = {"model": self.model_name, "messages": user_assistant_msgs, 
                   "temperature": temperature, "max_tokens": max_tokens}
        if system_prompt: 
            # How Poe handles system prompts needs to be confirmed.
            # Common patterns: "system_prompt" field, or prepending to user messages.
            # Assuming a "system_prompt" field for now.
            payload["system_prompt"] = system_prompt 

        headers = {"Authorization": f"Poe-Extra-Long-Term-Token {self.api_key}", # Common Poe auth header
                   "Content-Type": "application/json", "Accept": "application/json"}
        
        with llm_semaphore:
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout_seconds)
                response.raise_for_status()
                response_data = response.json()
                
                content, finish_reason, response_id = "", "unknown", response_data.get("id", "poe-unknown-id")

                # Poe's API response structure can vary. This is a guess.
                if "choices" in response_data and response_data["choices"]:
                    choice = response_data["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"].strip()
                    finish_reason = choice.get("finish_reason", "unknown")
                elif "data" in response_data and isinstance(response_data["data"], dict) and "text" in response_data["data"]:
                    content = response_data["data"]["text"].strip() # Another possible structure
                elif "text" in response_data: content = response_data["text"].strip()
                elif "completion" in response_data: content = response_data["completion"].strip()

                return {
                    "id": response_id, "content": content, 
                    "finish_reason": str(finish_reason).upper() if isinstance(finish_reason, str) else "UNKNOWN", 
                    "model": self.model_name
                }
            except requests.RequestException as e:
                err_content = e.response.text if e.response is not None else "No response content"
                logging.error(f"PoeProvider request for {self.model_name} failed: {e}. Response: {err_content[:200]}")
                raise
            except Exception as e:
                logging.error(f"PoeProvider processing error for {self.model_name}: {e}")
                raise

class LocalOpenAIProvider(LLMProvider):
    """Provider for local LLM servers with an OpenAI-compatible API (e.g., LM Studio)."""
    def __init__(self, model_name: str = DEFAULT_LOCAL_OPENAI_MODEL, 
                 base_url: Optional[str] = None, # Default handled in get_llm_provider
                 debug: bool = False): 
        super().__init__(model_name, debug=debug) 
        self.base_url = base_url or "http://localhost:1234/v1/" # Default if None passed
        if OpenAI is None or httpx is None: 
            raise ImportError("OpenAI client library and httpx not installed (for LocalOpenAIProvider).")
        self._init_client()
    
    def _init_client(self):
        # Unique attribute name per thread based on base_url and model_name for thread_local
        client_attr_name = f"local_openai_client_{re.sub(r'[^a-zA-Z0-9_]', '_', self.base_url)}_{self.model_name.replace('/', '_').replace(':', '_')}"
        if not hasattr(thread_local, client_attr_name):
            if self._debug: logging.debug(f"LocalOpenAIProvider: Initializing new client for thread {threading.get_ident()}, base_url: {self.base_url}, model: {self.model_name}")
            custom_timeouts = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
            client_instance = OpenAI(
                base_url=self.base_url, 
                api_key="local-dummy-key", 
                timeout=custom_timeouts
            )
            setattr(thread_local, client_attr_name, client_instance)
        return getattr(thread_local, client_attr_name)
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: Optional[int] = 500,
                        timeout_seconds: int = 180) -> Dict[str, Any]:
        client = self._init_client()
        with llm_semaphore:
            try:
                if self._debug: logging.debug(f"LocalOpenAIProvider: Sending request to {self.model_name} at {self.base_url} with timeout {timeout_seconds}s")
                response = client.chat.completions.create(
                    model=self.model_name, 
                    messages=messages, 
                    temperature=temperature, 
                    max_tokens=max_tokens,
                    timeout=float(timeout_seconds)
                )
                return {
                    "id": getattr(response, "id", f"local-openai-{time.time()}"),
                    "content": response.choices[0].message.content.strip() if response.choices and response.choices[0].message else "",
                    "finish_reason": response.choices[0].finish_reason if response.choices else "unknown",
                    "model": getattr(response, 'model', self.model_name)
                }
            except (OpenAIAPIConnectionError, OpenAITimeout) as e:
                logging.error(f"LocalOpenAIProvider ({self.model_name} at {self.base_url}) connection/timeout error: {e}")
                raise
            except Exception as e:
                logging.error(f"LocalOpenAIProvider ({self.model_name} at {self.base_url}) request failed: {e}", exc_info=self._debug)
                raise

class OllamaProvider(LLMProvider):
    _checked_models: set = set()

    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL_NAME,
                 host: Optional[str] = None,
                 debug: bool = False,
                 allow_fallback: bool = False,
                 fallback_order: Optional[str] = None):
        super().__init__(model_name, debug=debug)
        if ollama is None:
            raise ImportError("Official 'ollama' Python library not installed. 'pip install ollama'")

        self.original_model_name = model_name
        self.host = host
        self.allow_fallback = allow_fallback
        self.fallback_order = [m.strip() for m in fallback_order.split(',')] if fallback_order else []
        self._client_kwargs = {'host': self.host} if self.host else {}

        if self.model_name not in OllamaProvider._checked_models:
            self._ensure_model_available()
            OllamaProvider._checked_models.add(self.model_name)

    def _ensure_model_available(self):
        """
        Final, definitive, and correct check for model availability. This version
        handles the actual object structure returned by the ollama library.
        """
        if self._debug:
            tqdm.write("\n--- OLLAMA PROVIDER DIAGNOSTIC ---")
            tqdm.write(f"STARTING MODEL CHECK FOR: '{self.original_model_name}'")
        
        try:
            # --- 1. Fetch and CORRECTLY Parse Local Models ---
            models_info_response = ollama.list(**self._client_kwargs)
            
            valid_local_models = []
            # THE FINAL FIX: Check for the 'models' attribute on the response object.
            if hasattr(models_info_response, 'models') and isinstance(models_info_response.models, list):
                for model_obj in models_info_response.models:
                    # Access the 'model' attribute of the Model object.
                    if hasattr(model_obj, 'model') and isinstance(model_obj.model, str):
                        valid_local_models.append(model_obj.model)

            if self._debug:
                tqdm.write(f"PARSED MODEL NAMES ({len(valid_local_models)} found): {valid_local_models}\n")

            # --- 2. Robust Model Matching ---
            target_base, _, target_tag = self.original_model_name.rpartition(':')
            if not target_base:
                target_base, target_tag = self.original_model_name, 'latest'

            for name in valid_local_models:
                local_base, _, local_tag = name.rpartition(':')
                if not local_base:
                    local_base, local_tag = name, 'latest'
                
                if local_base == target_base and local_tag == target_tag:
                    logging.info(f"Ollama: SUCCESS - Found matching local model '{name}'.")
                    self.model_name = name
                    return

            logging.warning(f"Ollama: Desired model '{self.original_model_name}' not found locally.")

            # --- 3. GUARANTEED Fallback Logic ---
            if self.allow_fallback and valid_local_models:
                logging.info("Ollama: Fallback enabled. Finding a substitute...")
                fallback_found = next((p for p in self.fallback_order if p in valid_local_models), None) \
                              or next((n for n in valid_local_models if any(k in n.lower() for k in ["instruct", "chat"])), None) \
                              or valid_local_models[0]
                
                logging.warning(f"Ollama: Switching from '{self.original_model_name}' to fallback '{fallback_found}'.")
                self.model_name = fallback_found
                return

            # --- 4. Pull Logic with Graceful Shutdown ---
            logging.info(f"Ollama: No suitable local model. Proceeding to pull '{self.original_model_name}'.")
            pull_stream = ollama.pull(self.original_model_name, stream=True, **self._client_kwargs)
            self.model_name = self.original_model_name
            
            for _ in pull_stream:
                if shutdown_flag.is_set():
                    raise InterruptedError("Model pull cancelled by user.")
            logging.info(f"Ollama: Model pull for '{self.original_model_name}' completed.")

        except InterruptedError:
            raise
        except Exception as e:
            logging.error(f"Ollama: CRITICAL ERROR in _ensure_model_available: {e}", exc_info=self._debug)
            raise
        finally:
            if self._debug:
                tqdm.write(f"FINAL RESOLVED MODEL: {getattr(self, 'model_name', 'UNKNOWN')}")
                tqdm.write("--- OLLAMA DIAGNOSTIC COMPLETE ---\n")

    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: Optional[int] = 500,
                        timeout_seconds: int = 120) -> Dict[str, Any]:
        options = {"temperature": temperature}
        if max_tokens is not None and max_tokens > 0:
            options["num_predict"] = max_tokens
        
        with llm_semaphore:
            if shutdown_flag.is_set():
                raise InterruptedError("Chat completion cancelled by shutdown signal.")
            
            response = ollama.chat(
                model=self.model_name, messages=messages, stream=False,
                options=options, **self._client_kwargs
            )
            content = response.get('message', {}).get('content', '').strip()
            return {
                "id": f"ollama-{response.get('created_at', str(time.time()))}",
                "content": content, "finish_reason": "stop",
                "model": response.get('model', self.model_name)
            }

class LlamaCPPProvider(LLMProvider):
    """Provider for llama-cpp-python."""
    _loaded_models: Dict[str, Any] = {} 
    _model_load_lock = threading.Lock()

    def __init__(self, 
                 model_name_or_path: str = DEFAULT_LLAMACPP_REPO_ID, 
                 model_gguf_filename: Optional[str] = DEFAULT_LLAMACPP_FILENAME, 
                 n_ctx: int = 2048, 
                 n_gpu_layers: int = 0, 
                 verbose_llamacpp: bool = False, # Renamed to avoid conflict with self._debug
                 chat_format: Optional[str] = "llama-2",
                 debug: bool = False): # For LLMProvider base class
        
        display_model_name = f"{model_name_or_path}/{model_gguf_filename}" if model_gguf_filename and "/" not in model_name_or_path else model_name_or_path
        super().__init__(display_model_name, debug=debug)

        if Llama is None or hf_hub_download is None:
            raise ImportError("llama-cpp-python or huggingface_hub not installed. Run 'pip install llama-cpp-python huggingface_hub'")

        # Use a combination that uniquely identifies the model and its load parameters for caching key
        self.model_path_key = f"{model_name_or_path}_{model_gguf_filename or ''}_{n_ctx}_{n_gpu_layers}"
        self.model_name_or_path = model_name_or_path
        self.model_gguf_filename = model_gguf_filename
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llama_cpp_verbose = verbose_llamacpp # Store separately
        self.chat_format = chat_format
        
        self._init_client()

    def _resolve_model_path(self) -> str:
        resolved_path = Path(self.model_name_or_path)
        if resolved_path.is_file() and resolved_path.suffix.lower() == ".gguf":
            if self._debug: logging.debug(f"LlamaCPPProvider: Using local model path: {resolved_path}")
            return str(resolved_path)
        elif self.model_gguf_filename: 
            if self._debug: logging.debug(f"LlamaCPPProvider: Attempting to download model '{self.model_gguf_filename}' from HF repo '{self.model_name_or_path}' to cache '{LLAMACPP_MODELS_CACHE_DIR}'.")
            try:
                model_downloaded_path = hf_hub_download(
                    repo_id=self.model_name_or_path,
                    filename=self.model_gguf_filename,
                    cache_dir=LLAMACPP_MODELS_CACHE_DIR,
                    resume_download=True
                )
                if self._debug: logging.debug(f"LlamaCPPProvider: Model path resolved to: {model_downloaded_path}")
                return model_downloaded_path
            except Exception as e:
                logging.error(f"LlamaCPPProvider: Failed to download model '{self.model_gguf_filename}' from '{self.model_name_or_path}': {e}", exc_info=self._debug)
                raise
        else:
            msg = ("LlamaCPPProvider: model_name_or_path must be a local .gguf file path, "
                   "OR a HuggingFace repo_id with model_gguf_filename specified.")
            logging.error(msg)
            raise ValueError(msg)

    def _init_client(self):
        client_attr_name = f"llama_cpp_client_{self.model_path_key.replace('/', '_').replace('.', '_').replace('-', '_')}" # Sanitize key further

        if not hasattr(thread_local, client_attr_name):
            with LlamaCPPProvider._model_load_lock: 
                if not hasattr(thread_local, client_attr_name): # Double check after lock
                    actual_model_path = self._resolve_model_path()
                    if self._debug: logging.info(f"LlamaCPPProvider: Initializing Llama model from: {actual_model_path} "
                                     f"(n_ctx={self.n_ctx}, n_gpu_layers={self.n_gpu_layers}, chat_format='{self.chat_format}')")
                    try:
                        llama_instance = Llama(
                            model_path=actual_model_path,
                            n_ctx=self.n_ctx,
                            n_gpu_layers=self.n_gpu_layers,
                            verbose=self.llama_cpp_verbose, # Use specific verbose for llama.cpp
                            chat_format=self.chat_format
                        )
                        setattr(thread_local, client_attr_name, llama_instance)
                        if self._debug: logging.info(f"LlamaCPPProvider: Model '{self.model_name}' loaded successfully for thread {threading.get_ident()}.")
                    except Exception as e:
                        logging.error(f"LlamaCPPProvider: Error loading Llama model '{self.model_name}': {e}", exc_info=self._debug)
                        setattr(thread_local, client_attr_name, None) 
                        raise
        
        client = getattr(thread_local, client_attr_name, None)
        if client is None:
            raise RuntimeError(f"LlamaCPPProvider: Client for model {self.model_name} could not be initialized in thread {threading.get_ident()}.")
        return client

    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: Optional[int] = 500, 
                        timeout_seconds: int = 180) -> Dict[str, Any]: # timeout_seconds is aspirational here
        
        client: Llama = self._init_client() 
        
        effective_max_tokens = max_tokens
        if max_tokens is None or max_tokens <= 0: 
            effective_max_tokens = self.n_ctx // 2 
            if self._debug: logging.debug(f"LlamaCPPProvider: max_tokens not specified or invalid, defaulting to n_ctx/2 = {effective_max_tokens}")
            
        with llm_semaphore:
            try:
                if self._debug: logging.debug(f"LlamaCPPProvider: Sending chat completion request to '{self.model_name}' with temp={temperature}, max_tokens={effective_max_tokens}")
                response = client.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                )
                content = response['choices'][0]['message']['content'].strip() if response and response.get('choices') and response['choices'][0].get('message') else ""
                finish_reason = response['choices'][0].get('finish_reason', 'unknown') if response and response.get('choices') else "unknown"
                response_id = response.get('id', f"llama_cpp-{time.time()}")
                
                if self._debug: logging.debug(f"LlamaCPPProvider: Response received. Content length: {len(content)}, Finish reason: {finish_reason}")

                return {
                    "id": response_id,
                    "content": content,
                    "finish_reason": finish_reason,
                    "model": self.model_name 
                }
            except Exception as e:
                logging.error(f"LlamaCPPProvider ({self.model_name}) chat completion failed: {e}", exc_info=self._debug)
                raise

# --- Main Factory Function ---
def get_llm_provider(provider_type: str = "ollama", 
                     model_name: Optional[str] = None, 
                     api_key: Optional[str] = None,
                     debug: bool = False, # Added debug to be passed to providers
                     **kwargs # For provider-specific constructor args
                     ) -> LLMProvider:
    provider_type_lower = provider_type.lower()
    
    # Default model names for each provider type if not specified by user
    default_models = {
        "ollama": DEFAULT_OLLAMA_MODEL_NAME, 
        "groq": "llama3-8b-8192", 
        "openai": "gpt-3.5-turbo", 
        # ... (ensure other defaults are here)
        "local_openai": DEFAULT_LOCAL_OPENAI_MODEL,
        "llama_cpp": f"{kwargs.get('llamacpp_repo_id', DEFAULT_LLAMACPP_REPO_ID)}/"
                     f"{kwargs.get('llamacpp_gguf_filename', DEFAULT_LLAMACPP_FILENAME)}"
    }
    
    effective_model_name = model_name or default_models.get(provider_type_lower, "default_model_not_in_map")
    # For llama_cpp, if model_name is provided, it might be the full display name or just repo.
    # The LlamaCPPProvider handles parsing this if model_name_or_path is set from effective_model_name.
    if provider_type_lower == "llama_cpp" and not model_name and not kwargs.get('llamacpp_repo_id'):
        # If no specific model/repo for llama_cpp, use its structured default
        effective_model_name = default_models["llama_cpp"]

    provider_map: Dict[str, type[LLMProvider]] = {
        "ollama": OllamaProvider, 
        "openai": OpenAIProvider,
        "local_openai": LocalOpenAIProvider,
        "llama_cpp": LlamaCPPProvider,
        # Add other providers here: GroqProvider, CohereProvider, etc.
        "groq": GroqProvider, 
        "cohere": CohereProvider, 
        "huggingface": HuggingFaceProvider,
        "glhf": GLHFProvider, 
        "poe": PoeProvider,
    }

    if provider_type_lower in provider_map:
        ProviderClass = provider_map[provider_type_lower]
        if debug: logging.debug(f"get_llm_provider: Initializing '{provider_type_lower}' with effective_model_name='{effective_model_name}' and debug={debug}")
        try:
            if provider_type_lower == "ollama":
                 return ProviderClass(
                     model_name=effective_model_name, 
                     host=kwargs.get('ollama_host'), 
                     debug=debug,
                     allow_fallback=kwargs.get('ollama_allow_fallback', False),
                     fallback_order=kwargs.get('ollama_fallback_order')
                 )
            elif provider_type_lower == "local_openai":
                 return ProviderClass(model_name=effective_model_name, base_url=kwargs.get('local_openai_base_url'), debug=debug)
            elif provider_type_lower == "llama_cpp":
                 # If model_name was given, it might be a local path or HF "repo/file" string.
                 # LlamaCPPProvider's __init__ should parse model_name_or_path and model_gguf_filename.
                 # If args.llm_model was set, use it as the primary source for model_name_or_path.
                 # If args.llamacpp_repo_id is set, it takes precedence for repo.
                 
                 # Determine model_name_or_path and model_gguf_filename for LlamaCPPProvider
                 mn_or_path = kwargs.get('llamacpp_repo_id', model_name if model_name and "/" in model_name else None) or DEFAULT_LLAMACPP_REPO_ID
                 gguf_file = kwargs.get('llamacpp_gguf_filename')
                 if not gguf_file and model_name and model_name.lower().endswith(".gguf") and "/" not in model_name:
                     gguf_file = os.path.basename(model_name) # if model_name is just "file.gguf"
                 elif not gguf_file and "/" in mn_or_path and mn_or_path.lower().endswith(".gguf"): # if repo_id was actually "repo/file.gguf"
                     mn_or_path, gguf_file = os.path.split(mn_or_path)
                 elif not gguf_file:
                     gguf_file = DEFAULT_LLAMACPP_FILENAME


                 return ProviderClass(
                     model_name_or_path=mn_or_path, 
                     model_gguf_filename=gguf_file,
                     n_ctx=kwargs.get('llamacpp_n_ctx', 2048), # Use explicit default here too
                     n_gpu_layers=kwargs.get('llamacpp_n_gpu_layers', 0),
                     chat_format=kwargs.get('llamacpp_chat_format', "llama-2"),
                     verbose_llamacpp=debug, # Pass main debug flag to LlamaCPP's verbose
                     debug=debug # Pass to LLMProvider base
                    )
            else: # For cloud providers primarily needing api_key and model_name
                 return ProviderClass(model_name=effective_model_name, api_key=api_key, debug=debug)
        except (ImportError, ValueError, Exception) as e: # Broader exception for init issues
            logging.error(f"Error initializing '{provider_type_lower}' provider (model: '{effective_model_name}'): {e}", exc_info=debug)
            raise 
    else:
        raise ValueError(f"Unknown LLM provider type: {provider_type_lower}")


# --- Core LLM Interaction Functions  ---
def fix_malformed_xml_tags(text: str) -> str:
    """
    Fix common XML tag malformations from LLM responses
    """
    import re
    
    # Fix tags with equals signs: <TAG=value</TAG> -> <TAG>value</TAG>
    # Pattern: <(TAG)[=\s]+([^>]*)</TAG>
    def fix_malformed_tag(match):
        tag_name = match.group(1)
        content = match.group(2).strip()
        # Remove quotes if present
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        elif content.startswith("'") and content.endswith("'"):
            content = content[1:-1]
        return f"<{tag_name}>{content}</{tag_name}>"
    
    # Fix patterns like <AUTHOR=Watanabe, Morimichi</AUTHOR>
    text = re.sub(r'<(TITLE|AUTHOR|YEAR|LANGUAGE)[=\s]+([^>]*)</\1>', fix_malformed_tag, text, flags=re.IGNORECASE)
    
    # Fix unclosed malformed tags: <TAG=value> -> <TAG>value</TAG>
    text = re.sub(r'<(TITLE|AUTHOR|YEAR|LANGUAGE)[=\s]+([^<>]+)(?=\s*(?:<|$))', r'<\1>\2</\1>', text, flags=re.IGNORECASE)
    
    return text

def send_to_llm(text: str, filename: str, provider_instance: LLMProvider,
                max_attempts: int = 3, # Max attempts for the *same* prompt template
                verbose: bool = False,
                temperature_arg: float = 0.5, max_tokens_arg: Optional[int] = 300,
                prompt_template_index: int = 0  # for retries with different templates
                ) -> str:
    base_retry_wait = 5.0
    # Using the same prompt templates as before
    prompt_templates_definition = [
        ( # This is a TUPLE of strings
            f"Extract metadata from the following file extraction snippet. We need "
            f"(1) the publication title, "
            f"(2) the year of publication (4 digits), "
            f"(3) the main author name (format: Lastname Firstname), and "
            f"(4) the document language (2-letter ISO code) "
            f"from the text below. Also consider the filename '{os.path.basename(filename)}' for clues. "
            f"Respond ONLY in the following exact format, with no extra text or explanations: \n"
            f"<TITLE>Extracted Publication Title</TITLE>\n<YEAR>YYYY</YEAR>\n<AUTHOR>Lastname Firstname</AUTHOR>\n<LANGUAGE>lg</LANGUAGE>\n\n"
        ),
        ( # This is a SINGLE string (but can be kept as a tuple of one string for consistency if preferred)
            f"I need to extract metadata from a document with filename '{os.path.basename(filename)}'. "
            f"Provide ONLY these four tags with the information. Do not add any other text.\n"
            f"<TITLE>The Exact Title of the Publication</TITLE>\n"
            f"<YEAR>YYYY (The 4-digit publication year)</YEAR>\n"
            f"<AUTHOR>Lastname Firstname of first/main Author</AUTHOR>\n"
            f"<LANGUAGE>lg (The 2-letter language code, e.g., en, de, fr)</LANGUAGE>\n\n"
        ),
        (
            f"You are an expert metadata extraction tool. From the provided text (and filename '{os.path.basename(filename)}'), extract the following fields:\n"
            f"1. TITLE: The complete and exact title of the publication.\n"
            f"2. YEAR: The 4-digit year of publication. If not found, use 'UnknownYear'.\n"
            f"3. AUTHOR: The primary author's name, formatted as 'Lastname Firstname'. If multiple, list only the first.\n"
            f"4. LANGUAGE: The 2-letter ISO 639-1 language code (e.g., en, de, fr). If unsure, use 'ul'.\n"
            f"Format your response using ONLY these XML-like tags, with no additional commentary:\n"
            f"<TITLE>...</TITLE>\n<YEAR>...</YEAR>\n<AUTHOR>...</AUTHOR>\n<LANGUAGE>...</LANGUAGE>\n\n"
        )
    ]

    # Ensure the chosen prompt_template_index is valid
    if not 0 <= prompt_template_index < len(prompt_templates_definition):
        logging.error(f"send_to_llm: Invalid prompt_template_index {prompt_template_index}. Defaulting to 0.")
        prompt_template_index = 0
    
    chosen_template_config = prompt_templates_definition[prompt_template_index]
    
    # Handle if the template config is a tuple of strings or a single string
    if isinstance(chosen_template_config, tuple):
        prompt_template_str = "".join(chosen_template_config)
    else:
        prompt_template_str = chosen_template_config

    provider_debug_flag = getattr(provider_instance, '_debug', False)

    for attempt in range(1, max_attempts + 1):
        if shutdown_flag.is_set():
            logging.info("Shutdown: Aborting LLM request in send_to_llm.")
            return ""

        if provider_debug_flag or verbose:
            logging.debug(f"send_to_llm: Request for '{filename}' to {provider_instance.__class__.__name__} ('{provider_instance.model_name}'), "
                          f"using prompt template index {prompt_template_index} (Attempt {attempt}/{max_attempts} for this template).")

        prompt = prompt_template_str + f"Document text (first 3000 chars):\n{text[:3000]}"
        messages = [{"role": "user", "content": prompt}]

        # <<< VERBOSE LOGGING OF SENT DATA >>>
        if provider_debug_flag or verbose:
            try:
                messages_str_for_log = json.dumps(messages, indent=2, ensure_ascii=False)
                logging.debug(f"send_to_llm: Constructed messages payload for '{filename}' (template index {prompt_template_index}, attempt {attempt}):\n{messages_str_for_log}")
            except Exception as e_log:
                logging.debug(f"send_to_llm: Could not serialize messages payload for logging: {e_log}")

        # Define the network-related exceptions to catch.
        network_exceptions = []
        if OpenAIAPIConnectionError:
            network_exceptions.append(OpenAIAPIConnectionError)
        if OpenAITimeout:
            network_exceptions.append(OpenAITimeout)
        if ollama and hasattr(ollama, 'ResponseError'):
            network_exceptions.append(ollama.ResponseError)
        
        # Ensure the tuple is not empty before using it.
        # Fallback to a base error if no specific libraries are installed.
        exceptions_to_catch = tuple(network_exceptions) if network_exceptions else (RuntimeError,)

        try:
            response_data = provider_instance.chat_completion(
                messages=messages, temperature=temperature_arg,
                max_tokens=max_tokens_arg,
                timeout_seconds=120 # Default overall timeout for the call
            )
            output = response_data.get("content", "").strip()
            if provider_debug_flag or verbose: 
                logging.debug(f"send_to_llm: Raw LLM response for '{filename}' (template index {prompt_template_index}, attempt {attempt}): '{output}'")

            
            def has_valid_tag(output_text: str, tag_name: str) -> bool:
                """Check if output contains a valid XML tag, handling malformed variations"""
                import re
                # Check for proper format: <TAG>content</TAG>
                proper_pattern = f"<{tag_name}>"
                if proper_pattern in output_text:
                    return True
                
                # Check for malformed format with equals: <TAG=content</TAG> or <TAG="content"</TAG>
                malformed_pattern = rf"<{tag_name}[=\s][^>]*>"
                if re.search(malformed_pattern, output_text, re.IGNORECASE):
                    if provider_debug_flag or verbose:
                        logging.debug(f"send_to_llm: Found malformed {tag_name} tag in response, will attempt to fix")
                    return True
                
                return False

            has_title = has_valid_tag(output, "TITLE")
            has_author = has_valid_tag(output, "AUTHOR") 
            has_year = has_valid_tag(output, "YEAR")

            if output and has_title and has_author and has_year:
                # FIXED: Clean up malformed tags before returning
                cleaned_output = fix_malformed_xml_tags(output)
                return cleaned_output
            elif output and (has_valid_tag(output, "PUBLICATION TITLE") or has_valid_tag(output, "PUBLICATIONTITLE")) and \
                 has_author and has_year:
                if provider_debug_flag or verbose: 
                    logging.debug(f"send_to_llm: LLM for '{filename}' used alternative title tag. Standardizing.")
                cleaned_output = fix_malformed_xml_tags(output)
                cleaned_output = cleaned_output.replace("<PUBLICATION TITLE>", "<TITLE>").replace("<PUBLICATIONTITLE>", "<TITLE>")
                cleaned_output = cleaned_output.replace("</PUBLICATION TITLE>", "</TITLE>").replace("</PUBLICATIONTITLE>", "</TITLE>")
                return cleaned_output
            else:
                # This log means the response was received but was structurally bad for THIS attempt with THIS template
                logging.warning(f"send_to_llm: LLM response for '{filename}' (template index {prompt_template_index}, attempt {attempt}, model '{provider_instance.model_name}') "
                                f"missing key tags (TITLE, AUTHOR, YEAR) or empty: '{output[:100]}...'")
                # If max_attempts for this specific prompt template is reached, return the malformed output.
                if attempt == max_attempts:
                    logging.error(f"send_to_llm: Max attempts ({max_attempts}) reached for '{filename}' with prompt template index {prompt_template_index}. Returning last (malformed) output.")
                    return fix_malformed_xml_tags(output) # Still try to fix what we can

        except exceptions_to_catch as e_net:
            # This block is now valid and will catch the intended errors.
            log_msg = f"send_to_llm: LLM network/timeout error for '{filename}' with {provider_instance.__class__.__name__} (template index {prompt_template_index}, attempt {attempt}): {e_net}."
            if ollama and isinstance(e_net, ollama.ResponseError):
                log_msg += f" Status: {e_net.status_code}, Error: {e_net.error}"
            logging.warning(log_msg)
            if attempt < max_attempts:
                sleep_time = base_retry_wait * attempt
                logging.info(f"send_to_llm: Retrying prompt template index {prompt_template_index} in {sleep_time:.1f}s.")
                time.sleep(sleep_time)
            else:
                logging.error(f"send_to_llm: Max LLM retries ({max_attempts}) for '{filename}' with prompt template index {prompt_template_index} due to network/timeout errors.")
                return ""
        except Exception as e:
            wait_time = base_retry_wait * (1.5 ** (attempt - 1))
            logging.warning(f"send_to_llm: LLM call error for '{filename}' with {provider_instance.__class__.__name__} "
                            f"(template index {prompt_template_index}, attempt {attempt}): {e}. Retrying in {wait_time:.1f}s.", exc_info=provider_debug_flag or verbose)

            if attempt < max_attempts:
                time.sleep(wait_time)
            else:
                logging.error(f"send_to_llm: Max LLM retries ({max_attempts}) for '{filename}' with prompt template index {prompt_template_index} due to errors.")
                return "" # Return empty if all retries for this template fail due to other errors

    return "" # Should be reached if all attempts for the given prompt_template_index fail


def extract_metadata(text: str, filename: str,
                         llm_provider_arg: Union[str, LLMProvider, None],
                         model_name_arg: Optional[str] = None,
                         api_key_arg: Optional[str] = None,
                         verbose: bool = False, # For send_to_llm's own logging
                         prompt_template_index_to_try: int = 0, # fo retries
                         **provider_constructor_kwargs # Catches ollama_host, llamacpp_*, local_openai_base_url, debug flag
                         ) -> str:
    provider_instance: Optional[LLMProvider] = None

    debug_for_provider = provider_constructor_kwargs.get('debug', False)

    if isinstance(llm_provider_arg, LLMProvider):
        provider_instance = llm_provider_arg
    else:
        provider_name_str = llm_provider_arg if isinstance(llm_provider_arg, str) else "ollama"
        try:
            provider_instance = get_llm_provider(
                provider_type=provider_name_str,
                model_name=model_name_arg,
                api_key=api_key_arg,
                debug=debug_for_provider,
                **provider_constructor_kwargs
            )
        except Exception as e:
            logging.error(f"llm_extract_metadata: Failed to get LLMProvider '{provider_name_str}': {e}", exc_info=debug_for_provider)
            return ""

    if not provider_instance:
        logging.error("llm_extract_metadata: LLM provider instance could not be resolved.")
        return ""

    temp_for_metadata = provider_constructor_kwargs.get('temperature', 0.5)
    max_tokens_for_metadata = provider_constructor_kwargs.get('max_tokens', 300)

    return send_to_llm(
        text=text, filename=filename,
        provider_instance=provider_instance,
        verbose=verbose, 
        temperature_arg=temp_for_metadata,
        max_tokens_arg=max_tokens_for_metadata,
        prompt_template_index=prompt_template_index_to_try # for retries with different prompt templates
    )

def sort_author_names(
    author_names_input: str, 
    provider_arg, 
    verbose: bool = False, 
    filename_for_logging: str = "Unknown", 
    prompt_template_index: int = 0,
    **kwargs
) -> str:
    """
    Sort author names using LLM with multiple prompt templates for better reliability.
    """
    
    # Multiple prompt templates for author sorting
    author_sorting_prompts = [
        # Template 0: Direct and simple
        f"""Convert this author name to "Lastname Firstname" format. If there are multiple authors, use only the first/main author.

Author name: "{author_names_input}"

Respond with ONLY the sorted name in this exact format:
<AUTHOR>Lastname Firstname</AUTHOR>

Examples:
- "Smith John" → <AUTHOR>Smith John</AUTHOR>
- "John Smith" → <AUTHOR>Smith John</AUTHOR>
- "Smith, John" → <AUTHOR>Smith John</AUTHOR>
- "Dr. John Smith" → <AUTHOR>Smith John</AUTHOR>""",

        # Template 1: More detailed with examples
        f"""Please reformat the author name into "Lastname Firstname" order. Extract only the main author if multiple authors are present.

Input: "{author_names_input}"

Rules:
1. Put family name (surname) first
2. Put given name (first name) second
3. Remove titles like Dr., Prof., etc.
4. Use only the primary author if multiple authors
5. Keep original spelling and accents

Format your response as: <AUTHOR>Lastname Firstname</AUTHOR>

Examples:
- "Maria Garcia" → <AUTHOR>Garcia Maria</AUTHOR>
- "Prof. Hans Müller" → <AUTHOR>Müller Hans</AUTHOR>
- "Dr. Meier, Klaus" → <AUTHOR>Meier Klaus</AUTHOR>""",

        # Template 2: Academic focus with international considerations
        f"""Reorder this academic author name to surname-first format. Handle international names appropriately.

Author: "{author_names_input}"

Instructions:
- Convert to: Surname GivenName
- For Western names: "John Smith" becomes "Smith John"
- For names with particles: "van der Berg, Jan" becomes "van der Berg Jan"
- For hyphenated surnames, keep them together: "Carsten Schlüter-Knauer" becomes "Schlüter-Knauer Carsten" # <--- ADD THIS EXAMPLE
- Remove academic titles (Dr., Prof., etc.)
- If uncertain about name order, judge according to your knowledge about what is likelier as a Firstname.

Response format: <AUTHOR>Surname GivenName</AUTHOR>

Sample conversions:
- "Thomas Anderson" → <AUTHOR>Anderson Thomas</AUTHOR>
- "Marie-Claire Dubois" → <AUTHOR>Dubois Marie-Claire</AUTHOR>
- "José García López" → <AUTHOR>García López José</AUTHOR>"""
    ]
    
    if prompt_template_index >= len(author_sorting_prompts):
        prompt_template_index = 0  # Fallback to first template
    
    selected_prompt = author_sorting_prompts[prompt_template_index]
    
    if verbose:
        logging.debug(f"sort_author_names: Using prompt template {prompt_template_index} for '{author_names_input}' (file: {filename_for_logging})")
    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            if verbose:
                logging.debug(f"sort_author_names: LLM request for '{author_names_input}' (file: {filename_for_logging}) to {type(provider_arg).__name__} ('{provider_arg.model_name}'), template {prompt_template_index}, attempt {attempt + 1}")
            
            # FIX: Use chat_completion instead of send_request
            response = provider_arg.chat_completion(
                messages=[{"role": "user", "content": selected_prompt}],
                temperature=0.1,  # Low temperature for consistent sorting
                max_tokens=100,
                timeout_seconds=30
            )
            
            # Extract content from the response dictionary
            response_text = response.get("content", "") if isinstance(response, dict) else str(response)
            
            if verbose:
                logging.debug(f"sort_author_names: Raw LLM response for '{author_names_input}': '{response_text}'")
            
            if not response_text:
                if verbose:
                    logging.warning(f"sort_author_names: Empty response for '{author_names_input}' (template {prompt_template_index}, attempt {attempt + 1})")
                continue
            
            # Extract author name from response
            author_match = re.search(r'<AUTHOR[^>]*>(.*?)</AUTHOR>', response_text, re.DOTALL | re.IGNORECASE)
            if author_match:
                sorted_author = author_match.group(1).strip()
                
                # Clean up the result
                sorted_author = re.sub(r'\s+', ' ', sorted_author)  # Normalize whitespace
                sorted_author = re.sub(r'^(Dr\.?|Prof\.?|Mr\.?|Mrs\.?|Ms\.?)\s+', '', sorted_author, flags=re.IGNORECASE)  # Remove titles
                
                if sorted_author and len(sorted_author.strip()) > 1:
                    if verbose:
                        logging.debug(f"sort_author_names: LLM sorted '{author_names_input}' to '{sorted_author}' using {provider_arg.model_name} (template {prompt_template_index}) for {filename_for_logging}")
                    return sorted_author
                else:
                    if verbose:
                        logging.warning(f"sort_author_names: Extracted author name too short: '{sorted_author}' (template {prompt_template_index}, attempt {attempt + 1})")
            else:
                if verbose:
                    logging.warning(f"sort_author_names: Could not extract <AUTHOR> tags from response: '{response_text[:100]}...' (template {prompt_template_index}, attempt {attempt + 1})")
        
        except Exception as e:
            if verbose:
                logging.warning(f"sort_author_names: LLM request failed for '{author_names_input}' (template {prompt_template_index}, attempt {attempt + 1}): {e}")
            continue
    
    # All attempts failed
    if verbose:
        logging.warning(f"sort_author_names: All attempts failed for '{author_names_input}' with template {prompt_template_index}, returning original")
    
    return author_names_input  # Return original if all attempts fail

def sort_author_names_old(author_names_input: str, 
                      provider_arg: Union[str, LLMProvider, None], 
                      temperature: float = 0.2, # Specific temperature for this task
                      max_tokens: Optional[int] = 100, # Specific max_tokens for this task 
                      max_attempts: int = 2, 
                      verbose: bool = False, # For sort_author_names' direct logging
                      filename_for_logging: str = "UnknownFile", 
                      model_name_arg: Optional[str] = None,
                      api_key_arg: Optional[str] = None,
                      **provider_constructor_kwargs # Catches debug and other provider configs
                      ) -> str:
    if not author_names_input or not isinstance(author_names_input, str):
        return "UnknownAuthor"
    
    debug_for_provider = provider_constructor_kwargs.get('debug', False)
    
    # -- cleaning and heuristic pre-sort logic --
    cleaned_input_name = re.sub(r"^\s*<AUTHOR>\s*|\s*</AUTHOR>\s*$", "", author_names_input, flags=re.IGNORECASE).strip()
    cleaned_input_name = re.sub(r"^(Main Author: Lastname Firstname:|Main Author:)\s*", "", cleaned_input_name, flags=re.IGNORECASE).strip()
    if not cleaned_input_name or cleaned_input_name.lower() in ["unknown", "unknownauthor", "n a", ""]: return "UnknownAuthor"
    if ',' in cleaned_input_name:
        parts = [p.strip() for p in cleaned_input_name.split(',', 1)]
        if len(parts) == 2 and parts[0] and parts[1] and len(parts[1].split()) <= 3:
            reformatted = f"{parts[0]} {parts[1]}"; reformatted = re.sub(r'\s+', ' ', reformatted).strip()
            if verbose: logging.debug(f"sort_author_names: Pre-LLM comma sort for '{cleaned_input_name}' -> '{reformatted}' (file: {filename_for_logging})")
            return reformatted
    author_to_process = cleaned_input_name.split(';')[0].strip()
    if not author_to_process: return "UnknownAuthor"

    llm_instance: Optional[LLMProvider] = None
    if isinstance(provider_arg, LLMProvider):
        llm_instance = provider_arg
    else: 
        provider_name_str = provider_arg if isinstance(provider_arg, str) else "ollama"
        try:
            llm_instance = get_llm_provider(
                provider_type=provider_name_str, 
                model_name=model_name_arg, 
                api_key=api_key_arg,
                debug=debug_for_provider,
                **provider_constructor_kwargs
            )
        except Exception as e:
            logging.error(f"sort_author_names: Failed to get LLMProvider '{provider_name_str}': {e}. Returning '{author_to_process}'.", exc_info=debug_for_provider)
            return author_to_process 

    if not llm_instance:
        logging.error(f"sort_author_names: LLM instance is None for '{author_to_process}'. Returning as-is.")
        return author_to_process

    base_retry_wait = 1.0
    for attempt in range(1, max_attempts + 1):
        if shutdown_flag.is_set(): return author_to_process 

        prompt = ( # ... (prompt for author sorting remains the same) ...
            f"Reformat the author name '{author_to_process}' into the standard 'Lastname Firstname' format. "
            f"If it's a single name (e.g., 'Plato'), return it as is. "
            f"If it's an organization, return it as is. "
            f"Handle particles like 'van', 'de la', 'von' correctly (e.g., 'Vincent van Gogh' should become 'van Gogh Vincent'). "
            f"Prioritize 'Lastname Firstname'. Judge according to your knowledge about what names are likelier Firstnames."
            f"Respond ONLY with the formatted name inside <AUTHOR></AUTHOR> tags. Examples:\n"
            f"Input: John Doe -> Output: <AUTHOR>Doe John</AUTHOR>\n"
            f"Input: Plato -> Output: <AUTHOR>Plato</AUTHOR>\n"
            f"Input: J. R. R. Tolkien -> Output: <AUTHOR>Tolkien JRR</AUTHOR>\n"
            f"Input Author: {author_to_process}"
        )
        messages = [{"role": "user", "content": prompt}]
        
        provider_debug_flag = getattr(llm_instance, '_debug', False)

        with llm_semaphore:
            try:
                if provider_debug_flag or verbose: 
                    logging.debug(f"sort_author_names: LLM request for '{author_to_process}' (file: {filename_for_logging}) to {llm_instance.__class__.__name__} ('{llm_instance.model_name}'), attempt {attempt}")
                
                response = llm_instance.chat_completion(
                    messages=messages, temperature=temperature, max_tokens=max_tokens, timeout_seconds=30
                )
                llm_output = response.get("content", "").strip()
                if provider_debug_flag or verbose: logging.debug(f"sort_author_names: Raw LLM response for '{author_to_process}': '{llm_output}'")


                if llm_output:
                    name_match = re.search(r'<AUTHOR>(.*?)</AUTHOR>', llm_output, re.DOTALL | re.IGNORECASE)
                    if name_match:
                        ordered_name = name_match.group(1).strip()
                        ordered_name = re.sub(r'\s+', ' ', ordered_name).strip()
                        if ordered_name and ordered_name.lower() not in ["lastname firstname", "unknownauthor", "unknown author"]:
                            if verbose: logging.debug(f"sort_author_names: LLM sorted '{author_to_process}' to '{ordered_name}' using {llm_instance.model_name} for {filename_for_logging}")
                            return ordered_name
                    else: 
                        plain_name = re.sub(r'<[^>]+>', '', llm_output).strip()
                        if plain_name and len(plain_name.split()) >= 1 and len(plain_name) < 70 and plain_name.lower() not in ["lastname firstname", "unknownauthor", "unknown author"]:
                             if verbose: logging.debug(f"sort_author_names: LLM sorted (no tags) '{author_to_process}' to '{plain_name}' for {filename_for_logging}")
                             return plain_name
                    logging.warning(f"sort_author_names: LLM response for '{author_to_process}' (file: {filename_for_logging}) was invalid: '{llm_output[:100]}'")
            except Exception as e:
                logging.warning(f"sort_author_names: LLM error sorting '{author_to_process}' (attempt {attempt}, model {llm_instance.model_name}, file: {filename_for_logging}): {e}", exc_info=provider_debug_flag or verbose)
        
        if attempt < max_attempts: time.sleep(base_retry_wait * (1.5 ** (attempt - 1)))
    
    logging.warning(f"sort_author_names: Could not reliably sort '{author_names_input}' (processed as '{author_to_process}') for {filename_for_logging}. Returning best cleaned input.")
    return author_to_process