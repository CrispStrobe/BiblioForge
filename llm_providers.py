# llm_providers.py

import os
import logging
import threading
import time
from typing import Optional, List, Dict, Any, Union
import re # For sort_author_names
from pathlib import Path # For LlamaCPPProvider model path

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
    from utils import shutdown_flag
except ImportError:
    logging.critical("llm_providers.py: CRITICAL - Could not import shutdown_flag from utils. Graceful shutdown might not work.")
    shutdown_flag = threading.Event()


# Thread-local storage for LLM clients that benefit from it
thread_local = threading.local()
# Semaphore for LLM calls to limit concurrency
llm_semaphore = threading.Semaphore(4) 

# Default model names
DEFAULT_OLLAMA_MODEL_NAME = "cas/llama-3.2-3b-instruct:latest" # A smaller, faster default for Ollama
DEFAULT_LLAMACPP_REPO_ID = "TheBloke/phi-2-GGUF" # Example, user should verify/change
DEFAULT_LLAMACPP_FILENAME = "phi-2.Q4_K_M.gguf" # Example, ~1.6GB
DEFAULT_LOCAL_OPENAI_MODEL = "cas/llama-3.2-3b-instruct:latest" # Generic name for LM Studio etc.

# Cache directory for LlamaCPP models
LLAMACPP_MODELS_CACHE_DIR = Path.home() / ".cache" / "gguf"
LLAMACPP_MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class LLMProvider:
    """Base class for LLM providers."""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        if not hasattr(self, '_initialized_event'): # Ensure thread-safe init for subclasses
            self._initialized_event = threading.Event()

    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: Optional[int] = 500, # Optional, as not all models/APIs use it identically
                        timeout_seconds: int = 120) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method.")

class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Either pass as api_key or set OPENAI_API_KEY.")
        if OpenAI is None or httpx is None:
            raise ImportError("OpenAI client library or httpx not installed. pip install openai httpx.")
        self._init_client()
    
    def _init_client(self):
        if not hasattr(thread_local, "openai_provider_instance_client"):
            custom_timeouts = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
            thread_local.openai_provider_instance_client = OpenAI(api_key=self.api_key, timeout=custom_timeouts)
        return thread_local.openai_provider_instance_client
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: Optional[int] = 500,
                        timeout_seconds: int = 120) -> Dict[str, Any]:
        client = self._init_client()
        with llm_semaphore:
            try:
                response = client.chat.completions.create(
                    model=self.model_name, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, 
                    timeout=float(timeout_seconds) # Overall request timeout
                )
                return {
                    "id": getattr(response, "id", "openai-unknown-id"),
                    "content": response.choices[0].message.content.strip() if response.choices and response.choices[0].message else "",
                    "finish_reason": response.choices[0].finish_reason if response.choices else "unknown",
                    "model": getattr(response, 'model', self.model_name)
                }
            except (OpenAIAPIConnectionError, OpenAITimeout) as e:
                logging.error(f"OpenAIProvider ({self.model_name}) connection/timeout error: {e}")
                raise
            except Exception as e:
                logging.error(f"OpenAIProvider ({self.model_name}) request failed: {e}")
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
    """Provider for local LLM servers with an OpenAI-compatible API (e.g., LM Studio, older Ollama v1 endpoint)."""
    def __init__(self, model_name: str = DEFAULT_LOCAL_OPENAI_MODEL, 
                 base_url: str = "http://localhost:1234/v1/"): # Common LM Studio URL
        super().__init__(model_name) # API key is not typically used or is a dummy
        self.base_url = base_url
        if OpenAI is None or httpx is None: 
            raise ImportError("OpenAI client library and httpx not installed (required for LocalOpenAIProvider).")
        self._init_client()
    
    def _init_client(self):
        if not hasattr(thread_local, f"local_openai_client_{self.base_url}_{self.model_name}"):
            custom_timeouts = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0) # Longer read for local
            client_instance = OpenAI(
                base_url=self.base_url, 
                api_key="dummy-key", # API key is often not required or ignored
                timeout=custom_timeouts
            )
            setattr(thread_local, f"local_openai_client_{self.base_url}_{self.model_name}", client_instance)
        return getattr(thread_local, f"local_openai_client_{self.base_url}_{self.model_name}")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: Optional[int] = 500,
                        timeout_seconds: int = 180) -> Dict[str, Any]: # Longer default timeout for local
        client = self._init_client()
        with llm_semaphore:
            try:
                response = client.chat.completions.create(
                    model=self.model_name, # Model might be selected in the local server UI
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
                logging.error(f"LocalOpenAIProvider ({self.model_name} at {self.base_url}) request failed: {e}")
                raise

class OllamaProvider(LLMProvider):
    """Ollama LLM provider using the official 'ollama' Python library."""
    _checked_models: set = set() # Class-level set to track checked/pulled models

    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL_NAME, 
                 host: Optional[str] = None): # e.g., http://localhost:11434
        super().__init__(model_name)
        if ollama is None:
            raise ImportError("Official 'ollama' Python library not installed. 'pip install ollama'")
        
        self.host = host
        self._client_kwargs = {}
        if self.host:
            self._client_kwargs['host'] = self.host
        
        # One-time check/pull for the model when the first instance for this model is created
        if self.model_name not in OllamaProvider._checked_models:
            self._ensure_model_available()
            OllamaProvider._checked_models.add(self.model_name)

    def _ensure_model_available(self):
        try:
            models_info = ollama.list(**self._client_kwargs)
            local_model_names = [m.get('name') for m in models_info.get('models', [])]
            if self.model_name not in local_model_names:
                logging.info(f"OllamaProvider: Model '{self.model_name}' not found locally. Attempting to pull...")
                ollama.pull(self.model_name, **self._client_kwargs) 
                logging.info(f"OllamaProvider: Model '{self.model_name}' pulled successfully.")
            else:
                logging.debug(f"OllamaProvider: Model '{self.model_name}' already available locally.")
        except Exception as e:
            logging.warning(f"OllamaProvider: Error checking/pulling model '{self.model_name}': {e}. "
                            f"Ensure the model name is correct and Ollama server is running and accessible at '{self.host or 'default host'}'.")

    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: Optional[int] = 500,
                        timeout_seconds: int = 120) -> Dict[str, Any]: # timeout for ollama.Client
        
        options = {"temperature": temperature}
        if max_tokens is not None and max_tokens > 0:
            options["num_predict"] = max_tokens # Ollama uses num_predict

        # The ollama.Client can take a timeout.
        # We could instantiate it here if timeout_seconds is critical for each call,
        # or rely on a globally configured client if the library supports it well.
        # For now, passing host directly to ollama.chat
        
        # If a more persistent client with specific timeout is needed:
        # client = ollama.Client(host=self.host, timeout=timeout_seconds)
        # response = client.chat(...)
        
        with llm_semaphore:
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=False,
                    options=options,
                    **self._client_kwargs 
                )
                
                content = ""
                if response and 'message' in response and 'content' in response['message']:
                    content = response['message']['content'].strip()
                
                finish_reason = "stop" if response.get('done') else "unknown"

                return {
                    "id": "ollama-" + response.get('created_at', str(time.time())),
                    "content": content,
                    "finish_reason": finish_reason,
                    "model": response.get('model', self.model_name)
                }
            except ollama.ResponseError as e:
                logging.error(f"OllamaProvider: ResponseError for model {self.model_name} (host: {self.host or 'default'}): {e.status_code} - {e.error}")
                raise
            except Exception as e: # Other exceptions like connection errors
                logging.error(f"OllamaProvider: Generic error during chat with model {self.model_name} (host: {self.host or 'default'}): {e}")
                raise

class LlamaCPPProvider(LLMProvider):
    """Provider for llama-cpp-python."""
    _loaded_models: Dict[str, Any] = {} # Class-level cache for Llama instances {model_key: Llama_instance}
    _model_load_lock = threading.Lock() # Lock for loading models

    def __init__(self, 
                 model_name_or_path: str = DEFAULT_LLAMACPP_REPO_ID, # Can be HF repo or local GGUF path
                 model_gguf_filename: Optional[str] = DEFAULT_LLAMACPP_FILENAME, # Required if model_name_or_path is HF repo
                 n_ctx: int = 2048, 
                 n_gpu_layers: int = 0, # -1 for all layers on GPU, 0 for CPU only
                 verbose: bool = False, # llama-cpp-python verbosity
                 chat_format: Optional[str] = "llama-2" # Or other formats like "chatml", "phi-2" etc.
                 ):
        
        # model_name for LLMProvider base can be a composite or just the repo ID
        display_model_name = f"{model_name_or_path}/{model_gguf_filename}" if model_gguf_filename else model_name_or_path
        super().__init__(display_model_name)

        if Llama is None or hf_hub_download is None:
            raise ImportError("llama-cpp-python or huggingface_hub not installed. "
                              "'pip install llama-cpp-python huggingface_hub'")

        self.model_path_key = f"{model_name_or_path}_{model_gguf_filename or ''}" # Unique key for caching
        self.model_name_or_path = model_name_or_path
        self.model_gguf_filename = model_gguf_filename
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llama_verbose = verbose
        self.chat_format = chat_format

        self._init_client() # This will ensure the model is loaded into thread_local

    def _resolve_model_path(self) -> str:
        """Downloads model from HF if needed, returns local path."""
        if Path(self.model_name_or_path).is_file() and self.model_name_or_path.lower().endswith(".gguf"):
            logging.debug(f"LlamaCPPProvider: Using local model path: {self.model_name_or_path}")
            return self.model_name_or_path
        elif self.model_gguf_filename: # Assume HF repo
            logging.info(f"LlamaCPPProvider: Downloading/locating model '{self.model_gguf_filename}' from repo '{self.model_name_or_path}'...")
            try:
                model_path = hf_hub_download(
                    repo_id=self.model_name_or_path,
                    filename=self.model_gguf_filename,
                    cache_dir=LLAMACPP_MODELS_CACHE_DIR,
                    resume_download=True
                )
                logging.info(f"LlamaCPPProvider: Model path resolved to: {model_path}")
                return model_path
            except Exception as e:
                logging.error(f"LlamaCPPProvider: Failed to download model {self.model_gguf_filename} from {self.model_name_or_path}: {e}")
                raise
        else:
            raise ValueError("LlamaCPPProvider: model_name_or_path must be a local .gguf file path, or a HuggingFace repo_id with model_gguf_filename specified.")

    def _init_client(self):
        # Model loading can be slow, so we do it once per model_path_key and store in thread_local
        # We use a class-level lock to ensure only one thread tries to load a given model at a time.
        client_attr_name = f"llama_cpp_client_{self.model_path_key.replace('/', '_').replace('.', '_')}"

        if not hasattr(thread_local, client_attr_name):
            with LlamaCPPProvider._model_load_lock: # Ensure only one thread loads a specific model
                # Double check after acquiring lock
                if not hasattr(thread_local, client_attr_name):
                    actual_model_path = self._resolve_model_path()
                    logging.info(f"LlamaCPPProvider: Initializing Llama model from: {actual_model_path} "
                                 f"(n_ctx={self.n_ctx}, n_gpu_layers={self.n_gpu_layers})")
                    try:
                        llama_instance = Llama(
                            model_path=actual_model_path,
                            n_ctx=self.n_ctx,
                            n_gpu_layers=self.n_gpu_layers,
                            verbose=self.llama_verbose,
                            chat_format=self.chat_format
                        )
                        setattr(thread_local, client_attr_name, llama_instance)
                        logging.info(f"LlamaCPPProvider: Model '{self.model_name}' loaded successfully.")
                    except Exception as e:
                        logging.error(f"LlamaCPPProvider: Error loading Llama model '{self.model_name}': {e}")
                        # To prevent repeated load attempts for this thread for this broken model
                        setattr(thread_local, client_attr_name, None) 
                        raise
        
        client = getattr(thread_local, client_attr_name, None)
        if client is None:
            # This means loading failed previously for this thread after lock release or initial attempt
            raise RuntimeError(f"LlamaCPPProvider: Client for model {self.model_name} could not be initialized.")
        return client

    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: Optional[int] = 500, # llama_cpp uses max_tokens
                        timeout_seconds: int = 180) -> Dict[str, Any]: # Timeout for llama.cpp is less direct, more about processing time
        
        client: Llama = self._init_client() # Ensures model is loaded for this thread

        # llama-cpp-python's create_chat_completion handles timeout internally if supported by underlying calls,
        # but it's mostly for long generations. A hard timeout for the call isn't standard here.
        # We rely on the operation completing or an error.
        
        if max_tokens is None or max_tokens <= 0: # llama-cpp requires positive max_tokens or defaults.
            max_tokens = self.n_ctx // 2 # A reasonable default if not specified.
            
        with llm_semaphore:
            try:
                response = client.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    # stop=["\n"] # Example stop token, can be configured
                )
                
                content = ""
                finish_reason = "unknown"
                response_id = f"llama_cpp-{time.time()}"

                if response and response.get('choices'):
                    choice = response['choices'][0]
                    if choice.get('message') and choice['message'].get('content'):
                        content = choice['message']['content'].strip()
                    finish_reason = choice.get('finish_reason', 'stop') # Typically 'stop' or 'length'
                    if response.get('id'): response_id = response['id']
                
                return {
                    "id": response_id,
                    "content": content,
                    "finish_reason": finish_reason,
                    "model": self.model_name # The display name we set
                }
            except Exception as e:
                logging.error(f"LlamaCPPProvider ({self.model_name}) chat completion failed: {e}")
                raise

# --- Main Factory Function ---
def get_llm_provider(provider_type: str = "ollama", 
                     model_name: Optional[str] = None, # For most providers, this is the model identifier
                     api_key: Optional[str] = None,
                     # Specific args for certain providers
                     ollama_host: Optional[str] = None,
                     local_openai_base_url: Optional[str] = None,
                     llamacpp_repo_id: Optional[str] = None,
                     llamacpp_gguf_filename: Optional[str] = None,
                     llamacpp_n_ctx: int = 2048,
                     llamacpp_n_gpu_layers: int = 0,
                     llamacpp_chat_format: str = "llama-2"
                     ) -> LLMProvider:
    provider_type_lower = provider_type.lower()
    
    # Default model names for each provider type if not specified by user
    default_models = {
        "ollama": DEFAULT_OLLAMA_MODEL_NAME, 
        "groq": "llama3-8b-8192", 
        "openai": "gpt-3.5-turbo", 
        "cohere": "command-r",
        "huggingface": "mistralai/Mistral-7B-Instruct-v0.3", 
        "glhf": "mistralai/Mistral-7B-Instruct-v0.3", 
        "poe": "claude-3-haiku",
        "local_openai": DEFAULT_LOCAL_OPENAI_MODEL,
        "llama_cpp": f"{llamacpp_repo_id or DEFAULT_LLAMACPP_REPO_ID}/{llamacpp_gguf_filename or DEFAULT_LLAMACPP_FILENAME}"
    }
    
    effective_model_name = model_name or default_models.get(provider_type_lower, "default_model_not_in_map")
    if provider_type_lower == "llama_cpp" and not model_name: # If llama_cpp and no specific model, use defaults
        effective_model_name = default_models["llama_cpp"]


    provider_map: Dict[str, type[LLMProvider]] = {
        "ollama": OllamaProvider, 
        "groq": GroqProvider, 
        "openai": OpenAIProvider,
        "cohere": CohereProvider, 
        "huggingface": HuggingFaceProvider,
        "glhf": GLHFProvider, 
        "poe": PoeProvider,
        "local_openai": LocalOpenAIProvider,
        "llama_cpp": LlamaCPPProvider
    }

    if provider_type_lower in provider_map:
        ProviderClass = provider_map[provider_type_lower]
        try:
            if provider_type_lower == "ollama":
                 return ProviderClass(model_name=effective_model_name, host=ollama_host) 
            elif provider_type_lower == "local_openai":
                 return ProviderClass(model_name=effective_model_name, base_url=local_openai_base_url or "http://localhost:1234/v1/")
            elif provider_type_lower == "llama_cpp":
                 # model_name is purely for display; actual model is from repo_id/filename
                 # If user provides a model_name like "TheBloke/...", it's used for repo_id if llamacpp_repo_id is None
                 repo_to_use = llamacpp_repo_id or (model_name if model_name and "/" in model_name else DEFAULT_LLAMACPP_REPO_ID)
                 file_to_use = llamacpp_gguf_filename or (os.path.basename(model_name) if model_name and model_name.lower().endswith(".gguf") and "/" not in model_name else DEFAULT_LLAMACPP_FILENAME)

                 return ProviderClass(
                     model_name_or_path=repo_to_use, 
                     model_gguf_filename=file_to_use,
                     n_ctx=llamacpp_n_ctx,
                     n_gpu_layers=llamacpp_n_gpu_layers,
                     chat_format=llamacpp_chat_format,
                     verbose=logging.getLogger().level == logging.DEBUG # Pass debug status for llama.cpp verbosity
                    )
            else: # For cloud providers primarily needing api_key
                 return ProviderClass(model_name=effective_model_name, api_key=api_key)
        except (ImportError, ValueError) as e: 
            logging.error(f"Error initializing '{provider_type_lower}' provider (model: '{effective_model_name}'): {e}")
            raise 
    else:
        raise ValueError(f"Unknown LLM provider type: {provider_type_lower}")


# --- Core LLM Interaction Functions (send_to_llm, extract_metadata, sort_author_names) ---
# These remain largely the same, operating on the LLMProvider interface.
# (Ensure the refined send_to_llm from previous turn is used here)

def send_to_llm(text: str, filename: str, provider_instance: LLMProvider,
                max_attempts: int = 3, verbose: bool = False,
                temperature_arg: float = 0.5, max_tokens_arg: Optional[int] = 300,
                prompt_choice: int = 0 
                ) -> str:
    base_retry_wait = 5.0 
    prompt_templates = [
        ( 
            f"Extract metadata from the following file extraction snippet. We need (1) the main author name (format: Lastname Firstname), "
            f"(2) the year of publication (4 digits), (3) publication title, and (4) language (2-letter ISO code) "
            f"from the text below. Consider the filename '{os.path.basename(filename)}' for clues. "
            f"Respond ONLY in the following exact format, with no extra text or explanations: \n"
            f"<TITLE>The Full Title</TITLE>\n<YEAR>YYYY</YEAR>\n<AUTHOR>Lastname Firstname</AUTHOR>\n<LANGUAGE>lg</LANGUAGE>\n\n"
        ),
        ( 
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
    
    for attempt in range(1, max_attempts + 1):
        if shutdown_flag.is_set():
            logging.info("Shutdown: Aborting LLM request in send_to_llm.")
            return ""

        current_prompt_template_idx = min(prompt_choice + attempt - 1, len(prompt_templates) - 1)
        prompt_template = prompt_templates[current_prompt_template_idx]
        
        if verbose: logging.debug(f"LLM request for {filename} to {provider_instance.__class__.__name__} ({provider_instance.model_name}), attempt {attempt}, prompt template {current_prompt_template_idx+1}")

        prompt = prompt_template + f"Document text (first 3000 chars):\n{text[:3000]}"
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response_data = provider_instance.chat_completion(
                messages=messages, temperature=temperature_arg,
                max_tokens=max_tokens_arg, 
                timeout_seconds=120 
            )
            output = response_data.get("content", "").strip()
            if verbose: logging.debug(f"LLM raw response for {filename} (attempt {attempt}): '{output}'")
            
            # Primary tag check
            has_title = "<TITLE>" in output
            has_author = "<AUTHOR>" in output
            has_year = "<YEAR>" in output
            # has_language = "<LANGUAGE>" in output # Language is desirable but sometimes omitted by LLM

            if output and has_title and has_author and has_year:
                return output 
            # Check for alternative title tags if primary TITLE is missing
            elif output and ("<PUBLICATION TITLE>" in output or "<PUBLICATIONTITLE>" in output) and \
                 has_author and has_year:
                if verbose: logging.debug(f"LLM for {filename} used alternative title tag. Accepting and standardizing.")
                output = output.replace("<PUBLICATION TITLE>", "<TITLE>").replace("<PUBLICATIONTITLE>", "<TITLE>")
                return output
            else:
                logging.warning(f"LLM response for {filename} (attempt {attempt}) from {provider_instance.model_name} "
                                f"missing key tags (TITLE, AUTHOR, YEAR) or empty: '{output[:100]}...'")
                if attempt == max_attempts: 
                    logging.error(f"Max attempts reached for {filename}, returning last (possibly malformed) LLM output.")
                    return output 
        
        except (OpenAIAPIConnectionError, OpenAITimeout, ollama.ResponseError if ollama else Exception) as e_net: 
            # Catch specific network/timeout errors from different libraries if possible
            log_msg = f"LLM network/timeout error for {filename} with {provider_instance.__class__.__name__} (attempt {attempt}): {e_net}."
            # For ollama.ResponseError, e_net.status_code might be informative
            if ollama and isinstance(e_net, ollama.ResponseError):
                log_msg += f" Status: {e_net.status_code}, Error: {e_net.error}"

            logging.warning(f"{log_msg} Retrying in {base_retry_wait * attempt:.1f}s.")
            if attempt < max_attempts:
                time.sleep(base_retry_wait * attempt) 
            else:
                logging.error(f"Max LLM retries for {filename} due to network/timeout errors.")
                return ""
        except Exception as e: 
            wait_time = base_retry_wait * (1.5 ** (attempt - 1)) 
            logging.warning(f"LLM error for {filename} with {provider_instance.__class__.__name__} "
                            f"(attempt {attempt}): {e}. Retrying in {wait_time:.1f}s.", exc_info=verbose)
            
            if attempt < max_attempts:
                time.sleep(wait_time)
            else:
                logging.error(f"Max LLM retries for {filename} with {provider_instance.__class__.__name__} due to errors.")
                return "" 
        
    return ""


def extract_metadata(text: str, filename: str, 
                     llm_provider_arg: Union[str, LLMProvider, None], 
                     model_name_arg: Optional[str] = None,
                     api_key_arg: Optional[str] = None,
                     verbose: bool = False,
                     # Pass provider-specific args through kwargs to get_llm_provider
                     **provider_specific_kwargs 
                     ) -> str:
    provider_instance: Optional[LLMProvider] = None
    if isinstance(llm_provider_arg, LLMProvider):
        provider_instance = llm_provider_arg
    else: 
        provider_name_str = llm_provider_arg if isinstance(llm_provider_arg, str) else "ollama"
        try:
            # Pass all provider_specific_kwargs to get_llm_provider
            provider_instance = get_llm_provider(
                provider_name_str, 
                model_name_arg, 
                api_key_arg,
                **provider_specific_kwargs # Handles ollama_host, llamacpp_*, etc.
            )
        except Exception as e:
            logging.error(f"Failed to get LLMProvider for '{provider_name_str}' in extract_metadata: {e}.")
            return ""

    if not provider_instance:
        logging.error("LLM provider instance could not be resolved in extract_metadata.")
        return ""

    return send_to_llm(text=text, filename=filename, provider_instance=provider_instance, verbose=verbose, 
                       temperature_arg=provider_specific_kwargs.get('temperature', 0.5), # Get from kwargs or use default
                       max_tokens_arg=provider_specific_kwargs.get('max_tokens', 300))


def sort_author_names(author_names_input: str, 
                      provider_arg: Union[str, LLMProvider, None], 
                      temperature: float = 0.2, 
                      max_tokens: Optional[int] = 60,   
                      max_attempts: int = 2, 
                      verbose: bool = False,
                      filename_for_logging: str = "UnknownFile", 
                      model_name_arg: Optional[str] = None,
                      api_key_arg: Optional[str] = None,
                      **provider_specific_kwargs # For get_llm_provider if needed
                      ) -> str:
    # ... (This function's internal logic remains largely the same as your provided version)
    # Ensure it correctly calls get_llm_provider if provider_arg is a string,
    # passing along model_name_arg, api_key_arg, and any relevant **provider_specific_kwargs.
    if not author_names_input or not isinstance(author_names_input, str):
        return "UnknownAuthor"
    
    cleaned_input_name = re.sub(r"^\s*<AUTHOR>\s*|\s*</AUTHOR>\s*$", "", author_names_input, flags=re.IGNORECASE).strip()
    cleaned_input_name = re.sub(r"^(Main Author: Lastname Firstname:|Main Author:)\s*", "", cleaned_input_name, flags=re.IGNORECASE).strip()

    if not cleaned_input_name or cleaned_input_name.lower() in ["unknown", "unknownauthor", "n a", ""]:
        return "UnknownAuthor"

    if ',' in cleaned_input_name:
        parts = [p.strip() for p in cleaned_input_name.split(',', 1)]
        if len(parts) == 2 and parts[0] and parts[1] and len(parts[1].split()) <= 3:
            reformatted = f"{parts[0]} {parts[1]}"
            reformatted = re.sub(r'\s+', ' ', reformatted).strip()
            if verbose: logging.debug(f"Pre-LLM comma sort for '{cleaned_input_name}' -> '{reformatted}' (file: {filename_for_logging})")
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
                provider_name_str, 
                model_name_arg, 
                api_key_arg, 
                **provider_specific_kwargs # Pass along any relevant kwargs for this specific call context
            )
        except Exception as e:
            logging.error(f"Failed to get LLM provider for author sorting ('{provider_name_str}'): {e}. Returning '{author_to_process}'.")
            return author_to_process 

    if not llm_instance:
        logging.error(f"LLM instance is None in sort_author_names for '{author_to_process}'. Returning as-is.")
        return author_to_process

    base_retry_wait = 1.0
    for attempt in range(1, max_attempts + 1):
        if shutdown_flag.is_set(): return author_to_process 

        prompt = (
            f"Reformat the author name '{author_to_process}' into the standard 'Lastname Firstname' format. "
            f"If it's a single name (e.g., 'Plato'), return it as is. "
            f"If it's an organization, return it as is. "
            f"Handle particles like 'van', 'de la', 'von' correctly (e.g., 'Vincent van Gogh' should become 'van Gogh Vincent'). "
            f"Prioritize 'Lastname Firstname'. "
            f"Respond ONLY with the formatted name inside <AUTHOR></AUTHOR> tags. Examples:\n"
            f"Input: John Doe -> Output: <AUTHOR>Doe John</AUTHOR>\n"
            f"Input: Plato -> Output: <AUTHOR>Plato</AUTHOR>\n"
            f"Input: J. R. R. Tolkien -> Output: <AUTHOR>Tolkien JRR</AUTHOR>\n"
            f"Input Author: {author_to_process}"
        )
        messages = [{"role": "user", "content": prompt}]
        
        with llm_semaphore:
            try:
                response = llm_instance.chat_completion(
                    messages=messages, temperature=temperature, max_tokens=max_tokens, timeout_seconds=30
                )
                llm_output = response.get("content", "").strip()

                if llm_output:
                    name_match = re.search(r'<AUTHOR>(.*?)</AUTHOR>', llm_output, re.DOTALL | re.IGNORECASE)
                    if name_match:
                        ordered_name = name_match.group(1).strip()
                        ordered_name = re.sub(r'\s+', ' ', ordered_name).strip()
                        if ordered_name and ordered_name.lower() not in ["lastname firstname", "unknownauthor", "unknown author"]:
                            if verbose: logging.debug(f"LLM sorted '{author_to_process}' to '{ordered_name}' using {llm_instance.model_name} for {filename_for_logging}")
                            return ordered_name
                    else: 
                        plain_name = re.sub(r'<[^>]+>', '', llm_output).strip()
                        if plain_name and len(plain_name.split()) >= 1 and len(plain_name) < 70 and plain_name.lower() not in ["lastname firstname", "unknownauthor", "unknown author"]:
                             if verbose: logging.debug(f"LLM sorted (no tags) '{author_to_process}' to '{plain_name}' for {filename_for_logging}")
                             return plain_name
                    logging.warning(f"LLM response for author sort of '{author_to_process}' (file: {filename_for_logging}) was invalid: '{llm_output[:100]}'")
            except Exception as e:
                logging.warning(f"LLM error sorting author '{author_to_process}' (attempt {attempt}, model {llm_instance.model_name}, file: {filename_for_logging}): {e}")
        
        if attempt < max_attempts: time.sleep(base_retry_wait * (1.5 ** (attempt - 1)))
    
    logging.warning(f"Could not reliably sort author name '{author_names_input}' (processed as '{author_to_process}') for {filename_for_logging} via LLM. Returning best cleaned input.")
    return author_to_process