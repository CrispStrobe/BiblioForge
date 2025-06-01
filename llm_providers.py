# llm_providers.py

import os
import logging
import threading
import time
from typing import Optional, List, Dict, Any, Union
import re # For sort_author_names

# Attempt to import necessary HTTP client libraries
try:
    import requests
except ImportError:
    requests = None 
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None 
try:
    from huggingface_hub import InferenceClient 
except ImportError:
    InferenceClient = None
try:
    import cohere 
except ImportError:
    cohere = None
try:
    from groq import Groq
except ImportError:
    Groq = None

# Import shutdown_flag from utils.
try:
    # This assumes utils.py is in a directory that's on the PYTHONPATH
    # when llm_providers.py is imported by another root module like document_processor.py
    from utils import shutdown_flag
except ImportError:
    logging.critical("llm_providers.py: CRITICAL - Could not import shutdown_flag from utils. Graceful shutdown might not work.")
    # Create a dummy event so the script doesn't crash immediately if utils is missing
    # This is not a solution for production but helps during refactoring.
    shutdown_flag = threading.Event()


# Thread-local storage for LLM clients
thread_local = threading.local()
# Semaphore for LLM calls
llm_semaphore = threading.Semaphore(4) # Adjusted, can be configured

# Default model name for Ollama
DEFAULT_OLLAMA_MODEL_NAME = "cas/llama-3.2-3b-instruct:latest" 
# Main model name used by functions in this module if not overridden
# This is less relevant now as model is usually tied to the provider instance.
MODEL_NAME_FALLBACK = DEFAULT_OLLAMA_MODEL_NAME


class LLMProvider:
    """Base class for LLM providers."""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: int = 500,
                        timeout_seconds: int = 120) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method.")

class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Either pass as api_key or set OPENAI_API_KEY.")
        if OpenAI is None:
            raise ImportError("OpenAI client library not installed. pip install openai.")
        self._init_client()
    
    def _init_client(self):
        if not hasattr(thread_local, "openai_provider_instance_client"): # More specific name
            thread_local.openai_provider_instance_client = OpenAI(api_key=self.api_key)
        return thread_local.openai_provider_instance_client
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: int = 500,
                        timeout_seconds: int = 120) -> Dict[str, Any]:
        client = self._init_client()
        with llm_semaphore:
            try:
                response = client.chat.completions.create(
                    model=self.model_name, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, timeout=timeout_seconds
                )
                return {
                    "id": getattr(response, "id", "openai-unknown-id"),
                    "content": response.choices[0].message.content.strip() if response.choices and response.choices[0].message else "",
                    "finish_reason": response.choices[0].finish_reason if response.choices else "unknown",
                    "model": getattr(response, 'model', self.model_name) # Use model from response if available
                }
            except Exception as e:
                logging.error(f"OpenAIProvider request for {self.model_name} failed: {e}")
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

class OllamaProvider(LLMProvider):
    """Ollama LLM provider."""
    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL_NAME, 
                 base_url: Optional[str] = None): # base_url is optional, defaults in constructor
        super().__init__(model_name) # api_key is not used by OllamaProvider constructor
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/")
        if OpenAI is None: raise ImportError("OpenAI client not installed (for OllamaProvider).")
        self._init_client()
    
    def _init_client(self):
        if not hasattr(thread_local, "ollama_provider_instance_client"): # Specific name
            thread_local.ollama_provider_instance_client = OpenAI(base_url=self.base_url, api_key="ollama")
        return thread_local.ollama_provider_instance_client
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: int = 500,
                        timeout_seconds: int = 120) -> Dict[str, Any]:
        client = self._init_client()
        with llm_semaphore:
            try:
                response = client.chat.completions.create(
                    model=self.model_name, temperature=temperature, max_tokens=max_tokens,
                    messages=messages, timeout=timeout_seconds
                )
                return {
                    "id": getattr(response, "id", "ollama-unknown-id"),
                    "content": response.choices[0].message.content.strip() if response.choices and response.choices[0].message else "",
                    "finish_reason": response.choices[0].finish_reason if response.choices else "unknown",
                    "model": getattr(response, 'model', self.model_name)
                }
            except Exception as e:
                logging.error(f"OllamaProvider request for {self.model_name} failed: {e}")
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


def get_llm_provider(provider_type: str = "ollama", 
                     model_name: Optional[str] = None,
                     api_key: Optional[str] = None) -> LLMProvider:
    provider_type_lower = provider_type.lower()
    
    default_models = {
        "ollama": DEFAULT_OLLAMA_MODEL_NAME, "groq": "llama3-8b-8192", # Changed to 8b for wider access
        "openai": "gpt-3.5-turbo", "cohere": "command-r",
        "huggingface": "mistralai/Mistral-7B-Instruct-v0.3", 
        "glhf": "mistralai/Mistral-7B-Instruct-v0.3", "poe": "claude-3-haiku" 
    }
    
    effective_model_name = model_name or default_models.get(provider_type_lower, DEFAULT_OLLAMA_MODEL_NAME)
    
    provider_map: Dict[str, type[LLMProvider]] = {
        "ollama": OllamaProvider, "groq": GroqProvider, "openai": OpenAIProvider,
        "cohere": CohereProvider, "huggingface": HuggingFaceProvider,
        "glhf": GLHFProvider, "poe": PoeProvider
    }

    if provider_type_lower in provider_map:
        ProviderClass = provider_map[provider_type_lower]
        try:
            # OllamaProvider's __init__ doesn't take api_key, others might.
            if provider_type_lower == "ollama":
                 return ProviderClass(model_name=effective_model_name) # No api_key for OllamaProvider
            else:
                 return ProviderClass(model_name=effective_model_name, api_key=api_key)
        except (ImportError, ValueError) as e: 
            logging.error(f"Error initializing {provider_type_lower} provider with model {effective_model_name}: {e}")
            raise 
    else:
        raise ValueError(f"Unknown LLM provider type: {provider_type_lower}")


def send_to_llm(text: str, filename: str, provider_instance: LLMProvider,
                max_attempts: int = 3, verbose: bool = False,
                temperature_arg: float = 0.5, max_tokens_arg: int = 300, # Increased max_tokens slightly
                prompt_choice: int = 0 
                ) -> str:
    base_retry_wait = 2.0
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
                max_tokens=max_tokens_arg, timeout_seconds=120 
            )
            output = response_data.get("content", "").strip()
            if verbose: logging.debug(f"LLM raw response for {filename} (attempt {attempt}): '{output}'")
            
            # Check if essential tags are present. Parsing happens later.
            if output and "<TITLE>" in output and "<AUTHOR>" in output and "<YEAR>" in output:
                return output 
            else:
                logging.warning(f"LLM response for {filename} (attempt {attempt}) from {provider_instance.model_name} missing key tags or empty: '{output[:100]}...'")
                if attempt == max_attempts: 
                    logging.error(f"Max attempts reached for {filename}, returning last (possibly malformed) LLM output.")
                    return output # Return last attempt even if malformed
        
        except Exception as e:
            err_msg = str(e).lower()
            is_timeout_or_ratelimit = any(keyword in err_msg for keyword in ["rate_limit", "timeout", "timed out", "limit", "too many requests"])
            wait_time = base_retry_wait * (2 ** attempt if is_timeout_or_ratelimit else 1.5 ** attempt)
            log_level = logging.INFO if is_timeout_or_ratelimit else logging.ERROR
            logging.log(log_level, f"LLM error for {filename} with {provider_instance.__class__.__name__} (attempt {attempt}): {e}. Retrying in {wait_time:.1f}s.")
            
            if attempt < max_attempts: time.sleep(wait_time)
            else:
                logging.error(f"Max LLM retries for {filename} with {provider_instance.__class__.__name__} due to errors.")
                return "" 
        
    return "" # Fallback if all attempts fail (e.g. consistent format issues not caught by tag check)

def extract_metadata(text: str, filename: str, 
                     llm_provider_arg: Union[str, LLMProvider, None], 
                     model_name_arg: Optional[str] = None,
                     api_key_arg: Optional[str] = None,
                     verbose: bool = False) -> str:
    """
    High-level wrapper to get raw metadata string from an LLM provider.
    """
    provider_instance: Optional[LLMProvider] = None
    if isinstance(llm_provider_arg, LLMProvider):
        provider_instance = llm_provider_arg
    else: 
        provider_name_str = llm_provider_arg if isinstance(llm_provider_arg, str) else "ollama"
        try:
            provider_instance = get_llm_provider(provider_name_str, model_name_arg, api_key_arg)
        except Exception as e:
            logging.error(f"Failed to get LLMProvider for '{provider_name_str}' in extract_metadata: {e}.")
            return ""

    if not provider_instance:
        logging.error("LLM provider instance could not be resolved in extract_metadata.")
        return ""

    return send_to_llm(text=text, filename=filename, provider_instance=provider_instance, verbose=verbose)


def sort_author_names(author_names_input: str, 
                      provider_arg: Union[str, LLMProvider, None], 
                      temperature: float = 0.2, # Lower temp for more deterministic formatting
                      max_tokens: int = 60,   # Usually short response needed
                      max_attempts: int = 2, 
                      verbose: bool = False,
                      filename_for_logging: str = "UnknownFile", # For better context in logs
                      model_name_arg: Optional[str] = None,
                      api_key_arg: Optional[str] = None
                      ) -> str:
    """
    Formats author names to 'Lastname Firstname' using an LLM.
    Returns the best effort formatted name, or a cleaned version of input if LLM fails.
    """
    if not author_names_input or not isinstance(author_names_input, str):
        return "UnknownAuthor"
    
    # Clean initial string from any surrounding XML tags or known prefixes
    cleaned_input_name = re.sub(r"^\s*<AUTHOR>\s*|\s*</AUTHOR>\s*$", "", author_names_input, flags=re.IGNORECASE).strip()
    cleaned_input_name = re.sub(r"^(Main Author: Lastname Firstname:|Main Author:)\s*", "", cleaned_input_name, flags=re.IGNORECASE).strip()

    if not cleaned_input_name or cleaned_input_name.lower() in ["unknown", "unknownauthor", "n a", ""]:
        return "UnknownAuthor"

    # Heuristic: If it looks like "Lastname, Firstname" or "Lastname, F.", reformat directly
    if ',' in cleaned_input_name:
        parts = [p.strip() for p in cleaned_input_name.split(',', 1)] # Split only on first comma
        if len(parts) == 2 and parts[0] and parts[1]:
            # Check if the part after comma looks like a first name/initials (not another last name)
            # This is a simple check; more complex name structures might need LLM.
            if len(parts[1].split()) <= 3: # Allow for multiple middle names/initials
                reformatted = f"{parts[0]} {parts[1]}" # Assumes "Lastname" "Firstname Middle"
                reformatted = re.sub(r'\s+', ' ', reformatted).strip()
                if verbose: logging.debug(f"Pre-LLM comma sort for '{cleaned_input_name}' -> '{reformatted}' (file: {filename_for_logging})")
                return reformatted
            # Else, it might be "Org, Unit" or "Lastname1, Lastname2" - let LLM try

    # Reduce multiple authors (if clearly delimited by ';') to the first one for the LLM prompt
    author_to_process = cleaned_input_name.split(';')[0].strip()
    if not author_to_process: return "UnknownAuthor" # If splitting resulted in empty

    # Resolve LLM provider instance
    llm_instance: Optional[LLMProvider] = None
    if isinstance(provider_arg, LLMProvider):
        llm_instance = provider_arg
    else: 
        provider_name_str = provider_arg if isinstance(provider_arg, str) else "ollama"
        try:
            llm_instance = get_llm_provider(provider_name_str, model_name_arg, api_key_arg)
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
                    else: # LLM didn't use tags, try to use the whole response if it looks like a name
                        plain_name = re.sub(r'<[^>]+>', '', llm_output).strip()
                        if plain_name and len(plain_name.split()) >= 1 and len(plain_name) < 70 and plain_name.lower() not in ["lastname firstname", "unknownauthor", "unknown author"]:
                             if verbose: logging.debug(f"LLM sorted (no tags) '{author_to_process}' to '{plain_name}' for {filename_for_logging}")
                             return plain_name
                    logging.warning(f"LLM response for author sort of '{author_to_process}' (file: {filename_for_logging}) was invalid: '{llm_output[:100]}'")
            except Exception as e:
                logging.warning(f"LLM error sorting author '{author_to_process}' (attempt {attempt}, model {llm_instance.model_name}, file: {filename_for_logging}): {e}")
        
        if attempt < max_attempts: time.sleep(base_retry_wait * (1.5 ** (attempt - 1)))
    
    logging.warning(f"Could not reliably sort author name '{author_names_input}' (processed as '{author_to_process}') for {filename_for_logging} via LLM. Returning best cleaned input.")
    return author_to_process # Fallback to the pre-LLM cleaned/processed name