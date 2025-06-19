import json
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import Field, SecretStr
from requests import Response



class SambaNovaCloud(LLM):
    """
    SambaNova Cloud large language models.

    Setup:
        To use, you should have the environment variables:
        ``SAMBANOVA_URL`` set with SambaNova Cloud URL.
        defaults to http://cloud.sambanova.ai/
        ``SAMBANOVA_API_KEY`` set with your SambaNova Cloud API Key.
        Example:
        .. code-block:: python
            from langchain_community.llms.sambanova  import SambaNovaCloud
            SambaNovaCloud(
                sambanova_api_key="your-SambaNovaCloud-API-key,
                model = model name,
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                top_k = model top k
            )
    Key init args — completion params:
        model: str
            The name of the model to use, e.g., Meta-Llama-3-70B-Instruct-4096
            (set for bundle endpoints).
        streaming: bool
            Whether to use streaming handler when using non streaming methods
        max_tokens: int
            max tokens to generate
        temperature: float
            model temperature
        top_p: float
            model top p
        top_k: int
            model top k

    Key init args — client params:
        sambanova_url: str
            SambaNovaCloud Url defaults to http://cloud.sambanova.ai/
        sambanova_api_key: str
            SambaNovaCloud api key
    Instantiate:
        .. code-block:: python
            from langchain_community.llms.sambanova  import SambaNovaCloud
            SambaNovaCloud(
                sambanova_api_key="your-SambaNovaCloud-API-key,
                model = model name,
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                top_k = model top k
            )
    Invoke:
        .. code-block:: python
            prompt = "tell me a joke"
            response = llm.invoke(prompt)
    Stream:
        .. code-block:: python
        for chunk in llm.stream(prompt):
            print(chunk, end="", flush=True)
    Async:
        .. code-block:: python
        response = llm.ainvoke(prompt)
        await response
    """

    sambanova_url: str = "https://api.sambanova.ai/v1"
    """SambaNova Cloud Url"""

    sambanova_api_key: SecretStr ="7bd1a021-e60a-476f-abc7-837f09d4bf1f" 
    """SambaNova Cloud api key"""

    model: str = Field(default='Meta-Llama-3.1-8B-Instruct')
    """The name of the model"""

    streaming: bool = Field(default=False)
    """Whether to use streaming handler when using non streaming methods"""

    max_tokens: int = Field(default=1024)
    """max tokens to generate"""

    temperature: float = Field(default=0.7)
    """model temperature"""

    top_p: Optional[float] = Field(default=None)
    """model top p"""

    top_k: Optional[int] = Field(default=None)
    """model top k"""

    stream_options: dict = Field(default={'include_usage': True})
    """stream options, include usage to get generation metrics"""

    class Config:
        populate_by_name = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {'sambanova_api_key': 'sambanova_api_key'}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            'model': self.model,
            'streaming': self.streaming,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'stream_options': self.stream_options,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return 'sambanovacloud-llm'

    def __init__(self, **kwargs: Any) -> None:
        """init and validate environment variables"""
        kwargs['sambanova_url'] = get_from_dict_or_env(
            kwargs,
            'sambanova_url',
            'SAMBANOVA_URL',
            default='https://api.sambanova.ai/v1/chat/completions',
        )
        kwargs['sambanova_api_key'] = convert_to_secret_str(
            get_from_dict_or_env(kwargs, 'sambanova_api_key', 'SAMBANOVA_API_KEY')
        )
        super().__init__(**kwargs)

    def _handle_request(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        streaming: Optional[bool] = False,
    ) -> Response:
        """
        Performs a post request to the LLM API.

        Args:
            prompt: The prompt to pass into the model.
            stop: list of stop tokens

        Returns:
            A request Response object
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        messages_dict = [{'role': 'user', 'content': prompt[0]}]
        data = {
            'messages': messages_dict,
            'stream': streaming,
            'max_tokens': self.max_tokens,
            'stop': stop,
            'model': self.model,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
        }
        data = {key: value for key, value in data.items() if value is not None}
        headers = {
            'Authorization': f'Bearer ' f'{self.sambanova_api_key.get_secret_value()}',
            'Content-Type': 'application/json',
        }

        http_session = requests.Session()
        if streaming:
            response = http_session.post(self.sambanova_url, headers=headers, json=data, stream=True)
        else:
            response = http_session.post(self.sambanova_url, headers=headers, json=data, stream=False)

        if response.status_code != 200:
            raise RuntimeError(
                f'Sambanova / complete call failed with status code ' f'{response.status_code}.' f'{response.text}.'
            )
        return response

    def _process_response(self, response: Response) -> str:
        """
        Process a non streaming response from the api

        Args:
            response: A request Response object

        Returns
            completion: a string with model generation
        """

        # Extract json payload form response
        try:
            response_dict = response.json()
        except Exception as e:
            raise RuntimeError(
                f"Sambanova /complete call failed couldn't get JSON response {e}" f'response: {response.text}'
            )

        completion = response_dict['choices'][0]['message']['content']

        return completion

    def _process_stream_response(self, response: Response) -> Iterator[GenerationChunk]:
        """
        Process a streaming response from the api

        Args:
            response: An iterable request Response object

        Yields:
            GenerationChunk: a GenerationChunk with model partial generation
        """

        try:
            import sseclient
        except ImportError:
            raise ImportError('could not import sseclient library' 'Please install it with `pip install sseclient-py`.')

        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.event == 'error_event':
                raise RuntimeError(
                    f'Sambanova /complete call failed with status code ' f'{response.status_code}.' f'{event.data}.'
                )
            try:
                # check if the response is not a final event ("[DONE]")
                if event.data != '[DONE]':
                    if isinstance(event.data, str):
                        data = json.loads(event.data)
                    else:
                        raise RuntimeError(
                            f'Sambanova /complete call failed with status code '
                            f'{response.status_code}.'
                            f'{event.data}.'
                        )
                    if data.get('error'):
                        raise RuntimeError(
                            f'Sambanova /complete call failed with status code '
                            f'{response.status_code}.'
                            f'{event.data}.'
                        )
                    if len(data['choices']) > 0:
                        content = data['choices'][0]['delta']['content']
                    else:
                        content = ''
                    generated_chunk = GenerationChunk(text=content)
                    yield generated_chunk

            except Exception as e:
                raise RuntimeError(f'Error getting content chunk raw streamed response: {e}' f'data: {event.data}')

    def _call(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to SambaNovaCloud complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        if self.streaming:
            completion = ''
            for chunk in self._stream(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs):
                completion += chunk.text

            return completion

        response = self._handle_request(prompt, stop, streaming=False)
        completion = self._process_response(response)
        return completion

    def _stream(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call out to SambaNovaCloud complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        response = self._handle_request(prompt, stop, streaming=True)
        for chunk in self._process_stream_response(response):
            if run_manager:
                run_manager.on_llm_new_token(chunk.text)
            yield chunk