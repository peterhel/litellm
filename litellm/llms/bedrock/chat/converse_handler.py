import json
from typing import Any, Optional, Union, cast

import httpx

import litellm
from litellm.anthropic_beta_headers_manager import (
    update_headers_with_filtered_beta,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

from ..base_aws_llm import BaseAWSLLM, Credentials
from ..common_utils import BedrockError, _get_all_bedrock_regions
from .invoke_handler import AWSEventStreamDecoder, MockResponseIterator, make_call


def make_sync_call(
    client: Optional[HTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj: LiteLLMLoggingObject,
    json_mode: Optional[bool] = False,
    fake_stream: bool = False,
    stream_chunk_size: int = 1024,
):
    if client is None:
        client = _get_httpx_client()

    response = client.post(
        api_base,
        headers=headers,
        data=data,
        stream=not fake_stream,
        logging_obj=logging_obj,
    )

    if response.status_code != 200:
        raise BedrockError(
            status_code=response.status_code, message=str(response.read())
        )

    if fake_stream:
        model_response = litellm.AmazonConverseConfig()._transform_response(
            model=model,
            response=response,
            model_response=litellm.ModelResponse(),
            stream=True,
            logging_obj=logging_obj,
            optional_params={},
            api_key="",
            data=data,
            messages=messages,
            encoding=litellm.encoding,
        )
        completion_stream: Any = MockResponseIterator(
            model_response=model_response, json_mode=json_mode
        )
    else:
        decoder = AWSEventStreamDecoder(model=model, json_mode=json_mode)
        completion_stream = decoder.iter_bytes(
            response.iter_bytes(chunk_size=stream_chunk_size)
        )

    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


class BedrockConverseLLM(BaseAWSLLM):
    def __init__(self) -> None:
        super().__init__()

    def _scrub_converse_request(self, request_data: dict):
        """
        Final pass to ensure no 'text' fields in the Converse API payload are empty.
        Bedrock will reject the entire request if any block has blank text.
        """
        def clean_block(block):
            if isinstance(block, dict) and "text" in block:
                # If text is empty or just whitespace, use a single space
                if not block["text"] or not str(block["text"]).strip():
                    block["text"] = " "
            return block

        if "messages" in request_data:
            for msg in request_data["messages"]:
                if "content" in msg:
                    if isinstance(msg["content"], list):
                        msg["content"] = [clean_block(b) for b in msg["content"]]
                    elif isinstance(msg["content"], str) and not msg["content"].strip():
                        msg["content"] = " "

        if "system" in request_data:
            if isinstance(request_data["system"], list):
                request_data["system"] = [clean_block(b) for b in request_data["system"]]

        return request_data

    def _clean_messages(self, messages: list):
        if not messages or not isinstance(messages, list):
            return messages

        for msg in messages:
            content = msg.get("content")
            if content:
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            if not block["text"] or not str(block["text"]).strip():
                                block["text"] = " "
                elif isinstance(content, str) and not content.strip():
                    msg["content"] = " "
        return messages

    async def async_streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        stream,
        optional_params: dict,
        litellm_params: dict,
        credentials: Credentials,
        logger_fn=None,
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
        fake_stream: bool = False,
        json_mode: Optional[bool] = False,
        api_key: Optional[str] = None,
        stream_chunk_size: int = 1024,
    ) -> CustomStreamWrapper:
        messages = self._clean_messages(messages)

        request_data = await litellm.AmazonConverseConfig()._async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        request_data = self._scrub_converse_request(request_data)
        data = json.dumps(request_data)

        prepped = self.get_request_headers(
            credentials=credentials,
            aws_region_name=litellm_params.get("aws_region_name") or "us-west-2",
            extra_headers=headers,
            endpoint_url=api_base,
            data=data,
            headers=headers,
            api_key=api_key,
        )

        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": dict(prepped.headers),
            },
        )

        completion_stream = await make_call(
            client=client,
            api_base=api_base,
            headers=dict(prepped.headers),
            data=data,
            model=model,
            messages=messages,
            logging_obj=logging_obj,
            fake_stream=fake_stream,
            json_mode=json_mode,
            stream_chunk_size=stream_chunk_size,
        )
        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="bedrock",
            logging_obj=logging_obj,
        )

        return streaming_response

    async def async_completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj: LiteLLMLoggingObject,
        stream,
        optional_params: dict,
        litellm_params: dict,
        credentials: Credentials,
        logger_fn=None,
        headers: dict = {},
        client: Optional[AsyncHTTPHandler] = None,
        api_key: Optional[str] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        messages = self._clean_messages(messages)

        request_data = await litellm.AmazonConverseConfig()._async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        request_data = self._scrub_converse_request(request_data)
        data = json.dumps(request_data)

        prepped = self.get_request_headers(
            credentials=credentials,
            aws_region_name=litellm_params.get("aws_region_name") or "us-west-2",
            extra_headers=headers,
            endpoint_url=api_base,
            data=data,
            headers=headers,
            api_key=api_key,
        )

        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": prepped.headers,
            },
        )

        headers = dict(prepped.headers)
        if client is None or not isinstance(client, AsyncHTTPHandler):
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = get_async_httpx_client(
                params=_params, llm_provider=litellm.LlmProviders.BEDROCK
            )

        try:
            response = await client.post(
                url=api_base,
                headers=headers,
                data=data,
                logging_obj=logging_obj,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        return litellm.AmazonConverseConfig()._transform_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream if isinstance(stream, bool) else False,
            logging_obj=logging_obj,
            api_key="",
            data=data,
            messages=messages,
            optional_params=optional_params,
            encoding=encoding,
        )

    def completion(
        self,
        model: str,
        messages: list,
        api_base: Optional[str],
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        encoding,
        logging_obj: LiteLLMLoggingObject,
        optional_params: dict,
        acompletion: bool,
        timeout: Optional[Union[float, httpx.Timeout]],
        litellm_params: dict,
        logger_fn=None,
        extra_headers: Optional[dict] = None,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
        api_key: Optional[str] = None,
    ):
        messages = self._clean_messages(messages)

        stream = optional_params.pop("stream", None)
        stream_chunk_size = optional_params.pop("stream_chunk_size", 1024)
        unencoded_model_id = optional_params.pop("model_id", None)
        fake_stream = optional_params.pop("fake_stream", False)
        json_mode = optional_params.get("json_mode", False)

        if unencoded_model_id is not None:
            modelId = self.encode_model_id(model_id=unencoded_model_id)
        else:
            _model_for_id = model
            _stripped = _model_for_id
            for rp in ["bedrock/converse/", "bedrock/", "converse/"]:
                if _stripped.startswith(rp):
                    _stripped = _stripped[len(rp) :]
                    break

            _region_from_model: Optional[str] = None
            _potential_region = _stripped.split("/", 1)[0]
            if _potential_region in _get_all_bedrock_regions() and "/" in _stripped:
                _region_from_model = _potential_region
                _stripped = _stripped.split("/", 1)[1]
                _model_for_id = _stripped

            modelId = self.encode_model_id(model_id=_model_for_id)
            if (
                _region_from_model is not None
                and "aws_region_name" not in optional_params
            ):
                optional_params["aws_region_name"] = _region_from_model

        aws_region_name = self._get_aws_region_name(
            optional_params=optional_params, model=model, model_id=unencoded_model_id
        )

        credentials: Credentials = self.get_credentials(
            aws_access_key_id=optional_params.pop("aws_access_key_id", None),
            aws_secret_access_key=optional_params.pop("aws_secret_access_key", None),
            aws_session_token=optional_params.pop("aws_session_token", None),
            aws_region_name=aws_region_name,
            aws_session_name=optional_params.pop("aws_session_name", None),
            aws_profile_name=optional_params.pop("aws_profile_name", None),
            aws_role_name=optional_params.pop("aws_role_name", None),
            aws_web_identity_token=optional_params.pop("aws_web_identity_token", None),
            aws_sts_endpoint=optional_params.pop("aws_sts_endpoint", None),
            aws_external_id=optional_params.pop("aws_external_id", None),
        )
        litellm_params["aws_region_name"] = aws_region_name

        endpoint_url, proxy_endpoint_url = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=optional_params.pop(
                "aws_bedrock_runtime_endpoint", None
            ),
            aws_region_name=aws_region_name,
        )

        if stream is True and not fake_stream:
            proxy_endpoint_url = f"{proxy_endpoint_url}/model/{modelId}/converse-stream"
        else:
            proxy_endpoint_url = f"{proxy_endpoint_url}/model/{modelId}/converse"

        headers = update_headers_with_filtered_beta(
            headers={"Content-Type": "application/json", **(extra_headers or {})},
            provider="bedrock_converse",
        )

        if acompletion:
            if stream is True:
                return self.async_streaming(
                    model=model,
                    messages=messages,
                    api_base=proxy_endpoint_url,
                    model_response=model_response,
                    timeout=timeout,
                    encoding=encoding,
                    logging_obj=logging_obj,
                    stream=stream,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    credentials=credentials,
                    logger_fn=logger_fn,
                    headers=headers,
                    client=cast(Optional[AsyncHTTPHandler], client),
                    fake_stream=fake_stream,
                    json_mode=json_mode,
                    api_key=api_key,
                    stream_chunk_size=stream_chunk_size,
                )
            else:
                return self.async_completion(
                    model=model,
                    messages=messages,
                    api_base=proxy_endpoint_url,
                    model_response=model_response,
                    timeout=timeout,
                    encoding=encoding,
                    logging_obj=logging_obj,
                    stream=stream,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    credentials=credentials,
                    logger_fn=logger_fn,
                    headers=headers,
                    client=cast(Optional[AsyncHTTPHandler], client),
                    api_key=api_key,
                )

        _request_data = litellm.AmazonConverseConfig()._transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=extra_headers,
        )

        _request_data = self._scrub_converse_request(_request_data)
        data = json.dumps(_request_data)

        prepped = self.get_request_headers(
            credentials=credentials,
            aws_region_name=aws_region_name,
            extra_headers=extra_headers,
            endpoint_url=proxy_endpoint_url,
            data=data,
            headers=headers,
            api_key=api_key,
        )

        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": proxy_endpoint_url,
                "headers": dict(prepped.headers),
            },
        )

        if stream is True:
            completion_stream = make_sync_call(
                client=cast(Optional[HTTPHandler], client),
                api_base=proxy_endpoint_url,
                headers=dict(prepped.headers),
                data=data,
                model=model,
                messages=messages,
                logging_obj=logging_obj,
                json_mode=json_mode,
                fake_stream=fake_stream,
                stream_chunk_size=stream_chunk_size,
            )
            streaming_response = CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider="bedrock",
                logging_obj=logging_obj,
            )
            return streaming_response

        try:
            response = (client or _get_httpx_client()).post(
                url=proxy_endpoint_url,
                headers=prepped.headers,
                data=data,
                logging_obj=logging_obj,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        return litellm.AmazonConverseConfig()._transform_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream if isinstance(stream, bool) else False,
            logging_obj=logging_obj,
            api_key="",
            data=data,
            messages=messages,
            optional_params=optional_params,
            encoding=encoding,
        )
