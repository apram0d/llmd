from __future__ import annotations

import uuid
import signal
from threading import Event
from threading import Thread

import dbus.mainloop.glib
import dbus.service
import torch
from gi.repository import GLib
from optimum.intel import OVModelForCausalLM
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from transformers import TextIteratorStreamer

from llm_config import SUPPORTED_LLM_MODELS


class GenAI:
    class StopOnTokens(StoppingCriteria):
        def __init__(self, token_ids):
            """
            Initialize the StopOnTokens stopping criteria.

            Args:
                token_ids (List[int]): List of token IDs to stop generation on.
            """
            self.token_ids = token_ids

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            """
            Check if the input sequence contains any of the stop tokens.

            Args:
                input_ids (torch.LongTensor): The input sequence.
                scores (torch.FloatTensor): The scores of the generated tokens.
                **kwargs: Additional keyword arguments.

            Returns:
                bool: True if any of the stop tokens are present in the input sequence, False otherwise.
            """
            for stop_id in self.token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    def __init__(
        self,
        model_id,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        max_new_tokens,
        device
    ):
        """
        Initialize the GenAI class.

        Args:
            model_id (str): The ID of the model to use.
            temperature (float, optional): The temperature value for generation. Defaults to 0.1.
            top_k (int, optional): The value of top-k for generation. Defaults to 50.
            top_p (float, optional): The value of top-p for generation. Defaults to 0.9.
            repetition_penalty (float, optional): The value of repetition penalty for generation. Defaults to 1.1.
        """
        self.model_languages = list(SUPPORTED_LLM_MODELS)
        self.model_language = self.model_languages[0]
        self.model_ids = list(SUPPORTED_LLM_MODELS[self.model_language])
        self.model_id = model_id
        self.model_configuration = SUPPORTED_LLM_MODELS[self.model_language][
            self.model_id
        ]
        self.model_dir = (
            "/home/intel/llmd/phi-3-mini-instruct/FP16"  # TODO: update the dir
        )
        self.ov_config = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "CACHE_DIR": "",
        }
        self.model_name = self.model_configuration["model_id"]

        self.text_processor = self.model_configuration.get(
            "partial_text_processor",
            self.default_partial_text_processor,
        )
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
        )
        self.ov_model = OVModelForCausalLM.from_pretrained(
            self.model_dir,
            device=device,
            ov_config=self.ov_config,
            config=AutoConfig.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
            ),
            trust_remote_code=True,
        )

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.start_message = self.model_configuration["start_message"]
        self.history_template = self.model_configuration.get(
            "history_template",
        )
        self.current_message_template = self.model_configuration.get(
            "current_message_template",
        )
        self.stop_tokens = self.model_configuration.get("stop_tokens")
        if self.stop_tokens is not None:
            if isinstance(self.stop_tokens[0], str):
                self.stop_tokens = self.tok.convert_tokens_to_ids(
                    self.stop_tokens,
                )

            self.stop_tokens = [GenAI.StopOnTokens(self.stop_tokens)]

        self.tokenizer_kwargs = self.model_configuration.get(
            "tokenizer_kwargs",
            {},
        )

    def convert_history_to_token(self, history: list[tuple[str, str]]):
        """
        Convert the conversation history to input tokens.

        Args:
            history (List[Tuple[str, str]]): The conversation history.

        Returns:
            torch.LongTensor: The input tokens.
        """
        print(f"[DEBUG] history_template= {self.history_template}")
        print(f"[DEBUG]  start_message= {self.start_message}")
        print(f"[DEBUG] history = {history}")
        text = self.start_message + "".join(
            [
                "".join(
                    [
                        self.history_template.format(
                            num=round, user=item[0], assistant=item[1]
                        )
                    ]
                )
                for round, item in enumerate(history[:-1])
            ],
        )
        text += "".join(
            [
                "".join(
                    [
                        self.current_message_template.format(
                            num=len(history) + 1,
                            user=history[-1][0],
                            assistant=history[-1][1],
                        ),
                    ],
                ),
            ],
        )
        input_token = self.tok(
            text,
            return_tensors="pt",
            **self.tokenizer_kwargs,
        ).input_ids
        return input_token

    def default_partial_text_processor(self, partial_text: str, new_text: str):
        """
        Default partial text processor.

        Args:
            partial_text (str): The partial text.
            new_text (str): The new text to be appended.

        Returns:
            str: The processed partial text.
        """
        partial_text += new_text
        return partial_text

    def LLMGenerator(self, history):
        """
        Generate text using the LLM model.

        Args:
            history: The conversation history.

        Returns:
            str: The generated text.
        """
        input_ids = self.convert_history_to_token(history)
        if input_ids.shape[1] > 2000:
            history = [history[-1]]
            input_ids = self.convert_history_to_token(history)
        streamer = TextIteratorStreamer(
            self.tok,
            timeout=30.0,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0.0,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            streamer=streamer,
        )
        if self.stop_tokens is not None:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                self.stop_tokens,
            )

        stream_complete = Event()

        def generate_and_signal_complete():
            """
            Generation function for single thread.
            """
            global start_time
            self.ov_model.generate(**generate_kwargs)
            stream_complete.set()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()
        partial_text = ""

        for new_text in streamer:
            print(f"new text: {new_text}")
            partial_text = self.text_processor(partial_text, new_text)
            history[-1][1] = partial_text
            print(f"DEBUG {history}")
        return history[-1][1]


class LLMDService(dbus.service.Object):
    """
    A DBus service object for LLMD.

    Args:
        dbus.service.Object: The base class for DBus service objects.
    """

    def __init__(self):
        """
        Initialize the LLMDService class.
        """
        bus_name = dbus.service.BusName(
            "com.intel.llmd",
            bus=dbus.SessionBus(),
        )
        dbus.service.Object.__init__(self, bus_name, "/com/intel/llmd")
        self.clients = {}

    @dbus.service.method("com.intel.llmd", in_signature="sdiddis", out_signature="s")
    def configure(
        self,
        model_id,
        temperature=0.1,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=128,
        device="CPU",
    ):
        """
        Configure the LLMD model.

        Args:
            handler (str): The handler for the LLMD service.
            model_id (str): The ID of the model to use.
            temperature (float, optional): The temperature value for generation. Defaults to 0.1.
            top_k (int, optional): The value of top-k for generation. Defaults to 50.
            top_p (float, optional): The value of top-p for generation. Defaults to 0.9.
            repetition_penalty (float, optional): The value of repetition penalty for generation. Defaults to 1.1.
        """

        handler = str(uuid.uuid4())

        self.clients[handler] = GenAI(
            model_id,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            max_new_tokens,
            device,
        )

        return handler

    @dbus.service.method("com.intel.llmd", in_signature="ss", out_signature="s")
    def generate(self, handler, text):
        """
        Generate text using the LLMD model.

        Args:
            text (str): The input text.

        Returns:
            str: The generated text.
        """
        if handler not in self.clients.keys():
            return ValueError("invalid handler provided.")
        text = [[text, ""]]
        output = self.clients[handler].LLMGenerator(text)
        print(f"LLMD returned: {output}")
        return output

    @dbus.service.method("com.intel.llmd", out_signature="as")
    def supported_models(self):
        """
        Get the list of supported models.

        Returns:
            List[str]: The list of supported models.
        """
        model_languages = list(SUPPORTED_LLM_MODELS)
        model_language = model_languages[0]
        model_ids = list(SUPPORTED_LLM_MODELS[model_language])
        return model_ids


if __name__ == "__main__":
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    service = LLMDService()
    loop = GLib.MainLoop()
    print("LLMD Service Running...")
    loop.run()
