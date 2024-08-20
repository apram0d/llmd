from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

DEFAULT_RAG_PROMPT = """\
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
"""


def llama_partial_text_processor(partial_text, new_text):
    new_text = new_text.replace("[INST]", "").replace("[/INST]", "")
    partial_text += new_text
    return partial_text


SUPPORTED_LLM_MODELS = {
    "English": {
        "tiny-llama-1b-chat": {
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "remote_code": False,
            "start_message": f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}</s>\n",
            "history_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}</s> \n",
            "current_message_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}",
            "rag_prompt_template": f"""<|system|> {DEFAULT_RAG_PROMPT }</s>"""
            + """
            <|user|>
            Question: {input}
            Context: {context}
            Answer: </s>
            <|assistant|>""",
        },
        "gemma-2b-it": {
            "model_id": "google/gemma-2b-it",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT + ", ",
            "history_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}<end_of_turn>",
            "current_message_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}",
            "rag_prompt_template": f"""{DEFAULT_RAG_PROMPT},"""
            + """<start_of_turn>user{input}<end_of_turn><start_of_turn>context{context}<end_of_turn><start_of_turn>model""",
        },
        "phi-3-mini-instruct": {
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "remote_code": True,
            "start_message": f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}<|end|>\n",
            "history_template": "<|user|>\n{user}<|end|> \n<|assistant|>\n{assistant}<|end|>\n",
            "current_message_template": "<|user|>\n{user}<|end|> \n<|assistant|>\n{assistant}",
            "stop_tokens": ["<|end|>"],
        },
        "gemma-7b-it": {
            "model_id": "google/gemma-7b-it",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT + ", ",
            "history_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}<end_of_turn>",
            "current_message_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}",
            "rag_prompt_template": f"""{DEFAULT_RAG_PROMPT},"""
            + """<start_of_turn>user{input}<end_of_turn><start_of_turn>context{context}<end_of_turn><start_of_turn>model""",
        },
        "llama-2-chat-7b": {
            "model_id": "meta-llama/Llama-2-7b-chat-hf",
            "remote_code": False,
            "start_message": f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT }\n<</SYS>>\n\n",
            "history_template": "{user}[/INST]{assistant}</s><s>[INST]",
            "current_message_template": "{user} [/INST]{assistant}",
            "tokenizer_kwargs": {"add_special_tokens": False},
            "partial_text_processor": llama_partial_text_processor,
            "rag_prompt_template": f"""[INST]Human: <<SYS>> {DEFAULT_RAG_PROMPT }<</SYS>>"""
            + """
            Question: {input}
            Context: {context}
            Answer: [/INST]""",
        },
        "llama-3-8b-instruct": {
            "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "remote_code": False,
            "start_message": DEFAULT_SYSTEM_PROMPT,
            "stop_tokens": ["<|eot_id|>"],
            "rag_prompt_template": f"<|start_header_id|>system<|end_header_id|>\n\n{DEFAULT_RAG_PROMPT}<|eot_id|>"
            + """<|start_header_id|>user<|end_header_id|>


            Question: {input}
            Context: {context}
            Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>


            """,
        },
        "mistral-7b": {
            "model_id": "mistralai/Mistral-7B-v0.1",
            "remote_code": False,
            "start_message": f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT }\n<</SYS>>\n\n",
            "history_template": "{user}[/INST]{assistant}</s><s>[INST]",
            "current_message_template": "{user} [/INST]{assistant}",
            "tokenizer_kwargs": {"add_special_tokens": False},
            "partial_text_processor": llama_partial_text_processor,
            "rag_prompt_template": f"""<s> [INST] {DEFAULT_RAG_PROMPT } [/INST] </s>"""
            + """
            [INST] Question: {input}
            Context: {context}
            Answer: [/INST]""",
        },
        "zephyr-7b-beta": {
            "model_id": "HuggingFaceH4/zephyr-7b-beta",
            "remote_code": False,
            "start_message": f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}</s>\n",
            "history_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}</s> \n",
            "current_message_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}",
            "rag_prompt_template": f"""<|system|> {DEFAULT_RAG_PROMPT }</s>"""
            + """
            <|user|>
            Question: {input}
            Context: {context}
            Answer: </s>
            <|assistant|>""",
        },
        "notus-7b-v1": {
            "model_id": "argilla/notus-7b-v1",
            "remote_code": False,
            "start_message": f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}</s>\n",
            "history_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}</s> \n",
            "current_message_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}",
            "rag_prompt_template": f"""<|system|> {DEFAULT_RAG_PROMPT }</s>"""
            + """
            <|user|>
            Question: {input}
            Context: {context}
            Answer: </s>
            <|assistant|>""",
        },
        "neural-chat-7b-v3-1": {
            "model_id": "Intel/neural-chat-7b-v3-3",
            "remote_code": False,
            "start_message": f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT }\n<</SYS>>\n\n",
            "history_template": "{user}[/INST]{assistant}</s><s>[INST]",
            "current_message_template": "{user} [/INST]{assistant}",
            "tokenizer_kwargs": {"add_special_tokens": False},
            "partial_text_processor": llama_partial_text_processor,
            "rag_prompt_template": f"""<s> [INST] {DEFAULT_RAG_PROMPT } [/INST] </s>"""
            + """
            [INST] Question: {input}
            Context: {context}
            Answer: [/INST]""",
        },
    },
}
