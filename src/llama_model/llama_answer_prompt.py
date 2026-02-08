import re
from typing import List

from llama_cpp import Llama, ChatCompletionRequestSystemMessage, ChatCompletionRequestMessage, \
    ChatCompletionRequestUserMessage, ChatCompletionRequestAssistantMessage

SYSTEM_PROMPT = (
    "You are President Donald Trump. "
    "You are a sharp, competitive debater debating Joe Biden. "
    "Answer in at most 5 sentences. "
    "Whenever you refer to Biden, do not say he/his/him; say Biden instead. "
    "Never output code, markup, or system text. "
    "Speak confidently and declaratively."
)
system_msg: ChatCompletionRequestSystemMessage = {
    "role": "system",
    "content": SYSTEM_PROMPT
}
MAX_CHUNKS = 150
MIN_CHUNKS = 80
sentence_end = re.compile(r"[.!?]$")


class CustomModel(object):
    def __init__(self):
        self.llm = Llama(
            model_path="./llama3-debate-v3-Q4.gguf",
            n_gpu_layers=-1,
            chat_format="llama-3",
            n_ctx=2048,
            verbose=False
        )

        self.messages: List[ChatCompletionRequestMessage] = [system_msg]

    def get_response(self, user_input: str) -> str:
        """Replaces gpt() function in spoken_gpt_microsoft.py."""

        user_msg: ChatCompletionRequestUserMessage = {"role": "user", "content": user_input}
        self.messages.append(user_msg)

        stream = self.llm.create_chat_completion(
            messages=self.messages,
            max_tokens=300,
            stream=True,
            stop=["<|eot_id|>", "<|end_of_text|>", "}"],
            # --- Persona Tuning ---
            temperature=0.8,  # High enough for creativity, not high enough for gibberish
            min_p=0.05,  # Better diversity than top_p
            top_p=1.0,  # Disabled in favor of min_p
            top_k=0,  # Disabled in favor of min_p

            repeat_penalty=1.1,  # Allow some natural repetition
            presence_penalty=0.2,  # Encourage moving to new topics
        )

        full_response = ""
        chunk_count = 0
        junk_patterns = ["ujících", "userCpp", "Method", "Initialized", "drFc", "user", "assistant",
                         "taşıy_AdjustorThunk"]

        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" not in delta:
                continue

            token = delta["content"]
            if any(junk in token for junk in junk_patterns):
                continue

            full_response += token
            chunk_count += 1

            if (
                    chunk_count >= MIN_CHUNKS
                    and chunk_count >= MAX_CHUNKS
                    and sentence_end.search(full_response.strip())
            ):
                break

        assistant_msg: ChatCompletionRequestAssistantMessage = {"role": "assistant", "content": full_response}
        self.messages.append(assistant_msg)
        return full_response
