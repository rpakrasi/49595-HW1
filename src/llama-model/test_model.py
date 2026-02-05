from llama_cpp import Llama

llm = Llama(
    model_path="./debater.q4_k_m.gguf",
    n_gpu_layers=-1,  # Use all GPU layers
    chat_format="llama-3",
    n_ctx=2048,
    verbose=False
)


messages = [
    {"role": "system", "content": "You are a President Donald Trump. You are a sharp, competitive debater. You almost never back down from your stance."}
]

print("\n--- DEBATE MODE ACTIVE ---")
print("(Type 'exit' or 'quit' to stop)\n")

while True:
    user_input = input("YOU: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Debate adjourned. Goodbye!")
        break

    # Add user message to history
    messages.append({"role": "user", "content": user_input})

    print("\nTRUMP: ", end="", flush=True)

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=250,
        temperature=0.8,
        stream=True,
        # Custom:
        stop=["<|eot_id|>", "<|end_of_text|>", "assistant", "user", "\n\n\n"],
        repeat_penalty=1.18,
        frequency_penalty=0.5,
        presence_penalty=0.4
    )

    full_response = ""
    for chunk in stream:
        if "content" in chunk["choices"][0]["delta"]:
            token = chunk["choices"][0]["delta"]["content"]
            print(token, end="", flush=True)
            full_response += token

    print("\n")

    messages.append({"role": "assistant", "content": full_response})