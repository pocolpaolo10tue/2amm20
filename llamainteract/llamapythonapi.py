from llama_cpp import Llama
llm = Llama.from_pretrained(
    repo_id="lmstudio-community/Llama-3.3-70B-Instruct-GGUF",
    filename="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
    verbose=False
)

output = llm(
      "Q: Name the planets in the solar system? A: ",
      max_tokens=32,
      stop=["Q:", "\n"],
      echo=True
)
print(output)
