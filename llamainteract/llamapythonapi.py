from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import multiprocessing as mp

class LlamaInterface:
    def __init__(self,
                 repo_id="lmstudio-community/Llama-3.3-70B-Instruct-GGUF",
                 filename="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
                 n_gpu_layers=-1,
                 n_batch=512):
        """Initialize Llama model for local inference (Python 3.9-safe)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            device=device,
            n_gpu_layers=n_gpu_layers if device == "cuda" else 0,
            n_batch=n_batch,
            use_mmap=True,
            use_mlock=True,
            verbose=False,
        )

    def qa(self, question, max_tokens=64):
        prompt = f"Q: {question} A: "
        return self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            stop=["Q:", "\n"],
            echo=False,
        )

    def batch_qa(self, questions, max_tokens=64):
        prompts = [f"Q: {q} A: " for q in questions]
        results = self.llm.create_completion(
            prompt=prompts,
            max_tokens=max_tokens,
            stop=["Q:", "\n"],
            echo=False
        )
        return results


    def prompt(self, text, max_tokens=128, stop=None, echo=False):
        """Generate a response to a text prompt."""
        return self.llm(text, max_tokens=max_tokens, stop=stop, echo=echo)

    def qa(self, question, max_tokens=64):
        """Ask a question and get an answer."""
        prompt = f"Q: {question} A: "
        return self.llm(prompt, max_tokens=max_tokens, stop=["Q:", "\n"], echo=False)
    
if __name__ == "__main__":
    llama = LlamaInterface()
    questions = ["What is the capital of France?", "Name the largest planet."]
    answers = llama.batch_qa(questions)
    for ans in answers:
        print(ans["choices"][0]["text"].strip())
