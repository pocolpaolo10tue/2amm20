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

    def batch_qa(self, questions, max_tokens=64, max_workers=8):
        """
        Threaded batching for Python 3.9 environments.
        Compatible with llama_cpp<=0.3.16.
        """
        prompts = [f"Q: {q} A: " for q in questions]
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.llm.create_completion,
                    prompt=p,
                    max_tokens=max_tokens,
                    stop=["Q:", "\n"],
                    echo=False
                ): p
                for p in prompts
            }
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({"choices": [{"text": f"[ERROR: {e}]"}]})
        # preserve ordering (optional)
        results.sort(key=lambda x: prompts.index(futures.get(x, "")))
        return results

    def prompt(self, text, max_tokens=128, stop=None, echo=False):
        """Generate a response to a text prompt."""
        return self.llm(text, max_tokens=max_tokens, stop=stop, echo=echo)

    def qa(self, question, max_tokens=64):
        """Ask a question and get an answer."""
        prompt = f"Q: {question} A: "
        return self.llm(prompt, max_tokens=max_tokens, stop=["Q:", "\n"], echo=False)