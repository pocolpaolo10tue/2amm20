from llama_cpp import Llama
import torch
import os
import multiprocessing

class LlamaInterface:
    def __init__(self,
                 repo_id="lmstudio-community/Llama-3.3-70B-Instruct-GGUF", 
                 filename="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
                 n_gpu_layers=-1,
                 n_threads=None
                ):
        """Initialize the Llama model interface with diagnostics."""
        
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        # Threads: default to number of physical CPU cores if not set
        if n_threads is None:
            n_threads = multiprocessing.cpu_count()
        
        # GPU layers: if not specified, default to 0 for CPU or partial offload
        if n_gpu_layers is None:
            n_gpu_layers = 20 if device == "cuda" else 0
        
        # Load model
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            device=device,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=True  # verbose=True for detailed load info
        )
        
        # Diagnostic information
        print("[LlamaInterface] Model loaded.")
        print(f"  - Requested device: {device}")
        print(f"  - Actual device: {self.llm.device}")
        print(f"  - CPU threads used: {n_threads}")
        print(f"  - Number of GPU layers offloaded: {n_gpu_layers}")
        print(f"  - Context size (max tokens): {self.llm.context_size}")
        print(f"  - Model filename: {filename}")
        print(f"  - Hugging Face repo ID: {repo_id}")
        
        # Optional: show environment info
        if device == "cuda":
            print(f"  - CUDA device count: {torch.cuda.device_count()}")
            print(f"  - Current CUDA device: {torch.cuda.current_device()}")
            print(f"  - CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("  - Running on CPU")

    def prompt(self, text, max_tokens=128, stop=None, echo=False):
        """Generate a response to a text prompt."""
        return self.llm(text, max_tokens=max_tokens, stop=stop, echo=echo)

    def qa(self, question, max_tokens=64):
        """Ask a question and get an answer."""
        prompt = f"Q: {question} A: "
        return self.llm(prompt, max_tokens=max_tokens, stop=["Q:", "\n"], echo=False)

    # def chat(self, message, max_tokens=128):
    #     """Simple chat interface."""
    #     return self.llm(message, max_tokens=max_tokens, echo=False)


# Example usage
if __name__ == "__main__":
    # Initialize the interface with diagnostics
    llama = LlamaInterface()
    
    # Test QA
    result = llama.qa("Name the planets in the solar system?")
    print("QA Result:", result)
    
    # Test chat (optional)
    # chat_result = llama.chat("Hello, how are you?")
    # print("Chat Result:", chat_result)
