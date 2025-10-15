from llama_cpp import Llama
import torch
import multiprocessing as mp

class LlamaInterface:
    def __init__(self,
                 repo_id="lmstudio-community/Llama-3.3-70B-Instruct-GGUF", 
                 filename="Llama-3.3-70B-Instruct-Q4_K_M.gguf"
                ):
        """Initialize the Llama model interface."""
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
         
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            device=device,
            verbose=False
        )
        
        # print(f"[LlamaInterface] Model loaded. Actual device: {self.llm.device}")
        
    def batch_qa(self, questions, max_tokens=64):
        # Use native batch if available
        if hasattr(self.llm, "create_completion_batch"):
            prompts = [{"prompt": f"Q: {q} A: "} for q in questions]
            return self.llm.create_completion_batch(
                prompts, max_tokens=max_tokens, stop=["Q:", "\n"], echo=False
            )
        else:
            # fallback to thread pool
            from concurrent.futures import ThreadPoolExecutor, as_completed
            prompts = [f"Q: {q} A: " for q in questions]
            results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(
                        self.llm.create_completion, prompt=p,
                        max_tokens=max_tokens, stop=["Q:", "\n"], echo=False
                    ): p
                    for p in prompts
                }
                for fut in as_completed(futures):
                    results.append(fut.result())
            return results

    def run_in_parallel(all_questions, n_workers=4, chunk_size=500):
        # Split questions into chunks
        chunks = [all_questions[i:i + chunk_size] for i in range(0, len(all_questions), chunk_size)]
        results = []
        with mp.Pool(n_workers) as pool:
            for ans_chunk in pool.imap(worker_fn, chunks):
                results.extend(ans_chunk)
        return results
    
    def worker_fn(chunk_of_questions):
        li = LlamaInterface()
        answers = li.batch_qa(chunk_of_questions)
        return answers
    
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
    # Initialize the interface
    llama = LlamaInterface()
    
    # Test QA
    result = llama.qa("Name the planets in the solar system?")
    print("QA Result:", result)
    
    # Test chat
    # chat_result = llama.chat("Hello, how are you?")
    # print("Chat Result:", chat_result)
