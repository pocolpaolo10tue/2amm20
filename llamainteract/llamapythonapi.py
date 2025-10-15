from llama_cpp import Llama
import torch

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
        
        print(f"[LlamaInterface] Model loaded. Actual device: {self.llm.device}")
    
    def prompt(self, text, max_tokens=128, stop=None, echo=False):
        """Generate a response to a text prompt."""
        return self.llm(text, max_tokens=max_tokens, stop=stop, echo=echo)
    
    def batch_qa(self, questions, max_tokens=64):
        """Process a list of questions in a batch."""
        prompts = [f"Q: {q} A: " for q in questions]
        return self.llm(prompts, max_tokens=max_tokens, stop=["Q:", "\n"], echo=False)
    
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
