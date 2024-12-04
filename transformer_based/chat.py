import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

class ChatBot:
    """
    A chatbot class that uses a pre-trained transformer model for generating responses.
    
    The chatbot loads a specified model and tokeniser, handles the conversation flow,
    and generates contextually appropriate responses.
    """

    def __init__(self, model_path: str = "./dialogue_model_final") -> None:
        """
        Initialise the chatbot with a pre-trained model.

        Args:
            model_path (str): Path to the pre-trained model directory. Defaults to './dialogue_model_final'.
        """
        # Load the trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to the user's input using the transformer model.

        Args:
            user_input (str): The text input from the user.

        Returns:
            str: The generated response from the model, cleaned of special tokens.
        """
        # Set up prompt
        prompt = (
            f"<|user|>{user_input}<|assistant|>"
        )
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=256,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.6,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.encode("<|endoftext|>")[0]
            )
        
        # Decode and clean response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = response.replace(prompt, "").replace("<|endoftext|>", "").strip()
        return response
    