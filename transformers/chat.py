import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def chat_with_model(model_path="./dialogue_model_final"):
    # Load the trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print("Chat with the model (type 'quit' to exit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        # Format input with special tokens
        prompt = f"<|prompter|>{user_input}<|assistant|>"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=512,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1,
                eos_token_id=tokenizer.encode("<|endoftext|>")[0]
            )
        
        # Decode and clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = response.replace(prompt, "").replace("<|endoftext|>", "").strip()
        print(f"\nBot: {response}")

if __name__ == "__main__":
    chat_with_model()