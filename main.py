# Import required libraries
import json
import random
import re
import ssl
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from rule_based.rule_based_v2 import Chatbot as RuleBasedChatbot
from transformer_based.chat import ChatBot as TransformerChatbot

# Handle SSL certificate verification
# This is required for certain NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Initialise FastAPI app and configure static files and templates
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialise both chatbots
# Rule-based uses pre-processed conversation data
# Transformer-based uses a fine-tuned dialogue model
rule_based_chatbot = RuleBasedChatbot()
transformer_chatbot = TransformerChatbot()

# Route handlers
@app.get("/")
async def read_root(request: Request):
    """Serve the main landing page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/rule-based")
async def rule_based(request: Request):
    """Serve the rule-based chatbot interface"""
    return templates.TemplateResponse("rules.html", {"request": request})

@app.get("/transformer-based")
async def transformer_based(request: Request):
    """Serve the transformer-based chatbot interface"""
    return templates.TemplateResponse("trans.html", {"request": request})

# Chat endpoint handlers
@app.post("/chat/rule-based")
async def chat_rule_based(request: Request):
    """Handle chat interactions with the rule-based model
    Returns both the user's message and the bot's response"""
    form_data = await request.form()
    user_message = form_data['message']
    bot_response = rule_based_chatbot.get_response(user_message)
    return templates.TemplateResponse(
        "chat_response.html", 
        {
            "request": request, 
            "user_message": user_message, 
            "response": bot_response
        }
    )

@app.post("/chat/transformer-based")
async def chat_transformer_based(request: Request):
    """Handle chat interactions with the transformer-based model
    Returns both the user's message and the bot's response"""
    form_data = await request.form()
    user_message = form_data['message']
    bot_response = transformer_chatbot.generate_response(user_message)
    return templates.TemplateResponse(
        "chat_response.html", 
        {
            "request": request, 
            "user_message": user_message, 
            "response": bot_response
        }
    )

# Run the application if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)