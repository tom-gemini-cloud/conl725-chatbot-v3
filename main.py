import json
import random
import re
import ssl
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from rule_based.rule_based_v3 import Chatbot as RuleBasedChatbot
from transformer_based.chat import TransformerChatbot

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Initialize FastAPI app and templates
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize both chatbots
rule_based_chatbot = RuleBasedChatbot(data_path='./processed_data/processed_conversations.pkl')
transformer_chatbot = TransformerChatbot(model_path='./dialogue_model_final')

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/rule-based")
async def rule_based(request: Request):
    return templates.TemplateResponse("rules.html", {"request": request})

@app.get("/transformer-based")
async def transformer_based(request: Request):
    return templates.TemplateResponse("trans.html", {"request": request})

@app.post("/chat/rule-based")
async def chat_rule_based(request: Request):
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
    form_data = await request.form()
    user_message = form_data['message']
    bot_response = transformer_chatbot.get_response(user_message)
    return templates.TemplateResponse(
        "chat_response.html", 
        {
            "request": request, 
            "user_message": user_message, 
            "response": bot_response
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)