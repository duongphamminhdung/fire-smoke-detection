import telebot
import json

with open('telebot_utils/token.json', 'r') as f:
    token = json.load(f)
    BOT_TOKEN = token["token"]
    chat_id = token["chat_id"]
bot = telebot.TeleBot(BOT_TOKEN)
@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "your chat_id is"+message.chat.id)
    
@bot.message_handler(commands=['test'])
def sign_handler(text):
    bot.send_message(chat_id, text, parse_mode="Markdown")
    # bot.register_next_step_handler(sent_msg, day_handler)

# @bot.message_handler(func=lambda msg: True)
# def echo_all(message):
#     bot.reply_to(message, message.text)

print("starting the bot")
# sign_handler('hahaha')
