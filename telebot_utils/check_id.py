
from telebot import TeleBot
import json

with open('telebot_utils/token.json', 'r') as f:
    token = json.load(f)
    BOT_TOKEN = token["token"]

bot = TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start'])
def check_id(message):
    if bot.reply_to(message, "your chat_id is "+str(message.chat.id)):
        # raise Exception
        # bot.stop_bot()
        bot.stop_polling()
    
print("/start")
bot.polling()
print("stopped")