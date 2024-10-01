import discord
from discord.ext import commands
from cognition.engines.chat_engine import ChatEngine, Conversation
from cognition.llms.ollama import OllamaModel
from cognition.models.chat_models import ChatMessage

# Initialize the bot with command prefix and intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
intents.guild_messages = True
intents.guild_reactions = True
intents.guild_typing = True
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

# Initialize the chat engine
system_prompt = "You are May, the a Discord assistant designed to help users with their questions and tasks. You speak like a regular discord user does, rarely useing capital letters and occassionally using markdown."
conversation = Conversation()
conversation.add_message(ChatMessage(role="system", content=system_prompt))
llm = OllamaModel()
chat_engine = ChatEngine(llm=llm, conversation=conversation)

@bot.event
async def on_ready():
    print(f'Bot is ready. Logged in as {bot.user}')
    client_id = bot.user.id
    invite_url = f"https://discord.com/oauth2/authorize?client_id={client_id}&scope=bot+applications.commands&permissions=2147483648"
    print(f"Invite the bot using this URL: {invite_url}")

    # Sync the commands with Discord
    try:
        await bot.tree.sync()
        print("Commands synced successfully.")
    except Exception as e:
        print(f"Error syncing commands: {e}")

@bot.command(name='chat')
async def chat(ctx, *, message: str):
    """Start a chat thread with the bot."""
    try:
        # Create a thread
        thread = await ctx.channel.create_thread(name=f"Chat with {ctx.author}", type=discord.ChannelType.public_thread)
        
        # Add the user's message to the conversation
        conversation.add_message(ChatMessage(role="user", content=message))
        
        # Generate the response
        response = chat_engine.chat(message)
        conversation.add_message(ChatMessage(role="assistant", content=response))
        
        # Send the response in the thread
        await thread.send(response)
    except Exception as e:
        await ctx.send(f"An error occurred: {e}")
        print(f"Error in /chat command: {e}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message):
        user_message = message.content.replace(f'@{bot.user.name}', '').strip()
        conversation.add_message(ChatMessage(role="user", content=user_message))
        response = chat_engine.chat(user_message)
        conversation.add_message(ChatMessage(role="assistant", content=response))
        await message.channel.send(response)

    await bot.process_commands(message)

# Run the bot with your token
if __name__ == "__main__":
    import os
    TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    bot.run(TOKEN)