"""Run locally once to create TELEGRAM_SESSION; never share the printed value."""
import asyncio
import os
from telethon import TelegramClient
from telethon.sessions import StringSession


async def main() -> None:
    api_id = int(os.environ["TELEGRAM_API_ID"])
    api_hash = os.environ["TELEGRAM_API_HASH"]
    client = TelegramClient(StringSession(), api_id, api_hash)
    await client.start()
    me = await client.get_me()
    print(f"\nAuthenticated as {getattr(me, 'username', None) or me.id}.")
    print("Add this value to Railway as TELEGRAM_SESSION. Treat it like your account password.\n")
    print("TELEGRAM_SESSION=" + client.session.save())
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
