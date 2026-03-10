import asyncio
from pathlib import Path

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer

from utils.live_price import get_last_price_payload


class USOILLivePriceConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self._running = True
        self._task = asyncio.create_task(self._stream())

    async def disconnect(self, close_code):
        self._running = False
        task = getattr(self, "_task", None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _stream(self):
        processed = Path(__file__).resolve().parent / "data" / "processed" / "usoil_1h_processed.csv"
        while self._running:
            payload = await sync_to_async(get_last_price_payload)("usoil", "USOIL/TVC", processed, True)
            await self.send_json(payload)
            await asyncio.sleep(5)
