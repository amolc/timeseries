from django.urls import path

from .consumers import USOILLivePriceConsumer

websocket_urlpatterns = [
    path("ws/usoil/live-price/", USOILLivePriceConsumer.as_asgi()),
]
