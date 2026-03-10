from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

import btcusd.routing
import gold.routing
import nifty.routing
import paxusd.routing
import spx500.routing
import usoil.routing

all_websocket_patterns = (
    nifty.routing.websocket_urlpatterns
    + btcusd.routing.websocket_urlpatterns
    + gold.routing.websocket_urlpatterns
    + paxusd.routing.websocket_urlpatterns
    + spx500.routing.websocket_urlpatterns
    + usoil.routing.websocket_urlpatterns
)

application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": URLRouter(all_websocket_patterns),
    }
)
