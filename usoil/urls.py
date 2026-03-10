from django.urls import path
from . import views

app_name = 'usoil'

urlpatterns = [
    path('', views.usoil_dashboard, name='dashboard'),
    path('api/last-price/', views.last_price_api, name='last_price_api'),
    path('interval/<str:interval>/', views.interval_detail, name='interval_detail'),
]
