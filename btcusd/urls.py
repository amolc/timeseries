from django.urls import path
from . import views

app_name = 'btcusd'

urlpatterns = [
    path('', views.btcusd_dashboard, name='dashboard'),
    path('interval/<str:interval>/', views.interval_detail, name='interval_detail'),
    path('interval/<str:interval>/lr/', views.interval_detail_lr, name='interval_detail_lr'),
    path('interval/<str:interval>/arima/', views.interval_detail_arima, name='interval_detail_arima'),
]
