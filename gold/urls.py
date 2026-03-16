from django.urls import path
from . import views

app_name = 'gold'

urlpatterns = [
    path('', views.gold_dashboard, name='dashboard'),
    path('interval/<str:interval>/', views.interval_detail_arima, name='interval_detail'),
    path('interval/<str:interval>/arima/', views.interval_detail_arima, name='interval_detail_arima'),
]
