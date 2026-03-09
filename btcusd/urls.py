from django.urls import path
from . import views

app_name = 'btcusd'

urlpatterns = [
    path('', views.btcusd_dashboard, name='dashboard'),
    path('interval/<str:interval>/', views.interval_detail, name='interval_detail'),
]
