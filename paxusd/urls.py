from django.urls import path
from . import views

app_name = 'paxusd'

urlpatterns = [
    path('', views.paxusd_dashboard, name='dashboard'),
    path('interval/<str:interval>/', views.interval_detail, name='interval_detail'),
]
