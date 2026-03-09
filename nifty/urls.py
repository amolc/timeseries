from django.urls import path
from . import views

app_name = 'nifty'

urlpatterns = [
    path('', views.nifty_dashboard, name='dashboard'),
    path('interval/<str:interval>/', views.interval_detail, name='interval_detail'),
]
