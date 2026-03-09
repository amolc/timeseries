from django.urls import path
from . import views

app_name = 'gold'

urlpatterns = [
    path('', views.gold_dashboard, name='dashboard'),
    path('interval/<str:interval>/', views.interval_detail, name='interval_detail'),
]
