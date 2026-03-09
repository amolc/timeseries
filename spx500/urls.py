from django.urls import path
from . import views

app_name = 'spx500'

urlpatterns = [
    path('', views.spx500_dashboard, name='dashboard'),
    path('interval/<str:interval>/', views.interval_detail, name='interval_detail'),
]
