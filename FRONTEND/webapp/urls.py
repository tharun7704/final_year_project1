from django.urls import path
from . import views

urlpatterns = [
    path('',                views.home,             name='home'),
    path('input',           views.input,            name='input'),
    path('output',          views.output,           name='output'),
    path('blood-input/',    views.blood_input,      name='blood_input'),
    path('heart_input/',    views.heart_input,      name='heart_input'),
    path('register/',       views.register_view,    name='register'),
    path('register-success/', views.register_success, name='register_success'),
]