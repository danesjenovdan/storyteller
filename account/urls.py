from django.urls import path

from . import views

urlpatterns = [
    path("registration", views.registration, name="registration"),
    path("login", views.AccountLoginView.as_view(), name="login"),
    path("logout", views.AccountLogoutView.as_view(), name="logout"),
]
