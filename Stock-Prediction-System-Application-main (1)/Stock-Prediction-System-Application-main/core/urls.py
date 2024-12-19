"""core URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from app.views import index, search, predict, ticker, developer_info_view, Terms_of_service, privacy_policy, result_view, mock_page

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index),
    path('search/', search),
    path('predict/<str:ticker_value>/<str:number_of_days>/', predict),
    path('ticker/', ticker),
    path('dev/', developer_info_view, name='developer_info'),  # Add the URL pattern for dev.html
    path('terms/', Terms_of_service, name="terms_info"),      
    path('result/', result_view, name='result'),
    path('mock/', mock_page, name="mock_info"),  # URL pattern for mock.html
]
