from django.urls import path
from . import views

app_name = 'fingerprint_pred'

urlpatterns = [
    # Main pages
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('upload/', views.upload_fingerprint, name='upload_fingerprint'),
    path('result/<int:record_id>/', views.prediction_result, name='prediction_result'),
    
    # Tools
    path('bmi-calculator/', views.bmi_calculator, name='bmi_calculator'),
    
    # User management
    path('register/', views.register_view, name='register'),
    
    # Record management
    path('delete/<int:record_id>/', views.delete_record, name='delete_record'),
    
    # Information pages
    path('about/', views.about, name='about'),
    path('privacy/', views.privacy_policy, name='privacy_policy'),
    
    # Admin
    path('admin/train-model/', views.train_model_view, name='train_model'),
    
    # API endpoints
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/health/', views.health_check, name='health_check'),
]
