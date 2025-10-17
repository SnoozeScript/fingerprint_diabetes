from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.core.paginator import Paginator
from django.db.models import Q
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import json
import logging
import os

from .models import FingerprintRecord, MLModelMetrics
from .forms import CustomUserCreationForm, FingerprintUploadForm, BMICalculatorForm
from .fingerprint_utils import FingerprintProcessor
from .ml_model import DiabetesPredictionModel

logger = logging.getLogger(__name__)

def home(request):
    """Home page view"""
    context = {
        'total_predictions': FingerprintRecord.objects.count(),
        'recent_predictions': FingerprintRecord.objects.filter(
            predicted_risk__isnull=False
        ).order_by('-created_at')[:5] if request.user.is_authenticated else []
    }
    return render(request, 'fingerprint_pred/home.html', context)

def register_view(request):
    """User registration view"""
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('login')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'registration/register.html', {'form': form})

@login_required
def dashboard(request):
    """User dashboard showing prediction history"""
    user_records = FingerprintRecord.objects.filter(user=request.user)
    
    # Pagination
    paginator = Paginator(user_records, 10)
    page_number = request.GET.get('page')
    records = paginator.get_page(page_number)
    
    # Statistics
    total_predictions = user_records.count()
    risk_distribution = {
        'Low': user_records.filter(predicted_risk='Low').count(),
        'Medium': user_records.filter(predicted_risk='Medium').count(),
        'High': user_records.filter(predicted_risk='High').count(),
    }
    
    context = {
        'records': records,
        'total_predictions': total_predictions,
        'risk_distribution': risk_distribution,
    }
    
    return render(request, 'fingerprint_pred/dashboard.html', context)

@login_required
def upload_fingerprint(request):
    """Upload fingerprint and health data for prediction"""
    if request.method == 'POST':
        form = FingerprintUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Save the record (without prediction yet)
                record = form.save(commit=False)
                record.user = request.user
                record.save()
                
                # Process fingerprint image
                processor = FingerprintProcessor()
                image_path = record.fingerprint_image.path
                
                # Validate fingerprint image
                is_valid, validation_message = processor.validate_fingerprint_image(image_path)
                if not is_valid:
                    record.delete()  # Remove the record if image is invalid
                    messages.error(request, f'Invalid fingerprint image: {validation_message}')
                    return render(request, 'fingerprint_pred/upload.html', {'form': form})
                
                # Extract fingerprint features
                features = processor.extract_features(image_path)
                
                if features['success']:
                    record.fingerprint_type = features['fingerprint_type']
                    record.ridge_count = features['ridge_count']
                    
                    # Prepare data for ML prediction
                    prediction_data = {
                        'age': record.age,
                        'gender': record.gender,
                        'bmi': float(record.bmi),
                        'blood_pressure_systolic': record.blood_pressure_systolic,
                        'blood_pressure_diastolic': record.blood_pressure_diastolic,
                        'family_history': record.family_history,
                        'fingerprint_type': record.fingerprint_type,
                        'ridge_count': record.ridge_count
                    }
                    
                    # Make prediction
                    ml_model = DiabetesPredictionModel()
                    prediction_result = ml_model.predict(prediction_data)
                    
                    record.predicted_risk = prediction_result['predicted_risk']
                    record.confidence_score = prediction_result['confidence']
                    record.save()
                    
                    messages.success(request, 'Fingerprint processed successfully! Check your results below.')
                    return redirect('prediction_result', record_id=record.id)
                    
                else:
                    messages.warning(request, f'Feature extraction failed: {features.get("error", "Unknown error")}. Using default values.')
                    record.fingerprint_type = 'Loop'
                    record.ridge_count = 50
                    record.save()
                    return redirect('prediction_result', record_id=record.id)
                    
            except Exception as e:
                logger.error(f"Error processing fingerprint: {str(e)}")
                messages.error(request, 'An error occurred while processing your fingerprint. Please try again.')
                if 'record' in locals():
                    record.delete()
        else:
            messages.error(request, 'Please correct the errors in the form.')
    else:
        form = FingerprintUploadForm()
    
    return render(request, 'fingerprint_pred/upload.html', {'form': form})

@login_required
def prediction_result(request, record_id):
    """Display prediction results"""
    record = get_object_or_404(FingerprintRecord, id=record_id, user=request.user)
    
    # Get risk level color and recommendations
    risk_info = get_risk_info(record.predicted_risk)
    
    context = {
        'record': record,
        'risk_info': risk_info,
    }
    
    return render(request, 'fingerprint_pred/result.html', context)

def get_risk_info(risk_level):
    """Get risk level information and recommendations"""
    risk_data = {
        'Low': {
            'color': 'success',
            'icon': 'check-circle',
            'description': 'Your diabetes risk is low. Maintain your healthy lifestyle!',
            'recommendations': [
                'Continue regular physical activity',
                'Maintain a balanced diet',
                'Regular health check-ups',
                'Stay hydrated',
                'Manage stress levels'
            ]
        },
        'Medium': {
            'color': 'warning',
            'icon': 'exclamation-triangle',
            'description': 'You have a medium risk of diabetes. Consider lifestyle changes.',
            'recommendations': [
                'Increase physical activity to 150 minutes per week',
                'Reduce refined sugar and carbohydrate intake',
                'Monitor blood pressure regularly',
                'Maintain healthy weight (BMI 18.5-24.9)',
                'Consult with healthcare provider',
                'Consider diabetes screening tests'
            ]
        },
        'High': {
            'color': 'danger',
            'icon': 'exclamation-circle',
            'description': 'You have a high risk of diabetes. Please consult a healthcare professional.',
            'recommendations': [
                'Immediate consultation with healthcare provider',
                'Comprehensive diabetes screening',
                'Structured lifestyle modification program',
                'Regular monitoring of blood glucose',
                'Professional dietary counseling',
                'Stress management and adequate sleep',
                'Consider medication if recommended by doctor'
            ]
        }
    }
    
    return risk_data.get(risk_level, risk_data['Medium'])

@login_required
def bmi_calculator(request):
    """BMI calculator tool"""
    bmi_result = None
    bmi_category = None
    
    if request.method == 'POST':
        form = BMICalculatorForm(request.POST)
        if form.is_valid():
            bmi_result = form.calculate_bmi()
            bmi_category = get_bmi_category(bmi_result)
    else:
        form = BMICalculatorForm()
    
    context = {
        'form': form,
        'bmi_result': bmi_result,
        'bmi_category': bmi_category,
    }
    
    return render(request, 'fingerprint_pred/bmi_calculator.html', context)

def get_bmi_category(bmi):
    """Get BMI category and color"""
    if bmi < 18.5:
        return {'category': 'Underweight', 'color': 'info'}
    elif 18.5 <= bmi < 25:
        return {'category': 'Normal weight', 'color': 'success'}
    elif 25 <= bmi < 30:
        return {'category': 'Overweight', 'color': 'warning'}
    else:
        return {'category': 'Obese', 'color': 'danger'}

@csrf_exempt
@require_http_methods(["POST"])
def api_predict(request):
    """API endpoint for diabetes prediction"""
    try:
        data = json.loads(request.body)
        
        # Validate required fields
        required_fields = ['age', 'gender', 'bmi', 'blood_pressure_systolic', 
                          'blood_pressure_diastolic', 'family_history']
        
        for field in required_fields:
            if field not in data:
                return JsonResponse({'error': f'Missing required field: {field}'}, status=400)
        
        # Add default fingerprint data if not provided
        if 'fingerprint_type' not in data:
            data['fingerprint_type'] = 'Loop'
        if 'ridge_count' not in data:
            data['ridge_count'] = 50
        
        # Make prediction
        ml_model = DiabetesPredictionModel()
        prediction_result = ml_model.predict(data)
        
        return JsonResponse({
            'predicted_risk': prediction_result['predicted_risk'],
            'confidence': prediction_result['confidence'],
            'probabilities': prediction_result['probabilities'],
            'recommendations': get_risk_info(prediction_result['predicted_risk'])['recommendations']
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return JsonResponse({'error': 'Internal server error'}, status=500)

def health_check(request):
    """Health check endpoint"""
    return JsonResponse({
        'status': 'OK',
        'service': 'Fingerprint Diabetes Prediction API',
        'version': '1.0.0'
    })

@login_required
def delete_record(request, record_id):
    """Delete a fingerprint record"""
    if request.method == 'POST':
        record = get_object_or_404(FingerprintRecord, id=record_id, user=request.user)
        record.delete()
        messages.success(request, 'Record deleted successfully.')
    
    return redirect('dashboard')

def about(request):
    """About page with information about the system"""
    return render(request, 'fingerprint_pred/about.html')

def privacy_policy(request):
    """Privacy policy page"""
    return render(request, 'fingerprint_pred/privacy.html')

@login_required
def train_model_view(request):
    """Train the ML model (admin only)"""
    if not request.user.is_staff:
        messages.error(request, 'You do not have permission to access this page.')
        return redirect('home')
    
    if request.method == 'POST':
        try:
            dataset_path = os.path.join(settings.BASE_DIR, 'dataset', 'Fingerprint_Based_Diabetes_Prediction_Clean_Balanced.csv')
            
            if not os.path.exists(dataset_path):
                messages.error(request, 'Dataset file not found. Please upload the dataset first.')
                return render(request, 'fingerprint_pred/train_model.html')
            
            # Train model
            ml_model = DiabetesPredictionModel()
            metrics = ml_model.train_model(dataset_path)
            
            # Save metrics
            model_metrics = MLModelMetrics.objects.create(
                model_version=f"v{MLModelMetrics.objects.count() + 1}",
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                is_active=True
            )
            
            # Deactivate previous models
            MLModelMetrics.objects.exclude(id=model_metrics.id).update(is_active=False)
            
            messages.success(request, f'Model trained successfully! Accuracy: {metrics["accuracy"]:.3f}')
            
        except Exception as e:
            logger.error(f"Model training error: {str(e)}")
            messages.error(request, f'Model training failed: {str(e)}')
    
    # Get model history
    model_history = MLModelMetrics.objects.all()
    
    return render(request, 'fingerprint_pred/train_model.html', {'model_history': model_history})
