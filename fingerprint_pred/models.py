from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
import os

def fingerprint_upload_path(instance, filename):
    """Generate upload path for fingerprint images"""
    return f'fingerprints/{instance.user.username}/{filename}'

class FingerprintPattern(models.TextChoices):
    """Fingerprint pattern types"""
    LOOP = 'Loop', 'Loop'
    WHORL = 'Whorl', 'Whorl'
    ARCH = 'Arch', 'Arch'

class DiabetesRiskLevel(models.TextChoices):
    """Diabetes risk levels"""
    LOW = 'Low', 'Low Risk'
    MEDIUM = 'Medium', 'Medium Risk'
    HIGH = 'High', 'High Risk'

class FingerprintRecord(models.Model):
    """Model to store fingerprint data and health information"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='fingerprint_records')
    
    # Fingerprint image and extracted features
    fingerprint_image = models.ImageField(upload_to=fingerprint_upload_path)
    fingerprint_type = models.CharField(
        max_length=10, 
        choices=FingerprintPattern.choices,
        null=True, blank=True
    )
    ridge_count = models.IntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(200)]
    )
    
    # Health data for prediction
    age = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(120)])
    gender = models.CharField(max_length=1, choices=[('M', 'Male'), ('F', 'Female')])
    bmi = models.FloatField(
        validators=[MinValueValidator(10.0), MaxValueValidator(50.0)],
        help_text="Body Mass Index"
    )
    blood_pressure_systolic = models.IntegerField(
        validators=[MinValueValidator(70), MaxValueValidator(250)]
    )
    blood_pressure_diastolic = models.IntegerField(
        validators=[MinValueValidator(40), MaxValueValidator(150)]
    )
    family_history = models.BooleanField(default=False, help_text="Family history of diabetes")
    
    # Prediction results
    predicted_risk = models.CharField(
        max_length=10,
        choices=DiabetesRiskLevel.choices,
        null=True, blank=True
    )
    confidence_score = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Fingerprint Record'
        verbose_name_plural = 'Fingerprint Records'
    
    def __str__(self):
        return f"{self.user.username} - {self.predicted_risk or 'Pending'} Risk ({self.created_at.strftime('%Y-%m-%d')})"
    
    def delete(self, *args, **kwargs):
        """Delete associated image file when record is deleted"""
        if self.fingerprint_image:
            if os.path.isfile(self.fingerprint_image.path):
                os.remove(self.fingerprint_image.path)
        super().delete(*args, **kwargs)

class MLModelMetrics(models.Model):
    """Model to store ML model performance metrics"""
    model_version = models.CharField(max_length=50, unique=True)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    training_date = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-training_date']
    
    def __str__(self):
        return f"Model {self.model_version} - Accuracy: {self.accuracy:.3f}"
