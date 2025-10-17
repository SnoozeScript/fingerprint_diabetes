from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import FingerprintRecord

class CustomUserCreationForm(UserCreationForm):
    """Custom user registration form with additional fields"""
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        if commit:
            user.save()
        return user

class FingerprintUploadForm(forms.ModelForm):
    """Form for uploading fingerprint and health data"""
    
    class Meta:
        model = FingerprintRecord
        fields = [
            'fingerprint_image', 'age', 'gender', 'bmi', 
            'blood_pressure_systolic', 'blood_pressure_diastolic', 
            'family_history'
        ]
        widgets = {
            'fingerprint_image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'capture': 'camera'  # Mobile camera capture
            }),
            'age': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 1,
                'max': 120,
                'placeholder': 'Enter your age'
            }),
            'gender': forms.Select(attrs={
                'class': 'form-control'
            }),
            'bmi': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': 0.1,
                'min': 10.0,
                'max': 50.0,
                'placeholder': 'e.g., 22.5'
            }),
            'blood_pressure_systolic': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 70,
                'max': 250,
                'placeholder': 'e.g., 120'
            }),
            'blood_pressure_diastolic': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 40,
                'max': 150,
                'placeholder': 'e.g., 80'
            }),
            'family_history': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            })
        }
        labels = {
            'fingerprint_image': 'Upload Fingerprint Image',
            'age': 'Age (years)',
            'gender': 'Gender',
            'bmi': 'BMI (Body Mass Index)',
            'blood_pressure_systolic': 'Systolic Blood Pressure (mmHg)',
            'blood_pressure_diastolic': 'Diastolic Blood Pressure (mmHg)',
            'family_history': 'Family History of Diabetes'
        }
        help_texts = {
            'fingerprint_image': 'Upload a clear image of your fingerprint (JPEG, PNG)',
            'bmi': 'BMI = Weight(kg) / Height(m)²',
            'blood_pressure_systolic': 'The top number in blood pressure reading',
            'blood_pressure_diastolic': 'The bottom number in blood pressure reading',
            'family_history': 'Check if you have family history of diabetes'
        }
    
    def clean_fingerprint_image(self):
        """Validate fingerprint image"""
        image = self.cleaned_data.get('fingerprint_image')
        if image:
            # Check file size (max 5MB)
            if image.size > 5 * 1024 * 1024:
                raise forms.ValidationError('Image file size should not exceed 5MB.')
            
            # Check file type
            allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
            if image.content_type not in allowed_types:
                raise forms.ValidationError('Only JPEG and PNG images are allowed.')
        
        return image
    
    def clean(self):
        """Additional form validation"""
        cleaned_data = super().clean()
        systolic = cleaned_data.get('blood_pressure_systolic')
        diastolic = cleaned_data.get('blood_pressure_diastolic')
        
        if systolic and diastolic:
            if systolic <= diastolic:
                raise forms.ValidationError(
                    'Systolic blood pressure must be higher than diastolic blood pressure.'
                )
        
        return cleaned_data

class BMICalculatorForm(forms.Form):
    """Helper form to calculate BMI"""
    weight = forms.FloatField(
        min_value=1.0,
        max_value=500.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': 0.1,
            'placeholder': 'Weight in kg'
        }),
        label='Weight (kg)'
    )
    height = forms.FloatField(
        min_value=0.5,
        max_value=3.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': 0.01,
            'placeholder': 'Height in meters'
        }),
        label='Height (m)'
    )
    
    def calculate_bmi(self):
        """Calculate BMI from weight and height"""
        if self.is_valid():
            weight = self.cleaned_data['weight']
            height = self.cleaned_data['height']
            return round(weight / (height ** 2), 1)
        return None
