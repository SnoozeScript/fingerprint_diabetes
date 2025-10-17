from django.contrib import admin
from django.utils.html import format_html
from .models import FingerprintRecord, MLModelMetrics

@admin.register(FingerprintRecord)
class FingerprintRecordAdmin(admin.ModelAdmin):
    list_display = [
        'user', 'predicted_risk', 'confidence_score', 'fingerprint_type', 
        'ridge_count', 'age', 'bmi', 'family_history', 'created_at'
    ]
    list_filter = [
        'predicted_risk', 'fingerprint_type', 'gender', 'family_history', 'created_at'
    ]
    search_fields = ['user__username', 'user__email', 'user__first_name', 'user__last_name']
    readonly_fields = ['created_at', 'updated_at', 'fingerprint_preview']
    ordering = ['-created_at']
    
    fieldsets = (
        ('User Information', {
            'fields': ('user',)
        }),
        ('Fingerprint Data', {
            'fields': ('fingerprint_image', 'fingerprint_preview', 'fingerprint_type', 'ridge_count')
        }),
        ('Health Information', {
            'fields': (
                'age', 'gender', 'bmi', 
                'blood_pressure_systolic', 'blood_pressure_diastolic', 
                'family_history'
            )
        }),
        ('Prediction Results', {
            'fields': ('predicted_risk', 'confidence_score')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def fingerprint_preview(self, obj):
        if obj.fingerprint_image:
            return format_html(
                '<img src="{}" style="max-height: 100px; max-width: 100px;" />',
                obj.fingerprint_image.url
            )
        return "No image"
    fingerprint_preview.short_description = "Fingerprint Preview"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')

@admin.register(MLModelMetrics)
class MLModelMetricsAdmin(admin.ModelAdmin):
    list_display = [
        'model_version', 'accuracy', 'precision', 'recall', 'f1_score', 
        'is_active', 'training_date'
    ]
    list_filter = ['is_active', 'training_date']
    readonly_fields = ['training_date']
    ordering = ['-training_date']
    
    def get_readonly_fields(self, request, obj=None):
        # Make all fields readonly except is_active for existing objects
        if obj:
            return ['model_version', 'accuracy', 'precision', 'recall', 'f1_score', 'training_date']
        return ['training_date']

# Customize admin site
admin.site.site_header = "DiabetesPred Administration"
admin.site.site_title = "DiabetesPred Admin"
admin.site.index_title = "Welcome to DiabetesPred Administration"
