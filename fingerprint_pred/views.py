from django.shortcuts import render
from django.http import HttpResponse

def home(request):
    """Home page view"""
    return HttpResponse("""
    <html>
        <head>
            <title>Fingerprint Diabetes Prediction</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .status { background: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 Fingerprint-Based Diabetes Prediction System</h1>
                <div class="status">
                    <h3>✅ Django Setup Complete!</h3>
                    <p><strong>Status:</strong> Django 5.2.7 is running with uv package manager</p>
                    <p><strong>Python:</strong> 3.13.7</p>
                    <p><strong>Installed Packages:</strong></p>
                    <ul>
                        <li>Django 5.2.7</li>
                        <li>Pillow (Image processing)</li>
                        <li>OpenCV (Computer vision)</li>
                        <li>Scikit-learn (Machine learning)</li>
                        <li>Pandas & NumPy (Data processing)</li>
                    </ul>
                </div>
                <h3>Next Steps:</h3>
                <ul>
                    <li>Create fingerprint upload form</li>
                    <li>Implement feature extraction</li>
                    <li>Train ML model on dataset</li>
                    <li>Build prediction API</li>
                </ul>
            </div>
        </body>
    </html>
    """)

def health_check(request):
    """Health check endpoint"""
    return HttpResponse("OK - Django with uv is running!")
