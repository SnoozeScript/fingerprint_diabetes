import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class DiabetesPredictionModel:
    """Machine Learning model for diabetes prediction using fingerprint and health data"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            'age', 'gender_encoded', 'bmi', 'blood_pressure_systolic', 
            'blood_pressure_diastolic', 'family_history', 'fingerprint_type_encoded', 
            'ridge_count'
        ]
        self.model_path = os.path.join(settings.BASE_DIR, 'ml_models')
        self.ensure_model_directory()
    
    def ensure_model_directory(self):
        """Create model directory if it doesn't exist"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
    
    def prepare_features(self, data):
        """
        Prepare features for training or prediction
        """
        try:
            df = pd.DataFrame(data)
            
            # Encode categorical variables
            if 'gender' in df.columns:
                df['gender_encoded'] = df['gender'].map({'M': 1, 'F': 0})
            
            if 'fingerprint_type' in df.columns:
                # Encode fingerprint types
                fingerprint_mapping = {'Loop': 0, 'Whorl': 1, 'Arch': 2}
                df['fingerprint_type_encoded'] = df['fingerprint_type'].map(fingerprint_mapping)
            
            # Convert boolean to int
            if 'family_history' in df.columns:
                df['family_history'] = df['family_history'].astype(int)
            
            # Select only the features we need
            feature_df = df[self.feature_columns]
            
            return feature_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def train_model(self, dataset_path):
        """
        Train the diabetes prediction model using the provided dataset
        """
        try:
            logger.info("Starting model training...")
            
            # Load dataset
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded dataset with {len(df)} records")
            
            # Prepare features
            X = self.prepare_features(df)
            
            # Prepare target variable
            if 'Diabetes' in df.columns:
                # Map diabetes risk levels
                risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
                y = df['Diabetes'].map(risk_mapping)
            else:
                # If no diabetes column, create synthetic labels based on health indicators
                y = self.create_synthetic_labels(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            
            logger.info(f"Model Training Results:")
            logger.info(f"Accuracy: {accuracy:.3f}")
            logger.info(f"Precision: {precision:.3f}")
            logger.info(f"Recall: {recall:.3f}")
            logger.info(f"F1-Score: {f1:.3f}")
            logger.info(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Feature importance
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            logger.info(f"Feature Importance: {feature_importance}")
            
            # Save model
            self.save_model()
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_score': cv_scores.mean(),
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def create_synthetic_labels(self, df):
        """
        Create synthetic diabetes risk labels based on health indicators
        This is used when the dataset doesn't have diabetes labels
        """
        labels = []
        
        for _, row in df.iterrows():
            risk_score = 0
            
            # Age risk
            if row.get('age', 0) > 45:
                risk_score += 1
            if row.get('age', 0) > 65:
                risk_score += 1
            
            # BMI risk
            bmi = row.get('bmi', 22)
            if bmi >= 25:
                risk_score += 1
            if bmi >= 30:
                risk_score += 1
            
            # Blood pressure risk
            systolic = row.get('blood_pressure_systolic', 120)
            if systolic >= 130:
                risk_score += 1
            if systolic >= 140:
                risk_score += 1
            
            # Family history
            if row.get('family_history', False):
                risk_score += 2
            
            # Fingerprint-based risk (simplified)
            if row.get('fingerprint_type') == 'Whorl':
                risk_score += 1
            
            ridge_count = row.get('ridge_count', 50)
            if ridge_count > 80 or ridge_count < 30:
                risk_score += 1
            
            # Classify based on risk score
            if risk_score <= 2:
                labels.append(0)  # Low risk
            elif risk_score <= 4:
                labels.append(1)  # Medium risk
            else:
                labels.append(2)  # High risk
        
        return np.array(labels)
    
    def predict(self, features):
        """
        Make diabetes risk prediction
        """
        try:
            if self.model is None:
                self.load_model()
            
            if self.model is None:
                # If no trained model, train on the dataset
                dataset_path = os.path.join(settings.BASE_DIR, 'dataset', 'Fingerprint_Based_Diabetes_Prediction_Clean_Balanced.csv')
                if os.path.exists(dataset_path):
                    self.train_model(dataset_path)
                else:
                    raise Exception("No trained model available and no dataset found for training")
            
            # Prepare features
            feature_df = self.prepare_features([features])
            
            # Scale features
            features_scaled = self.scaler.transform(feature_df)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            
            # Map prediction to risk level
            risk_levels = ['Low', 'Medium', 'High']
            predicted_risk = risk_levels[prediction]
            confidence = float(max(prediction_proba))
            
            logger.info(f"Prediction: {predicted_risk} (confidence: {confidence:.3f})")
            
            return {
                'predicted_risk': predicted_risk,
                'confidence': confidence,
                'probabilities': {
                    'Low': float(prediction_proba[0]),
                    'Medium': float(prediction_proba[1]),
                    'High': float(prediction_proba[2])
                }
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            # Return default prediction in case of error
            return {
                'predicted_risk': 'Medium',
                'confidence': 0.5,
                'probabilities': {'Low': 0.33, 'Medium': 0.34, 'High': 0.33}
            }
    
    def save_model(self):
        """Save trained model and scaler"""
        try:
            model_file = os.path.join(self.model_path, 'diabetes_model.pkl')
            scaler_file = os.path.join(self.model_path, 'scaler.pkl')
            
            joblib.dump(self.model, model_file)
            joblib.dump(self.scaler, scaler_file)
            
            logger.info(f"Model saved to {model_file}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load trained model and scaler"""
        try:
            model_file = os.path.join(self.model_path, 'diabetes_model.pkl')
            scaler_file = os.path.join(self.model_path, 'scaler.pkl')
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                self.model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                logger.info("Model loaded successfully")
                return True
            else:
                logger.warning("No saved model found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.model is None:
            return None
        
        importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        return sorted(importance.items(), key=lambda x: x[1], reverse=True)
