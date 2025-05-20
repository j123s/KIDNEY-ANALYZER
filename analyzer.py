import os
import sys
import logging
from datetime import datetime
from flask import (
    Flask, 
    request, 
    render_template_string, 
    send_file, 
    session, 
    redirect, 
    url_for
)
from flask import send_file

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from io import BytesIO

# ==================== CONFIGURATION ====================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-please-change-in-prod')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kidney_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# ==================== ML MODEL SERVICE ====================
class KidneyModelService:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'age', 'bp', 'sg', 'al', 'su', 'hemo'
        ]
        self.load_model()

    def load_model(self):
        """Load or create the ML model"""
        try:
            if os.path.exists('kidney_model.pkl'):
                self.model = joblib.load('kidney_model.pkl')
                logging.info("Loaded pre-trained model from disk")
            else:
                self.train_model()
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            self.train_model()  # Fallback to training

    def train_model(self):
        """Train model using real clinical data"""
        try:
            # Try loading from UCI repository
            ckd_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00337/Chronic_Kidney_Disease.csv"
            df = pd.read_csv(ckd_url, na_values=['?','\t?'])
            
            # Preprocessing
            df = df.dropna()
            df['class'] = df['class'].map({'ckd':1, 'notckd':0})
            
            # Feature selection
            X = df[self.feature_names]
            y = df['class']
            
            # Model training
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X, y)
            
            # Save model
            joblib.dump(self.model, 'kidney_model.pkl')
            logging.info("Trained new model with clinical data")
            
        except Exception as e:
            logging.warning(f"Using synthetic data: {str(e)}")
            # Fallback to synthetic data
            data = {
                'age': [48,53,63,42,58,61,45,50,55,60,65,70,40,35,72]*20,
                'bp': [80,90,70,80,90,80,70,80,90,100,85,75,95,65,110]*20,
                'sg': [1.02,1.02,1.01,1.01,1.02,1.01,1.02,1.01,1.02,1.01,1.02,1.01,1.02,1.01,1.01]*20,
                'al': [1,2,3,1,2,3,1,2,3,1,2,3,1,2,4]*20,
                'su': [0,0,0,1,1,0,1,0,1,0,1,0,1,0,2]*20,
                'hemo': [12.5,11.8,9.5,13.1,10.2,8.7,14.0,11.5,9.8,12.1,10.5,8.9,13.5,12.8,7.5]*20,
                'class': [0,1,1,0,1,1,0,1,1,1,0,1,0,0,1]*20
            }
            df = pd.DataFrame(data)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(df[self.feature_names], df['class'])
            joblib.dump(self.model, 'kidney_model.pkl')

# Initialize service
model_service = KidneyModelService()

# ==================== CLINICAL CALCULATIONS ====================
class ClinicalCalculator:
    @staticmethod
    def calculate_egfr(creatinine, age, gender, race='other'):
        """Enhanced CKD-EPI formula with race adjustment"""
        if gender == 'female':
            k = 0.7
            alpha = -0.329
        else:
            k = 0.9
            alpha = -0.411
        
        cr_ratio = creatinine / k
        egfr = 141 * (min(cr_ratio, 1)**alpha) * (max(cr_ratio, 1)**-1.209) * (0.993**age)
        
        # Adjustments
        if gender == 'female':
            egfr *= 1.018
        if race == 'black':
            egfr *= 1.159
            
        return round(max(egfr, 1), 1)  # Ensure eGFR doesn't go below 1

    @staticmethod
    def get_ckd_stage(egfr):
        """Return CKD stage with description"""
        if egfr >= 90: 
            return "G1", "Normal kidney function"
        elif egfr >= 60: 
            return "G2", "Mildly reduced kidney function"
        elif egfr >= 45: 
            return "G3a", "Mild-moderate CKD"
        elif egfr >= 30: 
            return "G3b", "Moderate-severe CKD"
        elif egfr >= 15: 
            return "G4", "Severe CKD"
        else: 
            return "G5", "Kidney failure (needs dialysis)"

    @staticmethod
    def calculate_water_intake(egfr, weight_kg):
        """Personalized fluid recommendations"""
        if egfr < 15:
            return 1000  # Strict restriction for dialysis patients
        elif egfr < 30:
            return min(1500, weight_kg * 20)
        else:
            return weight_kg * 30  # Standard recommendation

    @staticmethod
    def interpret_creatinine(creatinine, age, gender):
        """Detailed creatinine interpretation"""
        ranges = {
            'male': {
                '18-30': (0.7, 1.3),
                '30-60': (0.7, 1.3),
                '60+': (0.6, 1.2)
            },
            'female': {
                '18-30': (0.5, 1.1),
                '30-60': (0.5, 1.1),
                '60+': (0.5, 1.0)
            }
        }
        
        # Determine age group
        age_group = '60+' if age >= 60 else '30-60' if age >= 30 else '18-30'
        low, high = ranges[gender][age_group]
        
        if creatinine < low:
            status = "Low (possible muscle loss or malnutrition)"
        elif creatinine > high * 1.5:
            status = "Dangerously High (acute kidney injury possible)"
        elif creatinine > high:
            status = "High (possible kidney impairment)"
        else:
            status = "Normal"
        
        return {
            'value': creatinine,
            'normal_range': f"{low}-{high} mg/dL",
            'status': status,
            'age_group': f"{gender.title()}, {age_group}"
        }

# ==================== REPORT GENERATOR ====================
class PDFReportGenerator:
    def __init__(self):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
    
    def generate_report(self, patient_data, result_data, diet_plan):
        """Generate comprehensive PDF report"""
        # Setup
        self.pdf.add_page()
        self._add_header()
        self._add_patient_info(patient_data)
        self._add_results_section(result_data)
        self._add_diet_section(diet_plan)
        self._add_footer()
        
        # Save to buffer
        pdf_buffer = BytesIO()
        self.pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        return pdf_buffer

    def _add_header(self):
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Kidney Health Report', ln=1, align='C')
        self.pdf.set_font('Arial', '', 12)
        self.pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1, align='C')
        self.pdf.ln(10)

    def _add_patient_info(self, data):
        self.pdf.set_font('Arial', 'B', 14)
        self.pdf.cell(0, 10, 'Patient Information', ln=1)
        self.pdf.set_font('Arial', '', 12)
        
        info = [
            f"Age: {data.get('age', 'N/A')} years",
            f"Gender: {data.get('gender', 'N/A').title()}",
            f"Weight: {data.get('weight', 'N/A')} kg",
            f"Blood Pressure: {data.get('bp', 'N/A')} mmHg",
            f"Creatinine: {data.get('creatinine', 'N/A')} mg/dL"
        ]
        
        for item in info:
            self.pdf.cell(0, 10, item, ln=1)
        self.pdf.ln(5)

    def _add_results_section(self, results):
        self.pdf.set_font('Arial', 'B', 14)
        self.pdf.cell(0, 10, 'Test Results', ln=1)
        self.pdf.set_font('Arial', '', 12)
        
        # Risk assessment
        risk_status = "High Risk" if results.get('prediction') else "Low Risk"
        risk_color = (255, 0, 0) if results.get('prediction') else (0, 128, 0)
        self.pdf.set_text_color(*risk_color)
        self.pdf.cell(0, 10, f"Kidney Disease Risk: {risk_status} ({results.get('probability', 0)}% confidence)", ln=1)
        self.pdf.set_text_color(0, 0, 0)
        
        # eGFR info
        self.pdf.cell(0, 10, f"eGFR: {results.get('egfr', 'N/A')} mL/min/1.73m²", ln=1)
        self.pdf.cell(0, 10, f"CKD Stage: {results.get('stage', 'N/A')} - {results.get('stage_desc', '')}", ln=1)
        
        # Water intake
        self.pdf.ln(5)
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 10, 'Daily Fluid Recommendation:', ln=1)
        self.pdf.set_font('Arial', '', 12)
        self.pdf.cell(0, 10, f"{results.get('water_intake', 0)} mL ({round(results.get('water_intake', 0)/250, 1)}) cups", ln=1)
        
        if results.get('egfr', 0) < 30:
            self.pdf.set_text_color(255, 0, 0)
            self.pdf.cell(0, 10, "WARNING: Fluid restriction required - consult your nephrologist", ln=1)
            self.pdf.set_text_color(0, 0, 0)
        
        self.pdf.ln(10)

    def _add_diet_section(self, diet):
        self.pdf.set_font('Arial', 'B', 14)
        self.pdf.cell(0, 10, 'Dietary Recommendations', ln=1)
        
        # Recommended foods
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 10, 'Recommended Foods:', ln=1)
        self.pdf.set_font('Arial', '', 12)
        for food in diet.get('recommended', []):
            self.pdf.cell(0, 10, f"- {food.get('name', '')}", ln=1)
        
        # Foods to avoid
        self.pdf.ln(5)
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 10, 'Foods to Limit/Avoid:', ln=1)
        self.pdf.set_font('Arial', '', 12)
        for food in diet.get('avoid', []):
            self.pdf.cell(0, 10, f"- {food.get('name', '')}", ln=1)
        
        # General advice
        self.pdf.ln(5)
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 10, 'Additional Advice:', ln=1)
        self.pdf.set_font('Arial', '', 12)
        for advice in diet.get('advice', []):
            if advice.strip():
                self.pdf.cell(0, 10, f"- {advice}", ln=1)
        
        self.pdf.ln(10)

    def _add_footer(self):
        self.pdf.set_y(-15)
        self.pdf.set_font('Arial', 'I', 8)
        self.pdf.cell(0, 10, 'This report is not a substitute for professional medical advice', 0, 0, 'C')

# ==================== FLASK ROUTES ====================
@app.route('/', methods=['GET', 'POST'])
def home():
    """Main form page"""
    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            
            # Validate inputs
            required_fields = ['age', 'bp', 'sg', 'al', 'su', 'creatinine', 'gender', 'weight']
            for field in required_fields:
                if not form_data.get(field):
                    raise ValueError(f"Missing required field: {field}")
            
            # Convert types
            input_data = {
                'age': int(form_data['age']),
                'bp': int(form_data['bp']),
                'sg': float(form_data['sg']),
                'al': int(form_data['al']),
                'su': int(form_data['su']),
                'creatinine': float(form_data['creatinine']),
                'gender': form_data['gender'],
                'weight': float(form_data['weight']),
                'hemo': float(form_data.get('hemo', 12.5)),  # Default if not provided
                'race': form_data.get('race', 'other')
            }
            
            # Validate ranges
            if not (18 <= input_data['age'] <= 120):
                raise ValueError("Age must be between 18-120")
            if not (50 <= input_data['bp'] <= 200):
                raise ValueError("Blood pressure must be between 50-200 mmHg")
            
            # Clinical calculations
            egfr = ClinicalCalculator.calculate_egfr(
                input_data['creatinine'],
                input_data['age'],
                input_data['gender'],
                input_data['race']
            )
            stage, stage_desc = ClinicalCalculator.get_ckd_stage(egfr)
            
            # ML Prediction
            model_input = pd.DataFrame([{
                k: v for k, v in input_data.items() 
                if k in model_service.feature_names
            }])
            
            pred = model_service.model.predict(model_input)[0]
            proba = round(model_service.model.predict_proba(model_input)[0][pred] * 100, 2)
            
            # Feature importance
            feat_importance = dict(zip(
                model_service.feature_names,
                model_service.model.feature_importances_
            ))
            
            # Diet plan
            diet = get_diet_plan(
                pred,
                input_data['age'],
                input_data['bp'],
                egfr
            )
            
            # Store results in session
            session['kidney_results'] = {
    'prediction': int(pred),
    'probability': float(proba),
    'egfr': float(egfr),
    'stage': str(stage),
    'stage_desc': str(stage_desc),
    'water_intake': float(ClinicalCalculator.calculate_water_intake(egfr, float(input_data['weight']))),
    'feature_importance': {k: float(v) for k, v in feat_importance.items()},
    'creatinine_analysis': {
        'value': float(input_data['creatinine']),
        'normal_range': ClinicalCalculator.interpret_creatinine(
            input_data['creatinine'], input_data['age'], input_data['gender']
        )['normal_range'],
        'status': ClinicalCalculator.interpret_creatinine(
            input_data['creatinine'], input_data['age'], input_data['gender']
        )['status'],
        'age_group': ClinicalCalculator.interpret_creatinine(
            input_data['creatinine'], input_data['age'], input_data['gender']
        )['age_group']
    },
    'diet_plan': diet,
    'form_data': {k: (float(v) if isinstance(v, (np.float64, np.float32)) else int(v) if isinstance(v, (np.int64, np.int32)) else v)
                  for k, v in input_data.items()}
}

            
            return redirect(url_for('results'))
            
        except Exception as e:
            logging.error(f"Error processing form: {str(e)}")
            return render_template_string(HOME_TEMPLATE, error=str(e))
    
    return render_template_string(HOME_TEMPLATE)

@app.route('/results')
def results():
    """Results display page with visualizations"""
    result = session.get('kidney_results')
    if not result:
        return redirect(url_for('home'))
    
    try:
        # Generate visualizations
        visualization_paths = generate_visualizations(result)
        
        return render_template_string(
            RESULTS_TEMPLATE,
            result=result,
            visualizations=visualization_paths
        )
    except Exception as e:
        logging.error(f"Error generating results: {str(e)}")
        return redirect(url_for('home', error="Could not generate results"))

@app.route('/download-report')
def download_report():
    """Generate and download PDF report"""
    result = session.get('kidney_results')
    if not result:
        return redirect(url_for('home'))
    
    try:
        report_gen = PDFReportGenerator()
        pdf_buffer = report_gen.generate_report(
            result['form_data'],
            result,
            result['diet_plan']
        )
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"kidney_report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        logging.error(f"Error generating PDF: {str(e)}")
        return redirect(url_for('results', error="Could not generate report"))

# ==================== HELPER FUNCTIONS ====================
def get_diet_plan(prediction, age, bp, egfr):
    """Generate personalized diet recommendations"""
    if prediction == 1 or egfr < 60:
        return {
            "recommended": [
                {"name": "Cauliflower", "img": "cauliflower.png"},
                {"name": "Blueberries", "img": "blueberries.png"},
                {"name": "Fish", "img": "fish.png"},
                {"name": "Egg Whites", "img": "egg.png"},
                {"name": "Cabbage", "img": "cabbage.png"}
            ],
            "avoid": [
                {"name": "Processed Meats", "img": "sausage.png"},
                {"name": "Bananas", "img": "banana.png"},
                {"name": "Canned Foods", "img": "can.png"},
                {"name": "Dairy", "img": "milk.png"},
                {"name": "Potatoes", "img": "potato.png"}
            ],
            "advice": [
                f"Low-protein diet (0.6-0.8g/kg)" if egfr < 60 else "Moderate protein",
                "Limit potassium-rich foods",
                "Limit fluids to 1-1.5L/day" if egfr < 30 else "Maintain normal hydration",
                "Reduce sodium to <2g/day" if bp > 130 else "Moderate sodium intake",
                "Monitor phosphorus intake" if egfr < 45 else ""
            ]
        }
    else:
        return {
            "recommended": [
                {"name": "Vegetables", "img": "vegetables.png"},
                {"name": "Whole Grains", "img": "wheat.png"},
                {"name": "Lean Proteins", "img": "chicken.png"},
                {"name": "Berries", "img": "strawberry.png"},
                {"name": "Olive Oil", "img": "olive-oil.png"}
            ],
            "avoid": [
                {"name": "Excess Salt", "img": "salt.png"},
                {"name": "Processed Foods", "img": "fast-food.png"},
                {"name": "Sugary Drinks", "img": "soda.png"}
            ],
            "advice": [
                "Maintain balanced diet",
                "Regular kidney function tests",
                "Control blood pressure",
                "Stay hydrated"
            ]
        }

def generate_visualizations(result_data):
    """Generate matplotlib visualizations and return file paths"""
    visualization_paths = {}
    
    try:
        # Create static directory if not exists
        if not os.path.exists('static'):
            os.makedirs('static')
        
        # Visualization 1: eGFR Stage Comparison
        plt.figure(figsize=(10, 6))
        stages = ['G1', 'G2', 'G3a', 'G3b', 'G4', 'G5']
        colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336', '#9C27B0']
        stage_ranges = [90, 75, 52.5, 37.5, 22.5, 7.5]  # Midpoints for display
        
        bars = plt.bar(stages, stage_ranges, color=colors, alpha=0.6)
        plt.axhline(y=result_data['egfr'], color='black', linestyle='--', linewidth=2)
        
        # Add value annotations
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{stages[bars.index(bar)]}',
                    ha='center', va='bottom')
        
        plt.title('Your eGFR vs. CKD Stages', pad=20)
        plt.ylabel('eGFR (mL/min/1.73m²)')
        plt.xlabel('Chronic Kidney Disease Stages')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        egfr_path = os.path.join('static', 'egfr_plot.png')
        plt.savefig(egfr_path, bbox_inches='tight', dpi=100)
        plt.close()
        visualization_paths['egfr'] = egfr_path
        
        # Visualization 2: Feature Importance
        plt.figure(figsize=(10, 6))
        features = list(result_data['feature_importance'].keys())
        importance = list(result_data['feature_importance'].values())
        
        # Sort by importance
        sorted_idx = np.argsort(importance)
        features = [features[i] for i in sorted_idx]
        importance = [importance[i] for i in sorted_idx]
        
        bars = plt.barh(features, importance, color='#4361EE')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}',
                    va='center')
        
        plt.title('Factors Influencing Your Risk', pad=20)
        plt.xlabel('Relative Importance')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save plot
        feature_path = os.path.join('static', 'feature_plot.png')
        plt.savefig(feature_path, bbox_inches='tight', dpi=100)
        plt.close()
        visualization_paths['features'] = feature_path
        
        return visualization_paths
    
    except Exception as e:
        logging.error(f"Visualization generation failed: {str(e)}")
        return {}

# ==================== HTML TEMPLATES ====================
HOME_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kidney Health Assessment</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4361ee;
            --primary-dark: #3a56e8;
            --secondary: #3f37c9;
            --danger: #f72585;
            --success: #4cc9f0;
            --warning: #f8961e;
            --light: #f8f9fa;
            --dark: #212529;
            --gray: #6c757d;
            --light-gray: #e9ecef;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background-color: #f5f7fa;
            padding: 20px;
        }
        
       .container {
    max-width: 1200px;
    width: 100%;
    margin: 0 auto;
    padding: 40px 30px;
}

        
        .card {
    background: white;
    border-radius: 12px;
    box-shadow: 0 6px 30px rgba(0, 0, 0, 0.08);
    padding: 40px;
    margin-bottom: 40px;
}

        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(0,0,0,0.05);
        }
        
        .header img {
            height: 80px;
            margin-bottom: 15px;
        }
        
        .header h1 {
            font-size: 28px;
            font-weight: 600;
            color: var(--dark);
            margin: 0;
        }
        
        .subtitle {
            color: var(--gray);
            font-size: 16px;
            margin-top: 10px;
            font-weight: 400;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: var(--dark);
            font-size: 15px;
        }
        
        .form-control {
            width: 100%;
            padding: 12px 15px;
            border: 1px solid #ced4da;
            border-radius: 8px;
            font-size: 15px;
            font-family: 'Poppins', sans-serif;
            transition: all 0.3s ease;
        }
        
        .form-control:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.15);
        }
        
        select.form-control {
            appearance: none;
            background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
            background-repeat: no-repeat;
            background-position: right 12px center;
            background-size: 16px;
        }
        
        .btn {
            display: inline-block;
            background: var(--primary);
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            text-align: center;
            text-decoration: none;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .btn:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(67, 97, 238, 0.2);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .error-message {
            color: var(--danger);
            background-color: #ffebee;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 25px;
            border-left: 4px solid var(--danger);
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .card {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 24px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="header">
                <img src="https://img.icons8.com/color/96/000000/kidney.png" alt="Kidney Health">
                <h1>Kidney Health Analyzer</h1>
                <p class="subtitle">Assess your kidney disease risk with clinical accuracy</p>
            </div>
            
            {% if error %}
            <div class="error-message">
                <strong>Error:</strong> {{ error }}
            </div>
            {% endif %}
            
            <form method="POST" id="assessmentForm">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="age">Your Age</label>
                        <input type="number" class="form-control" id="age" name="age" min="18" max="120" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="weight">Weight (kg)</label>
                        <input type="number" class="form-control" id="weight" name="weight" min="30" max="300" step="0.1" required>
                    </div>
                </div>
                
                <div class="form-grid">
                    <div class="form-group">
                        <label for="bp">Blood Pressure (mm Hg)</label>
                        <input type="number" class="form-control" id="bp" name="bp" min="50" max="200" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="creatinine">Serum Creatinine (mg/dL)</label>
                        <input type="number" class="form-control" id="creatinine" name="creatinine" step="0.01" min="0.1" max="10" required>
                    </div>
                </div>
                
                <div class="form-grid">
                    <div class="form-group">
                        <label for="hemo">Hemoglobin (g/dL)</label>
                        <input type="number" class="form-control" id="hemo" name="hemo" step="0.1" min="5" max="20" value="12.5" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="gender">Gender</label>
                        <select class="form-control" id="gender" name="gender" required>
                            <option value="">Select</option>
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-grid">
                    <div class="form-group">
                        <label for="sg">Urine Specific Gravity</label>
                        <select class="form-control" id="sg" name="sg" required>
                            <option value="1.005">1.005 (Very dilute)</option>
                            <option value="1.010">1.010</option>
                            <option value="1.015">1.015</option>
                            <option value="1.020" selected>1.020 (Normal)</option>
                            <option value="1.025">1.025 (Concentrated)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="al">Albumin Level (0-5)</label>
                        <select class="form-control" id="al" name="al" required>
                            <option value="0">0 (Normal)</option>
                            <option value="1">1 (Trace)</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="5">5 (Heavy)</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-grid">
                    <div class="form-group">
                        <label for="su">Sugar Level (0-5)</label>
                        <select class="form-control" id="su" name="su" required>
                            <option value="0">0 (Normal)</option>
                            <option value="1">1 (Trace)</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="5">5 (Heavy)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="race">Ethnicity</label>
                        <select class="form-control" id="race" name="race">
                            <option value="other">Other</option>
                            <option value="black">Black/African American</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-group" style="margin-top: 15px;">
                    <button type="submit" class="btn">Analyze Kidney Health</button>
                </div>
            </form>
        </div>
    </div>

    <script>
        document.getElementById('assessmentForm').addEventListener('submit', function(e) {
            const btn = this.querySelector('button[type="submit"]');
            btn.disabled = true;
            btn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;">
                    <path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
                </svg>
                Analyzing...
            `;
        });
    </script>
</body>
</html>
'''

RESULTS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Kidney Health Results</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4361ee;
            --primary-dark: #3a56e8;
            --secondary: #3f37c9;
            --danger: #f72585;
            --success: #4cc9f0;
            --warning: #f8961e;
            --light: #f8f9fa;
            --dark: #212529;
            --gray: #6c757d;
            --light-gray: #e9ecef;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background-color: #f5f7fa;
            padding: 20px;
        }
        
        .container {
    max-width: 1200px;
    width: 100%;
    margin: 0 auto;
    padding: 40px 30px;
}

       .card {
    background: white;
    border-radius: 12px;
    box-shadow: 0 6px 30px rgba(0, 0, 0, 0.08);
    padding: 40px;
    margin-bottom: 40px;
}

        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(0,0,0,0.05);
        }
        
        .header img {
            height: 80px;
            margin-bottom: 15px;
        }
        
        .header h1 {
            font-size: 28px;
            font-weight: 600;
            color: var(--dark);
            margin: 0;
        }
        
        .result-section {
            margin-bottom: 30px;
        }
        
        .result-section h2 {
            font-size: 22px;
            margin-bottom: 15px;
            color: var(--dark);
            border-bottom: 2px solid var(--light-gray);
            padding-bottom: 8px;
        }
        
        .result-card {
            background: var(--light);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid var(--primary);
        }
        
        .risk-high {
            color: var(--danger);
            font-weight: 600;
        }
        
        .risk-low {
            color: var(--success);
            font-weight: 600;
        }
        
        .egfr-stage {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 600;
            margin-right: 10px;
        }
        
        .stage-G1 { background: #4CAF50; color: white; }
        .stage-G2 { background: #8BC34A; color: var(--dark); }
        .stage-G3a { background: #FFC107; color: var(--dark); }
        .stage-G3b { background: #FF9800; color: white; }
        .stage-G4 { background: #F44336; color: white; }
        .stage-G5 { background: #9C27B0; color: white; }
        
        .visualization-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 25px;
            margin: 30px 0;
        }
        
        .visualization {
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .visualization img {
            width: 100%;
            height: auto;
            border-radius: 6px;
        }
        
        .visualization h3 {
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 18px;
            color: var(--dark);
        }
        
        .diet-recommendations {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 25px 0;
        }
        
        @media (max-width: 768px) {
            .diet-recommendations {
                grid-template-columns: 1fr;
            }
        }
        
        .diet-column {
            background: var(--light);
            border-radius: 8px;
            padding: 20px;
        }
        
        .diet-column h3 {
            font-size: 18px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }
        
        .diet-column h3 svg {
            margin-right: 8px;
        }
        
        .diet-list {
            list-style: none;
        }
        
        .diet-list li {
            padding: 8px 0;
            border-bottom: 1px dashed #ddd;
            display: flex;
            align-items: center;
        }
        
        .diet-list li:last-child {
            border-bottom: none;
        }
        
        .diet-list li svg {
            margin-right: 8px;
            color: var(--gray);
        }
        
        .action-buttons {
            display: flex;
            gap: 15px;
            margin-top: 30px;
        }
        
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 14px 28px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.3s ease;
            flex: 1;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(67, 97, 238, 0.2);
        }
        
        .btn-secondary {
            background: white;
            color: var(--primary);
            border: 1px solid var(--primary);
        }
        
        .btn-secondary:hover {
            background: var(--light);
            transform: translateY(-2px);
        }
        
        .btn svg {
            margin-right: 8px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .card {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 24px;
            }
            
            .action-buttons {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="header">
                <img src="https://img.icons8.com/color/96/000000/kidney.png" alt="Kidney Health">
                <h1>Your Kidney Health Results</h1>
                <p class="subtitle">Comprehensive analysis based on your inputs</p>
            </div>
            
            <div class="result-section">
                <h2>Diagnosis Summary</h2>
                <div class="result-card">
                    <p style="margin-bottom: 15px;">
                        <strong>Risk Assessment:</strong> 
                        <span class="{% if result.prediction %}risk-high{% else %}risk-low{% endif %}">
                            {% if result.prediction %}High Risk ({{ result.probability }}% confidence){% else %}Low Risk ({{ result.probability }}% confidence){% endif %}
                        </span>
                    </p>
                    
                    <p style="margin-bottom: 15px;">
                        <strong>Kidney Function:</strong> 
                        <span class="egfr-stage stage-{{ result.stage }}">
                            Stage {{ result.stage }}
                        </span>
                        {{ result.stage_desc }}
                    </p>
                    
                    <p>
                        <strong>eGFR:</strong> {{ result.egfr }} mL/min/1.73m²
                        ({{ "Normal" if result.egfr >= 90 else "Mild reduction" if result.egfr >= 60 else "Moderate reduction" if result.egfr >= 30 else "Severe reduction" }})
                    </p>
                </div>
            </div>
            
            <div class="visualization-grid">
                <div class="visualization">
                    <h3>Your Kidney Function Compared to CKD Stages</h3>
                    <img src="{{ visualizations.egfr }}" alt="eGFR Stage Comparison">
                </div>
                
                <div class="visualization">
                    <h3>Key Factors Influencing Your Risk</h3>
                    <img src="{{ visualizations.features }}" alt="Feature Importance">
                </div>
            </div>
            
            <div class="result-section">
                <h2>Clinical Details</h2>
                <div class="result-card">
                    <p style="margin-bottom: 10px;">
                        <strong>Creatinine Level:</strong> 
                        {{ result.creatinine_analysis.value }} mg/dL
                        ({{ result.creatinine_analysis.status }})
                    </p>
                    <p style="margin-bottom: 10px;">
                        <strong>Normal Range for:</strong> 
                        {{ result.creatinine_analysis.age_group }}: {{ result.creatinine_analysis.normal_range }}
                    </p>
                    <p>
                        <strong>Daily Fluid Recommendation:</strong> 
                        {{ result.water_intake }} mL (about {{ (result.water_intake / 250)|round }} cups)
                    </p>
                    
                    {% if result.egfr < 30 %}
                    <p style="margin-top: 15px; color: var(--danger); font-weight: 500;">
                        ⚠️ Important: Fluid restriction recommended - please consult your doctor
                    </p>
                    {% endif %}
                </div>
            </div>
            
            <div class="result-section">
                <h2>Dietary Recommendations</h2>
                <div class="diet-recommendations">
                    <div class="diet-column">
                        <h3>
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                <polyline points="14 2 14 8 20 8"></polyline>
                                <line x1="16" y1="13" x2="8" y2="13"></line>
                                <line x1="16" y1="17" x2="8" y2="17"></line>
                                <polyline points="10 9 9 9 8 9"></polyline>
                            </svg>
                            Recommended Foods
                        </h3>
                        <ul class="diet-list">
                            {% for food in result.diet_plan.recommended %}
                            <li>
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                                    <polyline points="22 4 12 14.01 9 11.01"></polyline>
                                </svg>
                                {{ food.name }}
                            </li>
                            {% endfor %}
                        </ul>
                    </div>
                    
                    <div class="diet-column">
                        <h3>
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                                <line x1="12" y1="9" x2="12" y2="13"></line>
                                <line x1="12" y1="17" x2="12.01" y2="17"></line>
                            </svg>
                            Foods to Limit
                        </h3>
                        <ul class="diet-list">
                            {% for food in result.diet_plan.avoid %}
                            <li>
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <circle cx="12" cy="12" r="10"></circle>
                                    <line x1="4.93" y1="4.93" x2="19.07" y2="19.07"></line>
                                </svg>
                                {{ food.name }}
                            </li>
                            {% endfor %}
                        </ul>
                    </div>
                </div>
                
                <div class="result-card">
                    <h3 style="margin-top: 0; margin-bottom: 15px;">Medical Advice</h3>
                    <ul style="list-style: none;">
                        {% for advice in result.diet_plan.advice %}
                        {% if advice.strip() %}
                        <li style="padding: 8px 0; border-bottom: 1px dashed #eee; display: flex; align-items: flex-start;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0; margin-right: 8px; margin-top: 2px;">
                                <circle cx="12" cy="12" r="10"></circle>
                                <path d="M12 16v-4"></path>
                                <path d="M12 8h.01"></path>
                            </svg>
                            {{ advice }}
                        </li>
                        {% endif %}
                        {% endfor %}
                    </ul>
                </div>
            </div>
            
            <div class="action-buttons">
                <a href="/download-report" class="btn btn-primary">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    Download Full Report
                </a>
                <a href="/" class="btn btn-secondary">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                        <polyline points="9 22 9 12 15 12 15 22"></polyline>
                    </svg>
                    New Analysis
                </a>
            </div>
        </div>
    </div>
</body>
</html>
'''

# ==================== APPLICATION START ====================
if __name__ == '__main__':
    # Create necessary directories
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)