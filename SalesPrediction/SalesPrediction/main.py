import os
import json
import smtplib
import random
import pandas as pd
from datetime import datetime
from email.mime.text import MIMEText
import firebase_admin
from io import BytesIO
import uuid
from flask import (Flask, render_template, request, redirect, url_for, session,
                   flash, Response, make_response)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from firebase_admin import credentials, firestore, initialize_app
from weasyprint import HTML
import pytz
import numpy as np

# Import prediction models
from models import run_xgboost_prediction, run_arima_prediction

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'Fypxiexieni888_a_very_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------------ Firebase Init ------------------------
db = None
try:
    firebase_secret_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    if firebase_secret_json:
        service_account_info = json.loads(firebase_secret_json)
        cred = credentials.Certificate(service_account_info)
        if not firebase_admin._apps:
            initialize_app(cred)
        db = firestore.client()
    else:
        print("Warning: 'FIREBASE_SERVICE_ACCOUNT_JSON' secret not found.")
except Exception as e:
    print(f"Warning: Firebase initialization from secret failed: {e}")

# ------------------------ Helper Functions ------------------------
def check_db():
    if not db:
        flash("Database connection is not available.", "error")
        return False
    return True

# ------------------------ Authentication Routes ------------------------
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if not check_db(): return render_template('signup.html')
    if 'email' in session: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        if not name or not email:
            flash("Name and Email are required.", "error")
            return render_template('signup.html')
        user_ref = db.collection('users').document(email)
        if user_ref.get().exists:
            flash("User with this email already exists. Please log in.", "error")
            return redirect(url_for('login'))
        code = str(random.randint(100000, 999999))
        session['verify_code'] = code
        session['temp_email'] = email
        session['temp_name'] = name
        flash(f"Verification code sent (for testing: {code})", "info")
        return redirect(url_for('verify'))
    return render_template('signup.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if 'temp_email' not in session: return redirect(url_for('signup'))
    if request.method == 'POST':
        if request.form.get('code') == session.get('verify_code'):
            session.pop('verify_code', None)
            return redirect(url_for('set_password'))
        else:
            flash("Invalid verification code.", "error")
    return render_template('verify.html')

@app.route('/set-password', methods=['GET', 'POST'])
def set_password():
    if not check_db() or 'temp_email' not in session: return redirect(url_for('signup'))
    if request.method == 'POST':
        pw = request.form.get('password')
        if not pw or pw != request.form.get('confirm'):
            flash("Passwords do not match or are empty.", "error")
            return render_template('set_password.html')
        hashed = generate_password_hash(pw)
        email, name = session['temp_email'], session['temp_name']
        db.collection('users').document(email).set({'name': name, 'password': hashed, 'role': None})
        session['user'], session['email'] = name, email
        session.pop('temp_name', None); session.pop('temp_email', None)
        flash("Registration successful! Please complete your profile.", "success")
        return redirect(url_for('check_profile'))
    return render_template('set_password.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if not check_db(): return render_template('login.html')
    if 'email' in session: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        user_doc = db.collection('users').document(email).get()
        if user_doc.exists:
            user = user_doc.to_dict()
            if user and check_password_hash(user.get('password', ''), password):
                session['user'], session['email'] = user.get('name'), email
                return redirect(url_for('check_profile'))
        flash("Invalid email or password.", "error")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

@app.route('/check-profile')
def check_profile():
    if not check_db() or 'email' not in session: return redirect(url_for('login'))
    user_doc = db.collection('users').document(session['email']).get()
    if user_doc.exists and user_doc.to_dict().get('role'):
        return redirect(url_for('dashboard'))
    return redirect(url_for('complete_profile'))

@app.route('/complete-profile', methods=['GET', 'POST'])
def complete_profile():
    if not check_db() or 'email' not in session: return redirect(url_for('login'))
    if request.method == 'POST':
        profile_data = {
            'role': request.form.get('role'), 'phone': request.form.get('phone'),
            'gender': request.form.get('gender'), 'nationality': request.form.get('nationality'),
            'birthdate': request.form.get('birthdate'), 'address': request.form.get('address'),
            'age': int(request.form.get('age', 0))
        }
        db.collection('users').document(session['email']).update(profile_data)
        flash("Profile completed successfully!", "success")
        return redirect(url_for('dashboard'))
    return render_template('complete_profile.html')

@app.route('/dashboard')
def dashboard():
    if not check_db() or 'user' not in session: return redirect(url_for('login'))
    summary = {
        'total_predictions': 0, 'best_model': 'N/A',
        'most_forecasted_item': 'N/A', 'last_prediction_name': 'N/A'
    }
    try:
        predictions_ref = db.collection('predictions').where('user_email', '==', session['email']).stream()
        all_predictions = [doc.to_dict() for doc in predictions_ref]
        if all_predictions:
            summary['total_predictions'] = len(all_predictions)
            model_maes = {'XGBoost': [], 'ARIMA': []}
            for p in all_predictions:
                if p.get('xgboost_result', {}).get('mae', float('inf')) != float('inf'):
                    model_maes['XGBoost'].append(p['xgboost_result']['mae'])
                if p.get('arima_result', {}).get('mae', float('inf')) != float('inf'):
                    model_maes['ARIMA'].append(p['arima_result']['mae'])
            avg_mae_xgb = sum(model_maes['XGBoost']) / len(model_maes['XGBoost']) if model_maes['XGBoost'] else float('inf')
            avg_mae_arima = sum(model_maes['ARIMA']) / len(model_maes['ARIMA']) if model_maes['ARIMA'] else float('inf')
            if avg_mae_xgb != float('inf') or avg_mae_arima != float('inf'):
                 summary['best_model'] = 'XGBoost' if avg_mae_xgb < avg_mae_arima else 'ARIMA'
            items = [p.get('item_filtered') for p in all_predictions if p.get('item_filtered') != 'Overall']
            if items:
                summary['most_forecasted_item'] = max(set(items), key=items.count)
            last_pred = sorted(all_predictions, key=lambda x: x.get('timestamp'), reverse=True)[0]
            summary['last_prediction_name'] = last_pred.get('name', 'N/A')
    except Exception as e:
        flash(f"Could not load dashboard summary: {e}", "warning")
    user_doc = db.collection('users').document(session['email']).get()
    role = user_doc.to_dict().get('role') if user_doc.exists else 'N/A'
    return render_template('dashboard.html', name=session.get('user'), role=role, summary=summary)

@app.route('/account', methods=['GET', 'POST'])
def account():
    if not check_db() or 'email' not in session: return redirect(url_for('login'))
    user_ref = db.collection('users').document(session['email'])
    if request.method == 'POST':
        update_data = {
            'phone': request.form.get('phone'), 'address': request.form.get('address'),
            'nationality': request.form.get('nationality'), 'gender': request.form.get('gender'),
            'age': int(request.form.get('age', 0)), 'birthdate': request.form.get('birthdate')
        }
        user_ref.update({k: v for k, v in update_data.items() if v is not None})
        flash("Account updated successfully!", "success")
        return redirect(url_for('account'))
    user = user_ref.get().to_dict()
    return render_template("account.html", user=user)

# ------------------------ Prediction Routes ------------------------
@app.route('/predict')
def predict():
    if 'email' not in session: return redirect(url_for('login'))
    if 'temp_filepath' in session and os.path.exists(session['temp_filepath']):
        os.remove(session['temp_filepath'])
    for key in list(session.keys()):
        if key.startswith('temp_') or key.startswith('mapped_') or key == 'prediction_to_save':
            session.pop(key)
    return render_template('predict.html')

@app.route('/upload_and_customize', methods=['POST'])
def upload_and_customize():
    if 'email' not in session: return redirect(url_for('login'))
    if 'file' not in request.files or not request.files['file'].filename:
        flash('No file selected.', 'error')
        return redirect(url_for('predict'))
    file = request.files['file']
    original_filename = secure_filename(file.filename)
    if not (original_filename.lower().endswith('.csv') or original_filename.lower().endswith('.xlsx')):
        flash('Invalid file type.', 'error')
        return redirect(url_for('predict'))
    try:
        unique_id = uuid.uuid4().hex
        _, extension = os.path.splitext(original_filename)
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}{extension}")
        file.save(temp_filepath)
        session['temp_filepath'] = temp_filepath
        session['original_filename'] = original_filename
        return redirect(url_for('customize_prediction'))
    except Exception as e:
        flash(f"Error saving file: {e}", "error")
        return redirect(url_for('predict'))

@app.route('/customize_prediction', methods=['GET', 'POST'])
def customize_prediction():
    if 'email' not in session or 'temp_filepath' not in session: return redirect(url_for('predict'))
    temp_filepath = session.get('temp_filepath')
    items = []
    try:
        df_for_items = pd.read_csv(temp_filepath, low_memory=False) if temp_filepath.lower().endswith('.csv') else pd.read_excel(temp_filepath)
        item_col_actual = next((c for c in df_for_items.columns if c.lower() in ['item', 'productid']), None)
        if item_col_actual:
            item_counts = df_for_items[item_col_actual].value_counts()
            min_records = 20 
            valid_items = item_counts[item_counts >= min_records].index.tolist()
            items = sorted(valid_items)
            num_filtered = len(item_counts) - len(valid_items)
            if num_filtered > 0:
                flash(f"{num_filtered} products with insufficient data were hidden from the list.", "info")
    except Exception as e:
        flash(f"Could not read items from file: {e}", "warning")

    if request.method == 'POST':
        try:
            df = pd.read_csv(temp_filepath, low_memory=False) if temp_filepath.lower().endswith('.csv') else pd.read_excel(temp_filepath)
            
            date_col = next((c for c in df.columns if c.lower() == 'date'), None)
            amount_col = next((c for c in df.columns if c.lower() == 'amount'), None)
            quantity_col = next((c for c in df.columns if c.lower() == 'quantity'), None)
            item_col = next((c for c in df.columns if c.lower() in ['item', 'productid']), None)

            if not date_col or not amount_col:
                flash("Dataset must contain 'Date' and 'Amount' columns.", "error")
                return render_template('customize_prediction.html', items=items)

            sales_unit = request.form.get('sales_target', 'Amount')
            if sales_unit == 'Quantity' and not quantity_col:
                flash("To predict Quantity, a 'Quantity' column is required.", "error")
                return render_template('customize_prediction.html', items=items)
            
            sales_col = quantity_col if sales_unit == 'Quantity' else amount_col
            df.rename(columns={date_col: 'date', sales_col: 'sales'}, inplace=True)
            df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
            
            selected_item = request.form.get('item_filter')
            time_freq = request.form.get('time_frequency', 'W')
            forecast_horizon = int(request.form.get('forecast_horizon', 4))
            
            filtered_item_name = 'Overall'
            if item_col and selected_item and selected_item != 'overall':
                df = df[df[item_col] == selected_item]
                filtered_item_name = selected_item
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True).dt.normalize()
            df.dropna(subset=['date', 'sales'], inplace=True)
            if df['sales'].min() < 0:
                flash("Note: Negative sales (returns) were automatically set to zero for accuracy.", "warning")
                df['sales'] = df['sales'].clip(lower=0)
            
            df = df.set_index('date')['sales'].resample(time_freq).sum().reset_index()
            df = df.sort_values(by='date', ascending=True)

            q1 = df['sales'].quantile(0.25)
            q3 = df['sales'].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            outliers_count = (df['sales'] > upper_bound).sum()
            if outliers_count > 0:
                df = df[df['sales'] <= upper_bound]
                flash(f"Note: {outliers_count} extreme outlier(s) were removed to improve forecast accuracy.", "info")

            xgboost_results = run_xgboost_prediction(df.copy(), forecast_horizon, time_freq)
            arima_results = run_arima_prediction(df.copy(), forecast_horizon, time_freq)
            
            # --- MODIFIED: Handle ARIMA model skipping ---
            if arima_results.get('status') == 'skipped':
                recommended_model = 'XGBoost'
            else:
                recommended_model = 'XGBoost' if xgboost_results.get('mae', float('inf')) < arima_results.get('mae', float('inf')) else 'ARIMA'

            xgboost_insights = calculate_insights(xgboost_results, time_freq)
            arima_insights = calculate_insights(arima_results, time_freq)

            session['prediction_to_save'] = {
                'input_details': session.get('original_filename'),
                'xgboost_result': {**xgboost_results, 'insights': xgboost_insights},
                'arima_result': {**arima_results, 'insights': arima_insights},
                'sales_unit': sales_unit, 'forecast_horizon': forecast_horizon,
                'item_filtered': filtered_item_name, 'time_frequency': time_freq,
                'recommended_model': recommended_model,
            }
            return redirect(url_for('view_prediction'))
        except Exception as e:
            flash(f"An error occurred during prediction: {e}", "error")
            return redirect(url_for('predict'))
        finally:
            if 'temp_filepath' in session and os.path.exists(session['temp_filepath']): 
                os.remove(session['temp_filepath'])
            for key in list(session.keys()):
                if key.startswith('temp_'):
                    session.pop(key)
    
    return render_template('customize_prediction.html', items=items)


def calculate_insights(model_results, time_freq):
    insights = {
        'total_forecast': 0, 'peak_period': 'N/A',
        'growth_trend': 0, 'stability': 0
    }
    try:
        plot_data = model_results.get('plot_data', {})
        forecast_values = [v for v in plot_data.get('forecast_values', []) if v is not None]
        if not forecast_values: return insights
        insights['total_forecast'] = sum(forecast_values)
        peak_value = max(forecast_values)
        peak_index = forecast_values.index(peak_value)
        peak_date_str = plot_data['forecast_dates'][peak_index]
        peak_date = datetime.strptime(peak_date_str, '%Y-%m-%d')
        if time_freq == 'W':
            insights['peak_period'] = f"Week of {peak_date.strftime('%b %d')}"
        else:
            insights['peak_period'] = peak_date.strftime('%B %Y')
        historical_values = [v for v in plot_data.get('historical_values', []) if v is not None]
        if historical_values:
            last_historical_avg = np.mean(historical_values[-4:])
            forecast_avg = np.mean(forecast_values)
            if last_historical_avg > 0:
                insights['growth_trend'] = ((forecast_avg - last_historical_avg) / last_historical_avg) * 100
        lower_ci = [v for v in plot_data.get('lower_ci', []) if v is not None]
        upper_ci = [v for v in plot_data.get('upper_ci', []) if v is not None]
        if lower_ci and upper_ci and np.mean(forecast_values) > 0:
            avg_range = np.mean(np.array(upper_ci) - np.array(lower_ci))
            avg_forecast = np.mean(forecast_values)
            volatility_percent = (avg_range / avg_forecast) * 100
            insights['stability'] = max(0, 100 - volatility_percent)
    except Exception as e:
        print(f"Error calculating insights: {e}")
    return insights

@app.route('/view_prediction')
def view_prediction():
    if 'email' not in session: return redirect(url_for('login'))
    prediction_data = session.get('prediction_to_save')
    if not prediction_data:
        flash("No active prediction to view.", "info")
        return redirect(url_for('predict'))
    return render_template('prediction_result.html', prediction=prediction_data, is_saved=False)

@app.route('/save_prediction', methods=['POST'])
def save_prediction():
    if not check_db() or 'email' not in session: return redirect(url_for('login'))
    prediction_to_save = session.get('prediction_to_save')
    prediction_name = request.form.get('prediction_name', '').strip()
    if not prediction_to_save:
        flash("No prediction data to save.", "error")
        return redirect(url_for('predict'))
    if not prediction_name:
        flash("A name is required to save the prediction.", "error")
        return render_template('prediction_result.html', prediction=prediction_to_save, is_saved=False)
    try:
        malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
        now_utc = datetime.utcnow()
        now_myt = pytz.utc.localize(now_utc).astimezone(malaysia_tz)
        prediction_to_save['name'] = prediction_name
        prediction_to_save['user_email'] = session['email']
        prediction_to_save['timestamp'] = now_utc
        prediction_to_save['date'] = now_myt.strftime("%d %b %Y, %I:%M %p")
        db.collection('predictions').add(prediction_to_save)
        session.pop('prediction_to_save', None)
        flash("Prediction saved successfully!", "success")
        return redirect(url_for('history'))
    except Exception as e:
        flash(f"Could not save prediction: {e}", "error")
        return redirect(url_for('predict'))

@app.route('/history')
def history():
    if not check_db() or 'email' not in session: return redirect(url_for('login'))
    search_query = request.args.get('search', '').strip().lower()
    sort_order = request.args.get('sort', 'desc')
    try:
        base_query = db.collection('predictions').where('user_email', '==', session['email'])
        direction = firestore.Query.ASCENDING if sort_order == 'asc' else firestore.Query.DESCENDING
        docs_query = base_query.order_by('timestamp', direction=direction).stream()
        prediction_history = []
        for doc in docs_query:
            item = doc.to_dict()
            item['id'] = doc.id
            if not search_query or search_query in item.get('name', '').lower():
                 prediction_history.append(item)
    except Exception as e:
        prediction_history = []
        flash(f"Could not retrieve history: {e}", "error")
    return render_template('history.html', history=prediction_history, search_query=search_query, sort_order=sort_order)

@app.route('/view_history/<string:doc_id>')
def view_history(doc_id):
    if not check_db() or 'email' not in session: return redirect(url_for('login'))
    try:
        prediction_doc = db.collection('predictions').document(doc_id).get()
        if not prediction_doc.exists:
            flash("Prediction not found.", "error")
            return redirect(url_for('history'))
        prediction = prediction_doc.to_dict()
        if prediction.get('user_email') != session['email']:
            flash("You are not authorized to view this prediction.", "error")
            return redirect(url_for('history'))
        return render_template('prediction_result.html', prediction=prediction, is_saved=True, doc_id=doc_id)
    except Exception as e:
        flash(f"Could not load prediction: {e}", "error")
        return redirect(url_for('history'))

@app.route('/delete_prediction/<string:doc_id>', methods=['POST'])
def delete_prediction(doc_id):
    if not check_db() or 'email' not in session: return redirect(url_for('login'))
    db.collection('predictions').document(doc_id).delete()
    flash("Prediction deleted successfully!", "success")
    return redirect(url_for('history'))

@app.route('/inventory_recommendation', methods=['POST'])
def inventory_recommendation():
    if 'email' not in session: return {'error': 'Unauthorized'}, 401
    data = request.json
    current_stock = float(data.get('current_stock', 0))
    predicted_sales = data.get('prediction_data')
    if not predicted_sales or current_stock <= 0: return {'recommendation': 'Invalid input. Please enter a valid stock level.'}, 400
    predicted_sales = [sale for sale in predicted_sales if sale is not None]
    if not predicted_sales: return {'recommendation': 'Not enough future sales data for a recommendation.'}
    avg_period_sale = sum(predicted_sales) / len(predicted_sales)
    if avg_period_sale <= 0: return {'recommendation': "Predicted sales are zero or negative."}
    periods_of_stock = round(current_stock / avg_period_sale)
    return {'recommendation': f"You have enough stock for approx. {periods_of_stock} future periods (weeks/months)."}

@app.route('/export_csv/<string:doc_id>')
def export_csv(doc_id):
    if not check_db() or 'email' not in session: return redirect(url_for('login'))
    try:
        doc = db.collection('predictions').document(doc_id).get()
        if not doc.exists or doc.to_dict().get('user_email') != session['email']:
            flash("Prediction not found or unauthorized.", "error")
            return redirect(url_for('history'))
        pred = doc.to_dict()
        model_key = pred.get('recommended_model', 'XGBoost').lower() + '_result'
        plot_data = pred.get(model_key, {}).get('plot_data', {})
        if not plot_data or not plot_data.get('forecast_dates'):
             flash("Cannot export CSV, prediction data is incomplete.", "error")
             return redirect(url_for('view_history', doc_id=doc_id))

        df = pd.DataFrame.from_dict(plot_data, orient='index').transpose()
        df.rename(columns={'forecast_dates': 'date', 'forecast_values': 'predicted_sales'}, inplace=True)

        output = BytesIO()
        df[['date', 'predicted_sales']].to_csv(output, index=False)
        output.seek(0)
        return Response(output, mimetype="text/csv", headers={"Content-Disposition": f"attachment;filename=prediction_{doc_id}.csv"})
    except Exception as e:
        flash(f"Error exporting CSV: {e}", "error")
        return redirect(url_for('history'))

@app.route('/export_pdf/<string:doc_id>', methods=['POST'])
def export_pdf(doc_id):
    if not check_db() or 'email' not in session: return redirect(url_for('login'))
    try:
        doc = db.collection('predictions').document(doc_id).get()
        if not doc.exists or doc.to_dict().get('user_email') != session['email']:
            flash("Prediction not found or unauthorized.", "error")
            return redirect(url_for('history'))
        prediction = doc.to_dict()
        chart_image = request.form.get('chart_image')
        html_out = render_template('report_template.html', prediction=prediction, chart_image=chart_image, now=datetime.now())
        pdf = HTML(string=html_out).write_pdf()
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=report_{doc_id}.pdf'
        return response
    except Exception as e:
        flash(f"Error exporting PDF: {e}", "error")
        return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

