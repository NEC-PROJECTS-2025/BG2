from flask import Flask, request, render_template, redirect, url_for,session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid
import random
from datetime import datetime as today_date
import datetime
from database import get_patient_collection,get_appointment_collection,get_all_appointments # Import MongoDB collection
from flask import send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
app = Flask(__name__, static_folder='uploads')
app.secret_key = 'your_secret_key'  # Required for session handling
ADMIN_CREDENTIALS = {
    'username': 'admin',
    'password': 'admin123'
}
# Load the trained model
cxr_identifier = load_model('models/chestxray_classifier.h5')  # X-ray vs. Non-X-ray
cxr_classifier = load_model('models/model.h5')  # COVID, Normal, Pneumonia classifier


# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
@app.route('/appointment', methods=['GET'])
def appointment():
    return render_template('book_appointment.html')  # Ensure the template exists




@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    # Retrieve form data
    name = request.form['name']
    age = request.form['age']
    phone = request.form['phone']
    email = request.form['email']
    address = request.form['address']
    time = request.form['time']
    date = request.form['date']

    # Validate input
    if not name or not age or not phone or not email or not address or not time or not date:
        return "All fields are required", 400

    # Create appointment document
    appointment = {
        "name": name,
        "age": int(age),
        "phone": phone,
        "email": email,
        "address": address,
        "time": time,
        "date": datetime.datetime.strptime(date, '%Y-%m-%d'),
        "created_at": datetime.datetime.now()
    }

    # Save to database
    appointment_collection = get_appointment_collection()
    appointment_collection.insert_one(appointment)

    return f"Appointment booked successfully for {name} on {date} at {time}!"






@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Retrieve form data
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        smoking_history = request.form['smoking_history']
        pre_existing_conditions = request.form.getlist('pre_existing_conditions')  # List of conditions
        symptoms = request.form.getlist('symptoms')  # List of symptoms
        covid_exposure = request.form['covid_exposure']
        cxr_date = request.form['cxr_date']
        cxr_type = request.form['cxr_type']
        cxr_pic = request.files['cxr_pic']

        # Save the CXR image
        file_path = os.path.join(UPLOAD_FOLDER, cxr_pic.filename)
        cxr_pic.save(file_path)

        # Generate a unique short numeric patient ID
        patient_collection = get_patient_collection()
        while True:
            patient_id = str(random.randint(1000, 999999))  # 6-digit numeric ID
            if not patient_collection.find_one({'patient_id': patient_id}):  # Ensure uniqueness
                break

        # Save patient details into MongoDB
        patient_data = {
            'patient_id': patient_id,
            'name': name,
            'age': age,
            'gender': gender,
            'smoking_history': smoking_history,
            'pre_existing_conditions': pre_existing_conditions,
            'symptoms': symptoms,
            'covid_exposure': covid_exposure,
            'cxr_date': cxr_date,
            'cxr_type': cxr_type,
            'cxr_image': cxr_pic.filename
        }
        patient_collection.insert_one(patient_data)
        patient = patient_collection.find_one({'patient_id': patient_id})
        if not patient:
            return render_template('error.html', message="Patient not found.")

    # Load and check the CXR image
        file_path = os.path.join(UPLOAD_FOLDER, patient['cxr_image'])
        img_array = preprocess_image(file_path)
        # Redirect to the success page with patient ID
        is_xray = cxr_identifier.predict(img_array)[0][0]
        if is_xray > 0.5:
            return render_template('error.html', message="❌ Invalid Image! Please upload a Chest X-ray.", image_path=file_path)
        else:
            return render_template(
        'registration_success.html', 
        patient_id=patient_id,
        message="✅ Content Uploaded Successfully!",
        image_path=file_path
    )


    
    return render_template('register.html')



@app.route('/registration_success/<patient_id>', methods=['GET'])
def registration_success(patient_id):
    return render_template('registration_success.html', patient_id=patient_id)

@app.route('/view_report/<patient_id>', methods=['GET'])
def view_report(patient_id):
    # Retrieve patient details from MongoDB
    patient_collection = get_patient_collection()
    patient = patient_collection.find_one({'patient_id': patient_id})
    
    if not patient:
        return render_template('error.html', message="Patient not found.")

    # Load and check the CXR image
    file_path = os.path.join(UPLOAD_FOLDER, patient['cxr_image'])
    img_array = preprocess_image(file_path)

    # Step 1: Validate if it's a Chest X-ray
    is_xray = cxr_identifier.predict(img_array)[0][0]
    if is_xray > 0.5:
        return render_template('error.html', message="❌ Invalid Image! Please upload a Chest X-ray.")

    # Step 2: Predict disease if it's a valid CXR
    prediction = cxr_classifier.predict(img_array)
    covid_prob, normal_prob, pneumonia_prob = prediction[0]
    
    # Define class names
    classes = ['Covid', 'Normal', 'Pneumonia']
    predicted_disease = classes[np.argmax([covid_prob, normal_prob, pneumonia_prob])]

    # Calculate risk level
    risk_factors = sum([
        int(patient.get('age', 0)) > 60,
        patient.get('smoking_history') == 'Smoker',
        'COPD' in patient.get('pre_existing_conditions', []),
        'Asthma' in patient.get('pre_existing_conditions', []),
        'Heart Disease' in patient.get('pre_existing_conditions', []),
        'Diabetes' in patient.get('pre_existing_conditions', []),
        'Hypertension' in patient.get('pre_existing_conditions', []),
        patient.get('recent_covid_exposure') == 'Yes',
        len(patient.get('symptoms', [])) >= 3
    ])

    risk_level = "High" if risk_factors >= 6 else "Moderate" if risk_factors >= 3 else "Low"

    # Define precautions based on risk level
    precautions = {
        "High": [
            "Immediate medical attention is required.",
            "Avoid public places and stay isolated.",
            "Ensure regular monitoring of symptoms.",
            "Follow up with a healthcare provider urgently."
        ],
        "Moderate": [
            "Limit physical activity and rest adequately.",
            "Monitor symptoms closely for any worsening.",
            "Consult with a healthcare provider as needed."
        ],
        "Low": [
            "Maintain a healthy lifestyle and diet.",
            "Avoid smoking and exposure to pollutants.",
            "Regular check-ups with your doctor are recommended."
        ]
    }[risk_level]

    # Generate visualization (Predicted CXR image)
    visualization_path = os.path.join(UPLOAD_FOLDER, f"visualization_{patient_id}.png")
    plt.figure(figsize=(6, 4))
    plt.imshow(img_array[0])
    plt.title(f"Predicted: {predicted_disease}")
    plt.axis('off')
    plt.savefig(visualization_path)
    plt.close()

    # Generate a pie chart for disease distribution
    pie_chart_path = os.path.join(UPLOAD_FOLDER, f"pie_chart_{patient_id}.png")
    plt.figure(figsize=(6, 4))
    plt.pie([covid_prob, normal_prob, pneumonia_prob], labels=classes, autopct='%1.1f%%', colors=['red', 'green', 'blue'])
    plt.title('Disease Prediction Distribution')
    plt.savefig(pie_chart_path)
    plt.close()

    # Generate a bar plot for confidence levels
    confidence_plot_path = os.path.join(UPLOAD_FOLDER, f"confidence_plot_{patient_id}.png")
    plt.figure(figsize=(6, 4))
    plt.bar(classes, [covid_prob, normal_prob, pneumonia_prob], color=['red', 'green', 'blue'])
    plt.title('Disease Confidence Levels')
    plt.xlabel('Disease')
    plt.ylabel('Confidence')
    plt.ylim(0, 1)
    plt.savefig(confidence_plot_path)
    plt.close()

    # Return the report page
    return render_template(
        'view_report.html',
        patient=patient,
        predicted_disease=predicted_disease,
        risk_level=risk_level,
        precautions=precautions,
        image_path=file_path,
        visualization_path=visualization_path,
        pie_chart_path=f"uploads/{os.path.basename(pie_chart_path)}",
        confidence_plot_path=f"uploads/{os.path.basename(confidence_plot_path)}"
    )

@app.route('/update_report', methods=['GET', 'POST'])
def update_report():
    # Placeholder implementation
    if request.method == 'POST':
        # Handle POST request if needed
        pass
    return render_template('update_report.html')
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Load image
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/predict', methods=['GET'])
def predict():
    file_path = request.args.get('file_path')

    # Step 1: Check if image is a Chest X-ray
    img_array = preprocess_image(file_path)
    is_xray = cxr_identifier.predict(img_array)[0][0]

    if is_xray > 0.5:
        return render_template('result.html', result="❌ Invalid Image! Please upload a Chest X-ray.")

    # Step 2: Classify if it's a valid Chest X-ray
    prediction = cxr_classifier.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    classes = ['covid', 'normal', 'pneumonia']
    result = classes[predicted_class]

    return render_template('result.html', result=f"✅ Chest X-ray detected! Classification: {result}")


@app.route('/download_report/<patient_id>', methods=['GET'])
def download_report(patient_id):
    # Ensure the reports folder exists
    report_folder = 'reports'
    if not os.path.exists(report_folder):
        os.makedirs(report_folder)

    # Retrieve patient details from MongoDB
    patient_collection = get_patient_collection()
    patient = patient_collection.find_one({'patient_id': patient_id})

    # Load the CXR image for prediction
    file_path = os.path.join(UPLOAD_FOLDER, patient['cxr_image'])
    img = load_img(file_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    classes = ['covid', 'normal', 'pneumonia']
    result = classes[predicted_class]

    # Generate PDF report
    report_path = os.path.join(report_folder, f'report_{patient_id}.pdf')
    c = canvas.Canvas(report_path, pagesize=letter)
    
    # Add patient details
    c.drawString(100, 750, f"Patient ID: {patient_id}")
    c.drawString(100, 735, f"Name: {patient['name']}")
    c.drawString(100, 720, f"Age: {patient['age']}")
    c.drawString(100, 705, f"Gender: {patient['gender']}")
    c.drawString(100, 690, f"Disease: {result}")

    # Add the CXR image
    cxr_image = ImageReader(file_path)
    c.drawImage(cxr_image, 100, 450, width=200, height=200)

    # Add the visualization image
    visualization_path = os.path.join(UPLOAD_FOLDER, f"visualization_{patient_id}.png")
    c.drawImage(visualization_path, 100, 200, width=200, height=200)

    c.save()

    # Send the file as an attachment to the user
    return send_file(report_path, as_attachment=True, download_name=f'report_{patient_id}.pdf')
@app.route('/check_details', methods=['GET', 'POST'])
def check_details():
    if request.method == 'POST':
        patient_id = request.form['patient_id']
        patient_collection = get_patient_collection()
        
        # Query the database for the patient with the provided ID
        patient = patient_collection.find_one({'patient_id': patient_id})
        
        if patient:
            # If the patient is found, render the template with patient details
            return render_template('check_details.html', patient=patient)
        else:
            # If no patient is found, return an error message
            error = f"No patient found with ID: {patient_id}"
            return render_template('check_details.html', error=error)
    
    # If it's a GET request, just render the form
    return render_template('check_details.html')
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == ADMIN_CREDENTIALS['username'] and password == ADMIN_CREDENTIALS['password']:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_appointments'))  # Redirect after login
        else:
            return render_template('admin_login.html', error="Invalid Credentials")

    return render_template('admin_login.html')
@app.route('/admin_appointments')
def admin_appointments():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    appointments = get_all_appointments()  # Fetch from the database

    # Filter appointments for today
    today = today_date.today().date()
    today_appointments = [appointment for appointment in appointments if appointment['date'].date() == today]

    # Count today's appointments
    appointment_count = len(today_appointments)

    return render_template('admin_dashboard.html', appointments=today_appointments, appointment_count=appointment_count)

@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
  # Generate and insert a simple plot (e.g., prediction confidence)
    # fig, ax = plt.subplots()
    # ax.bar(classes, prediction[0])
    # ax.set_title('Prediction Confidence')
    # plt_path = os.path.join(report_folder, f'plot_{patient_id}.png')
    # fig.savefig(plt_path)
    # plt.close(fig)
    # c.drawImage(plt_path, 1 * inch, height = 87 * inch, width=40 * inch, preserveAspectRatio=True)

    # c.showPage()
    # c.save()

    # Send the file as an attachment to the user



#     import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.efficientnet import preprocess_input

# # Load the saved model
# model_path = "path_to_your_saved_model.h5"  # Replace with the actual model path
# ensemble_model = load_model(model_path)

# # Function to preprocess the uploaded image
# def preprocess_image(img_path, target_size=(224, 224)):
#     img = image.load_img(img_path, target_size=target_size)  # Load image and resize
#     img_array = image.img_to_array(img)  # Convert to numpy array
#     img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input shape
#     img_array = preprocess_input(img_array)  # Apply preprocessing (if required)
#     return img, img_array

# # Upload and preprocess an image
# image_path = "path_to_your_image.jpg"  # Replace with actual image path
# original_img, processed_img = preprocess_image(image_path)

# # Make prediction
# predictions = ensemble_model.predict(processed_img)
# predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class index
# confidence = np.max(predictions) * 100  # Get confidence score

# # Plot the image with the predicted class
# plt.imshow(original_img)
# plt.axis('off')
# plt.title(f"Predicted: Class {predicted_class} ({confidence:.2f}%)")
# plt.show()
# predicted_class = np.argmax(predictions, axis=1)[0]
# class_names = ["Class1", "Class2", "Class3"]  # Replace with actual class names
# predicted_class_name = class_names[predicted_class]
# plt.title(f"Predicted: {predicted_class_name} ({confidence:.2f}%)")


