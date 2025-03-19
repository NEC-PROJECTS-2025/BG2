from pymongo import MongoClient

def get_db():
    # MongoDB connection string for localhost
    client = MongoClient("mongodb://localhost:27017/")
    db = client['CXR']  # Database name
    return db

# Function to get the patient collection
def get_patient_collection():
    db = get_db()
    return db['new_registration']  # Collection name
def get_appointment_collection():
    db = get_db()
    return db['new_appointment'] 

def get_all_appointments():
    db = get_db()
    appointment_collection = db['new_appointment']
    
    # Fetch all appointments
    appointments = list(appointment_collection.find({}, {'_id': 0}))  # Exclude `_id` from results

    return appointments

