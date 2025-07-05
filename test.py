import numpy as np
from tensorflow.keras.models import load_model

#load model
model = load_model('my_model.h5') 


def get_user_input():
    print("Enter soil parameters for classification:")
    try:
        pH = float(input("  Soil pH (e.g., 6.5): "))
        N = float(input("  Nitrogen (N) content (e.g., 50): "))
        P = float(input("  Phosphorus (P) content (e.g., 40): "))
        K = float(input("  Potassium (K) content (e.g., 60): "))
        moisture = float(input("  Moisture percentage (e.g., 20): "))
    
    except ValueError:
        print("Invalid input! Please enter numeric values.")
        return None

    
    return np.array([[pH, N, P, K, moisture]])

# fungsi prediksi
def predict_plant_type(input_data):
    if input_data is None:
        print("No valid input provided. Cannot make predictions.")
        return

    
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)  

  
    plant_types = {0: 'Rice', 1: 'Wheat', 2: 'Corn', 3: 'Soybean'}  
    recommended_plant = plant_types.get(predicted_class[0], "Unknown Plant")

    print(f"\nRecommended Plant Type: {recommended_plant}")

# Main program 
if __name__ == "__main__":
    print("=== Soil to Plant Type Classification ===")
    while True:
        user_input = get_user_input()
        predict_plant_type(user_input)
        
       
        again = input("\nDo you want to classify another soil sample? (yes/no): ").strip().lower()
        if again != 'yes':
            print("Exiting simulation. Goodbye!")
            break
