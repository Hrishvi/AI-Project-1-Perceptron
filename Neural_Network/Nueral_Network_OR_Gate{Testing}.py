import pickle
import sys
import os


def load_model(filename):
    # Automatically look in the Trained_Models folder
    filepath = os.path.join('Trained_Models', filename)
    
    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        print(f"✓ Model loaded from: {filepath}")
        return model_data['weight_1'], model_data['weight_2'], model_data['bias']
    except FileNotFoundError:
        print(f"Error: Could not find '{filename}' in the Trained_Models folder.")
        sys.exit()


def step_function(weighted_sum):
    """Activation function"""
    if weighted_sum >= 0:
        return 1
    else:
        return 0


filename = input("What is the Name of the Neural Network File you are trying to open: ")

# FIX #1: Store the returned values in variables
weight_1, weight_2, bias = load_model(filename)

print(f"\nLoaded weights:")
print(f"Weight 1: {weight_1:.4f}")
print(f"Weight 2: {weight_2:.4f}")
print(f"Bias: {bias:.4f}\n")

while True:
    continue_testing = input("Do you want to start/continue testing[y for yes, anything else will quit]: ")
    if continue_testing.lower() == "y":
        input1 = int(input("Give your first input: "))
        input2 = int(input("Give your second input: "))
        
        weighted_sum = (input1 * weight_1) + (input2 * weight_2) + bias
        output = step_function(weighted_sum)
        print(f"Output: {output}\n")
    else:
        break

sys.exit()