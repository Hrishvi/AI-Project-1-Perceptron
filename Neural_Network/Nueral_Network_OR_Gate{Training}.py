import random
import sys
import pickle
import os

# 1. The Environment
# Format: [Input 1, Input 2, Target Output]
training_data = [
    [0, 0, 0], 
    [0, 1, 0], 
    [1, 0, 0], 
    [1, 1, 1]  
]

# 2. The Brain's Starting State
# We initialize weights and bias with random numbers. 
# The network knows nothing yet.
weight_1 = random.uniform(-1, 1)
weight_2 = random.uniform(-1, 1)
bias = random.uniform(-1, 1)

# The learning rate determines how big of an adjustment we 
# make when we make a mistake. Too big, we overshoot; too small, we learn too slowly.
learning_rate = 0.1

# 3. The Activation Function And The Save Code
def step_function(weighted_sum):
    """If the sum is positive, the neuron fires (1). Otherwise, it stays quiet (0)."""
    if weighted_sum >= 0:
        return 1
    else:
        return 0
    
def save_model(filename, w1, w2, b):
    # Ensure the directory exists
    if not os.path.exists('Trained_Models'):
        os.makedirs('Trained_Models')
    
    # Save inside the folder
    filepath = os.path.join('Trained_Models', filename)
    model_data = {'weight_1': w1, 'weight_2': w2, 'bias': b}
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✓ Model saved to: {filepath}")

# 4. The Training Loop
epoch = 0

print("Starting Training...")

while True:
    error_count = 0 # Track how many mistakes we make per epoch
    
    for data in training_data:
        x1 = data[0]
        x2 = data[1]
        target = data[2]
        
        # --- FORWARD PASS (Making a Guess) ---
        # Calculate the total score: (Input 1 * Weight 1) + (Input 2 * Weight 2) + Bias
        weighted_sum = (x1 * weight_1) + (x2 * weight_2) + bias
        guess = step_function(weighted_sum)
        print(f"The Weighted Sum is {weighted_sum}")
        print(f"The Guess is {guess}")
        
        # --- CALCULATE ERROR ---
        # If target is 1 and guess is 0, error is +1 (we need to go higher).
        # If target is 0 and guess is 1, error is -1 (we need to go lower).
        # If they match, error is 0.
        error = target - guess
        print(f"The Error is {error}")
        
        if error != 0:
            error_count += 1
            
        # --- BACKWARD PASS (Learning from the Mistake) ---
        # We only update weights if we made a mistake (error isn't 0).
        # We multiply by the input so we only adjust weights for inputs that were active.
        weight_1 += learning_rate * error * x1
        weight_2 += learning_rate * error * x2
        bias += learning_rate * error
        
    epoch += 1
    print(f"Epoch {epoch} completed with {error_count} mistakes.")
    
    # If we made zero mistakes, the brain is perfectly trained!
    if error_count == 0:
        print("\nNeuron has mastered the Testing Data!")
        break

# Display the final learned parameters
print(f"\nFinal Brain State:")
print(f"Weight 1: {weight_1:.2f}")
print(f"Weight 2: {weight_2:.2f}")
print(f"Bias: {bias:.2f}")

what_to_do = input("The model has been trained. Do you want to Test the current model[t] or Save it[s], or you can Type anything else to quit: ")

if what_to_do.lower() == "t":
    while True:
        continue_testing = input("Do you want to continue testing[y for yes, aything else will quit]: ")
        if continue_testing.lower() == "y":
            input1 = int(input("Give your first input: "))
            input2 = int(input("Give your second input: "))

            weighted_sum = (input1 * weight_1) + (input2 * weight_2) + bias
            output = step_function(weighted_sum)
            print(output)
        else:
            break
    sys.exit()
    
elif what_to_do.lower() == "s":
    filename = input("Enter filename to save: ")
    save_model(filename, weight_1, weight_2, bias)
    sys.exit()
    
else:
    print("You have typed q or anything other than q, t, or s. The program is quiting")
    sys.exit()
    
    
