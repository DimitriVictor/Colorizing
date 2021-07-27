from process_image import process_image, grayscale
import numpy as np
import random

e = 2.71828

# metrics to keep track of loss
# keeps track of sum of losses for 300 pixels at a time
global current_300_loss_sum
global current_loss_count
current_300_loss_sum = 0
current_loss_count = 0


def sigmoid(x):
    return 1/(1+e**(-x))

# makes prediction for a color value given an input vector and weights/bias
# prediction = sigmoid(input dot weights + bias)
def make_prediction(weights, bias, input_vector):
    weights_np = np.array(weights)
    input_vector_np = np.array(input_vector)
    
    prediction = np.dot(weights_np, input_vector_np) + bias
    
    prediction = sigmoid(prediction)
    
    return prediction


# Take a gradient step
def gradient_step(input_vector, actual_value, current_weights, bias, learn_rate):
    prediction = make_prediction(current_weights, bias, input_vector)
    
    error = prediction - actual_value
    
    # metrics to display loss
    # every 300 pixels it prints the average loss over the previous 300 pixels
    global current_loss_count
    global current_300_loss_sum
    current_loss_count += 1
    current_300_loss_sum += error**2
    if current_loss_count % 300 == 0:
        current_loss_count = 0
        #print(current_300_loss_sum/300)
        current_300_loss_sum = 0
    
    
    # math of all of this explained in project document
    
    input_vector_np = np.array(input_vector)

    adjustment_factor = 2*(prediction-actual_value)*(prediction)*(1 - prediction/255)
    
    adjustment = learn_rate*(adjustment_factor*input_vector_np)
    
    bias = bias + learn_rate*adjustment_factor*1
    
    return np.subtract(current_weights, adjustment), bias
    
    
    """alternate version
    for i in range(len(current_weights)):
        current_weights[i] = current_weights[i] - learn_rate*error*input_vector[i]
    
    # bias = bias - learn_rate*adjustment_factor*bias
    bias = bias - learn_rate*error*bias
    
    #print(adjustment)
    #print("--------")
    
    # return np.subtract(current_weights, adjustment), bias
    return current_weights, bias
    """



# Return set of weights and bias based on a regression model
def get_model(image_name, color_type, learn_rate):
    print("Getting model")
    
    # get image and grayscale image objects
    image = process_image(image_name)
    grayscale_image = grayscale(image_name)
    
    width, height = image.size
    
    weights = [0.001,0.002,0.003,0.003,0.002,0.003,0.001,0.001,0.002]
    bias = 0.002
    steps = 0
    
    for x in range(width/2):
        for y in range(height):
            
            # skip borders
            if x == 0 or x == (width/2 - 1) or y == 0 or y == (height-1):
                continue
            
            """
            # dropout function
            skip_probability = random.uniform(0,1)
            if skip_probability > 0.8:
                continue
            """
            
            steps += 1
            
            # get input vector
            input_vector = []
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    pixel = grayscale_image.getpixel((x+i,y+j))
                    input_vector.append(pixel[0])
            
            # get actual color value
            actual_pixel = image.getpixel((x,y))
            actual_value = actual_pixel[color_type]
            actual_value = 1.0*actual_value/255
            
            # take gradient step to update weights and bias
            weights, bias = gradient_step(input_vector, actual_value, weights, bias, learn_rate)

    return weights, bias


# Return the loss of a model on the images left half
def test_model(image_name, color_type, weights, bias):
    image = process_image(image_name)
    
    width, height = image.size
    
    grayscale_image = grayscale(image_name)
    
    errors = []
    
    for x in range(width/2,width):
        for y in range(height):
            if x == 0 or x == (width-1) or y == 0 or y == (height-1):
                continue
            
            input_vector = []
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    pixel = grayscale_image.getpixel((x+i,y+j))
                    input_vector.append(pixel[0])
                    
                    
            actual_pixel = image.getpixel((x,y))
            actual_value = 1.0*actual_pixel[color_type]/255
            
            predicted_value = make_prediction(weights, bias, input_vector)
            
            errors.append((actual_value-predicted_value)**2)
    
    return sum(errors)/len(errors)
    