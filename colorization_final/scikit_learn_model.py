from sklearn.neural_network import MLPRegressor
from process_image import process_image, grayscale
from PIL import Image

# Generates a neural network model to predict a specific color
def get_sklearn_model(image_name, color_type, learn_rate):
    
    # generate image and grayscale image
    image = process_image(image_name)
    grayscale_image = grayscale(image_name)
    
    width, height = image.size
    
    X = [] # set of input vectors
    actual_values = [] # set of correlated actual values
    
    # get all input vectors
    for x in range(width/2):
        for y in range(height):
            if x == 0 or x == (width/2 - 1) or y == 0 or y == (height-1):
                continue

            # generate input vector for this coordinate
            # the 9 grayscale values surrounding and including (x,y)
            input_vector = []
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    pixel = grayscale_image.getpixel((x+i,y+j))
                    input_vector.append(pixel[0])
            
            # get actual color value for this coordinate
            actual_pixel = image.getpixel((x,y))
            actual_value = actual_pixel[color_type]
            
            # add to data
            X.append(input_vector)
            actual_values.append(actual_value)
    
    # make sklearn model object
    model = MLPRegressor(alpha=learn_rate, hidden_layer_sizes=(30,), random_state=1)
    
    # fit to data
    model.fit(X,actual_values)
    
    return model


# Given models for each color and an image, fill in the right half from the grayscale values
def grayscale_to_color(red_model, green_model, blue_model, image_name):
    # load color image
    image = Image.open(image_name)
    
    grayscale_image = grayscale(image_name)
    grayscale_image.show()
    
    res = Image.new(image.mode,image.size)
    
    width, height = image.size
    
    # fill in left half of image with actual pixel values
    for x in range(width/2):
        for y in range(height):
            if x == 0 or y == 0 or y == (height-1):
                res.putpixel((x,y),(0,0,0))
            else:
                res.putpixel((x,y), image.getpixel((x,y)))
                
    pixels_predicted = 0
    
    for x in range(width/2,width):
        for y in range(height):
            
            # color borders black
            if x == 0 or x == (width-1) or y == (height-1):
                res.putpixel((x,y),(0,0,0))
                
            else: # generate input vector
                input_vector = []
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        pixel = grayscale_image.getpixel((x+i,y+j))
                        input_vector.append(pixel[0])
                        
                
                # make prediction for red green and blue
                red_prediction = int(red_model.predict([input_vector])[0])
                green_prediction = int(green_model.predict([input_vector])[0])
                blue_prediction = int(blue_model.predict([input_vector])[0])
                
                if red_prediction > 255:
                    red_prediction = 255
                elif red_prediction < 0:
                    red_prediction = 0
                if green_prediction > 255:
                    green_prediction = 255
                elif green_prediction < 0:
                    green_prediction = 0
                if blue_prediction > 255:
                    blue_prediction = 255
                elif blue_prediction < 0:
                    blue_prediction = 0
                
                # place pixel
                predicted_pixel = (red_prediction,green_prediction,blue_prediction)
                res.putpixel((x,y),predicted_pixel)
                
                pixels_predicted += 1
                if pixels_predicted % 200 == 0:
                    print(predicted_pixel)
                
                
    return res

red_model = get_sklearn_model('images/beach.jpeg', 0, 0.005)
print("1")
green_model = get_sklearn_model('images/beach.jpeg', 1, 0.005)
print("2")
blue_model = get_sklearn_model('images/beach.jpeg', 2, 0.005)
print("3")

colorized = grayscale_to_color(red_model, green_model, blue_model, 'images/beach.jpeg')

colorized.show()


