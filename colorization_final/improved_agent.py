from train_model import get_model, make_prediction, test_model
from PIL import Image
from process_image import grayscale


# Run improved agent - generate red green and blue models to make predictions
def improved_agent(image_name, learn_rate):
    
    # get red, green, and blue models
    red_weights, red_bias = get_model(image_name, 0, learn_rate)
    print("red weights" + str(red_weights))
    green_weights, green_bias = get_model(image_name, 1, learn_rate)
    print("green weights" + str(green_weights))
    blue_weights, blue_bias = get_model(image_name, 2, learn_rate)
    print("blue weights" + str(blue_weights))

    # load color image
    image = Image.open(image_name)
    
    # generate grayscale image
    grayscale_image = grayscale(image_name)
    
    # generate result image object
    res = Image.new(image.mode,image.size)
    
    width, height = image.size
    
    # set borders to black and left half to original values
    for x in range(width/2):
        for y in range(height):
            if x == 0 or y == 0 or y == (height-1):
                res.putpixel((x,y),(0,0,0))
            else:
                res.putpixel((x,y), image.getpixel((x,y)))
    
    pixels_counted = 0
    
    # iterate through right half of image
    for x in range(width/2,width):
        for y in range(height):
            if x == 0 or x == (width-1) or y == (height-1):
                res.putpixel((x,y),(0,0,0))
                
            
            else:
                pixels_counted += 1
                
                # generate input vector
                input_vector = []
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        pixel = grayscale_image.getpixel((x+i,y+j))
                        input_vector.append(pixel[0])
                        
                # make separate red, green, and blue predictions
                red_prediction = int(255*make_prediction(red_weights, red_bias, input_vector))
                green_prediction = int(255*make_prediction(green_weights, green_bias, input_vector))
                blue_prediction = int(255*make_prediction(blue_weights, blue_bias, input_vector))
                
                # place predicted pixel in result
                predicted_pixel = (red_prediction,green_prediction,blue_prediction)
                res.putpixel((x,y),predicted_pixel)
                
    res.show()
    
improved_agent('images/trees_low_res.jpeg', 0.0000005)
#improved_agent('images/beach.jpeg', 0.000005)
#improved_agent('images/flame_thing.png', 0.00000005)
    
    