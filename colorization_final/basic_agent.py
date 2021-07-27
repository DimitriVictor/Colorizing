from process_image import *
from k_means import calculate_distance, get_representative_colors
from numpy import asarray

# NOTE: COORDINATES ARE IN (X,Y) FORM

# Given set of colors and a pixel color, calculates eucliden distance of the pixel
#     to each color and returns the closest color
def get_closest_color(colors,pixel_color):
    min_distance = calculate_distance(colors[0],pixel_color)
    min_color = 0
    
    for i in range(1,len(colors)):
        this_distance = calculate_distance(colors[i], pixel_color)
        if this_distance < min_distance:
            min_distance = this_distance
            min_color = i
            
    return colors[min_color]


# Input-an array of six tuples where the first index of each is the difference and the 2nd is coordinates
# Returns the index and value for which the difference is greater
def get_most_different(six_closest):
    max_difference, max_difference_index = -1,-1
    for i in range(6):
        this_difference = six_closest[i][0]
        
        if this_difference > max_difference: # update max difference if this is bigger
            max_difference = this_difference
            max_difference_index = i
            
    return max_difference, max_difference_index


# Given two coordinates and the image
# Finds the difference in two nine pixel chunks surrounding those coordinates
def nine_pixel_difference(grayscale_image, x1, y1, x2, y2, max_difference):
    difference = 0
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            difference += abs(grayscale_image[y1+j][x1+i][0] - grayscale_image[y2+j][x2+i][0])
            if difference > max_difference:
                return 9999
                
    return difference

    
    

# Given the grayscale image and coordinates in the right half
# finds the 6 most similar 9 pixel chunks in the left half of the image
# the grayscale image is in numpy array form
def get_six_closest(grayscale_image,x,y):
    height, width = len(grayscale_image),len(grayscale_image[0])
    
    six_closest = [(9999,(-1,-1)) for i in range(6)]
    max_difference, max_difference_index = 9999, 0
    
    for x2 in range(width/2):
        for y2 in range(height):
            if x2 == 0 or x2 == (width/2 - 1) or y2 == 0 or y2 == (height-1):
                continue
            this_difference = nine_pixel_difference(grayscale_image, x, y, x2, y2, max_difference)
            if this_difference < max_difference:
                six_closest[max_difference_index] = (this_difference, (x2,y2))
                max_difference, max_difference_index = get_most_different(six_closest)
    return six_closest


# Given a list of six tuples, where each tuple is an int value(the difference) and a coordinate, 
#     return the majority occurring color
def find_majority_of_six_closest(six_closest,image, colors):
    coordinates = []
    for entry in six_closest:
        coordinates.append(entry[1])
    
    color_counts = [0,0,0,0,0]
    
    for coordinate in coordinates:
        this_pixel = image.getpixel(coordinate)
        if this_pixel[0] == colors[0][0]:
            color_counts[0] += 1
        elif this_pixel[0] == colors[1][0]:
            color_counts[1] += 1
        elif this_pixel[0] == colors[2][0]:
            color_counts[2] += 1
        elif this_pixel[0] == colors[3][0]:
            color_counts[3] += 1
        elif this_pixel[0] == colors[4][0]:
            color_counts[4] += 1
    
            
    # get index of maximum color
    max_color_count = 0
    max_color_index = -1
    for i in range(len(color_counts)):
        if color_counts[i] >= max_color_count:
            max_color_count = color_counts[i]
            max_color_index = i
            
    color = colors[max_color_index]
    
    return (color[0],color[1],color[2])
    
    
    
# Runs basic agent as described in project doc
# Inputs are image name and the maximum length or width to be resized to
def basic_agent(image, max_dimension=260.0):
    
    colors = get_representative_colors(image)
    print("Got representative colors")
    
    # obtain gray image
    grayscale_image = grayscale(image)
    
    width, height = grayscale_image.size
    
    
    shrink_factor = max_dimension/max(width,height)
    new_width, new_height = int(width*shrink_factor), int(height*shrink_factor)
    
    # resize grayscale image
    grayscale_image = grayscale_image.resize((new_width, new_height))
    grayscale_image_array = asarray(grayscale_image)
    
    # load color image
    image = Image.open(image)
    image = image.resize((new_width, new_height))

    # create resulting image object to store predicted values in
    res = Image.new(image.mode,image.size)
    
    width, height = image.size
    
    # iterate through left half of photo, assign each pixel one of 5 colors in result
    for i in range(width/2):
        for j in range(height):
            if i == 0 or j == 0 or j == (height-1):
                res.putpixel((i,j),(0,0,0))
                continue
            pixel = image.getpixel((i,j))
            representative_color = get_closest_color(colors, pixel)
            res.putpixel((i,j),representative_color)
    res.show()
        
    
    # vars to keep track of progress
    filled_count = 0.0
    to_fill = width*height/2
    
    # iterate through right half of image making prediction
    for i in range(width/2,width-1):
        for j in range(height):
            
            # Make border black
            if i == (width-2) or j == 0 or j == (height-1):
                res.putpixel((i,j),(0,0,0))
                continue
            filled_count += 1.0
            
            # get six closest 9 pixel sets on the left
            six_closest = get_six_closest(grayscale_image_array,i,j)
            
            # find most common of the six closest
            majority_color = find_majority_of_six_closest(six_closest, res, colors)

            # place pixel
            res.putpixel((i,j),majority_color)
            
            # progress shower
            if filled_count % 25 == 0:
                print(str(100*filled_count/to_fill)+"%")
            if filled_count % 1000 == 0:
                res.show()
                
    
            
    res.show()
            

            
    
    
basic_agent('images/beach.jpeg')
