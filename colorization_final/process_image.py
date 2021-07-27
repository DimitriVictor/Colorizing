from PIL import Image

# Returns grayscale version of an image
# Input is an image file name or path, outputs a PIL Image in grayscale
def grayscale(image):
    image = Image.open(image)
    
    width, height = image.size
    
    # create new image object to store the grayscale values
    grayscale = Image.new(image.mode,image.size)
    for i in range(width):
        for j in range(height):
            pixel = image.getpixel((i,j))
            red = pixel[0]
            green = pixel[1]
            blue = pixel[2]
            
            
            #grayscale calculation from project doc
            gray = 0.21*red + 0.72*green + 0.07*blue
            gray = int(gray)
            
            grayscale.putpixel((i,j),(gray,gray,gray))
            
    return grayscale

# Input-image filename
# Returns a PIL image
def process_image(image):
    image = Image.open(image)
    return image

# returns a list of all the pixels in the left half of the image
def get_left_half_rgb_values(image):
    image = Image.open(image)
    width, height = image.size 
    
    colors = []
    
    for i in range(width/2):
        for j in range(height):
            colors.append(image.getpixel((i,j)))
            
    return colors
    
    
