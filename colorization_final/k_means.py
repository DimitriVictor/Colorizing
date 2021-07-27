from process_image import get_left_half_rgb_values
from random import randint
import math

# Calculates euclidean distance between two 3d points
# Used between the mean color of a cluster and a pixel color
def calculate_distance(cluster_color,pixel_color):
    distance = (cluster_color[0]-pixel_color[0])**2 + (cluster_color[1]-pixel_color[1])**2 + (cluster_color[2]-pixel_color[2])**2
    distance = math.sqrt(distance)
    return distance

# Returns the mean of a cluster
def get_cluster_mean(cluster):
    red_sum = 0
    green_sum = 0
    blue_sum = 0
    
    # add to sum of red, green, and blue values
    for i in range(len(cluster)):
        red_sum += cluster[i][0]
        green_sum += cluster[i][1]
        blue_sum += cluster[i][2]
    
    # divide each sum by the length of the cluster
    red_mean = red_sum*1.0/len(cluster)
    green_mean = green_sum*1.0/len(cluster)
    blue_mean = blue_sum*1.0/len(cluster)
    
    return (red_mean,green_mean,blue_mean,255)

# Returns whether or not the k-means centers are very similar to the last iteration
# Inputs - set of centers and set of old centers
def centers_unsettled(centers,prev_centers):
    center_count = len(centers)
    for i in range(center_count):
        if abs(centers[i][0]-prev_centers[i][0]) + abs(centers[i][1]-centers[i][1]) + abs(centers[i][2]-centers[i][2]) > 5:
            # this conditional runs if the sum of the difference of each color is greater than 5
            return True 
        
    return False
    
    
# Gets 5 most representative colors of an image
# Image input should be a string filename of an image in this directory
def get_representative_colors(image):
    colors = get_left_half_rgb_values(image) # get a list of pixel colors of left half of image
    
    # initialize centers to random pixels with RGB between 0 and 255
    centers = []
    for i in range(5):
        centers.append((randint(0,255),randint(0,255),randint(0,255),255))

    cluster_0 = []
    cluster_1 = []
    cluster_2 = []
    cluster_3 = []
    cluster_4 = []
    
    # initialize previous centers variable
    prev_centers = [(-1,-1,-1,255),(-1,-1,-1,255),(-1,-1,-1,255),(-1,-1,-1,255),(-1,-1,-1,255)]
    
    while(centers_unsettled(centers,prev_centers)):
        
        # for each color find the cluster center it is closest to
        for color_pixel in colors:
            min_cluster = 0
            min_distance = calculate_distance(centers[0],color_pixel)
            for i in range(1,5):
                this_dist = calculate_distance(centers[i], color_pixel)
                if this_dist < min_distance:
                    min_distance = this_dist
                    min_cluster = i
            
            # assign this pixel to its corresponding cluster
            if min_cluster == 0:
                cluster_0.append(color_pixel)
            elif min_cluster == 1:
                cluster_1.append(color_pixel)
            elif min_cluster == 2:
                cluster_2.append(color_pixel)
            elif min_cluster == 3:
                cluster_3.append(color_pixel)
            elif min_cluster == 4:
                cluster_4.append(color_pixel)
        
        # save the values of the centers as they are
        prev_centers = centers[:]
        
        # update centers
        if len(cluster_0) != 0:
            centers[0] = get_cluster_mean(cluster_0)
        else:
            centers[0] = (randint(0,255),randint(0,255),randint(0,255),255)
            
        if len(cluster_1) != 0:
            centers[1] = get_cluster_mean(cluster_1)
        else:
            centers[1] = (randint(0,255),randint(0,255),randint(0,255),255)
            
        if len(cluster_2) != 0:
            centers[2] = get_cluster_mean(cluster_2)
        else:
            centers[2] = (randint(0,255),randint(0,255),randint(0,255),255)
            
        if len(cluster_3) != 0:
            centers[3] = get_cluster_mean(cluster_3)
        else:
            centers[3] = (randint(0,255),randint(0,255),randint(0,255),255)
            
        if len(cluster_4) != 0:
            centers[4] = get_cluster_mean(cluster_4)     
        else:
            centers[4] = (randint(0,255),randint(0,255),randint(0,255),255)  
        
        
        # reset clusters
        cluster_0 = []
        cluster_1 = []
        cluster_2 = []
        cluster_3 = []
        cluster_4 = []
        
    
    res = []
    for center in centers:
        res.append((int(center[0]),int(center[1]),int(center[2])))
    return res
    
#colors = get_representative_colors('images/trees.jpeg')
#print(colors)
    
