from django.shortcuts import render
from . models import User
import os 

# Create your views here.

def index(request):
 
    return render(request, 'index.html')

def uploadImage(request):
    print("upload")
    pic = request.FILES['image']
    temp = pic.name
    user = User(img = pic)
    user.save()

    import cv2 
    import numpy as np
    import copy 
    import math

    def quantization(pixels, bins, range_):
        m = range_[0]
        interval_size = range_[1]-range_[0]
        interval_size/=bins

        for i in range(len(pixels)):
            for j in range(len(pixels[i])):
                pixels[i][j] = ((pixels[i][j]-m)/interval_size)
        return pixels

    def visualise(depth_map,name,beta):
        d = copy.deepcopy(depth_map)
        d = quantization(d,255,[d.min(),d.max()]).astype(np.uint8)

        d = cv2.applyColorMap(d, cv2.COLORMAP_HOT)
        cv2.imwrite("./output/" + name + "_" + str(beta) + ".jpg",d)

    def relu(x):
        if x<0:
            return 0
        else:
            return x
        
    def reverse_relu(limit,x):
        if x>limit:
            return limit
        else:
            return x

    def guided_filter(image, g_image, eps = 0):
        blur_factor = (50,50)
        mean_i = cv2.blur(image,blur_factor)
        mean_g = cv2.blur(g_image,blur_factor)

        corr_gi = cv2.blur(g_image*image,blur_factor)
        corr_gg = cv2.blur(g_image*g_image,blur_factor)

        var_g = corr_gg - mean_g*mean_g
        cov_gi = corr_gi - mean_g*mean_i

        a = cov_gi / (var_g + eps)
        b = mean_i - (a*mean_g)

        mean_a = cv2.blur(a,blur_factor)
        mean_b = cv2.blur(b,blur_factor)

        q = mean_a * g_image + mean_b

        return q

    # Variables
    # filename = "../media/uploaded/hazy_img.jpg"
    noise = 0 # guided filter, eps
    beta = 1 #dehazing strength (scattering coefficient)

    # Theta values
    theta0 = 0.180096
    theta1 = 1.014730
    theta2 = -0.734980
    sigma = 0.041339

    n_size = 5 # Size of neighbourhood considered for min filter
    blur_strength = 15 # Strength of blurring after min filter in depthmap

    # Reading the image
    hazed_img = cv2.imread("D:\\SEM6\\SDP\\SDP_Django\\imagedehazing\\media\\uploaded\\" + temp,1)
    # print(type(hazed_img)) 

    # Extracting the brightness and saturation values from the image
    hsv = cv2.cvtColor(hazed_img, cv2.COLOR_BGR2HSV)
    value = hsv[:,:,2].astype('float')/255 # Intensity values of image
    saturation = hsv[:,:,1].astype('float')/255 # Saturation values of image

    # Calculating depth map
    depth_map = theta0 + theta1*value + theta2*saturation + np.random.normal(0,sigma, hsv[:,:,0].shape)
    visualise(depth_map,"1_depth_map",beta)

    # Calculating min-filtered depth map
    new_depth_map = copy.deepcopy(depth_map) 

    width = depth_map.shape[1]
    height = depth_map.shape[0]

    for i in range(height):
        for j in range(width):
            x_low = relu(i-n_size)
            x_high =  reverse_relu(height-1,i+n_size)+1
            y_low = relu(j-n_size)
            y_high =  reverse_relu(width-1,j+n_size)+1
            new_depth_map[i][j] = np.min( depth_map[x_low:x_high,y_low:y_high] )

    visualise(new_depth_map,"2_min_filter_depth_map",beta)

    # Calculating Blurred depth map
    blurred_depth_map = guided_filter(new_depth_map, depth_map, noise)
    visualise(blurred_depth_map, "3_blurred_depth_map", beta)

    depth_map_1d = np.ravel(blurred_depth_map)
    rankings = np.argsort(depth_map_1d)

    top_one_percent = (99.9*len(rankings))/100 
    indices = np.argwhere(rankings > top_one_percent).ravel()

    indices_image_rows = indices//width
    indices_image_columns = indices % width

    atmospheric_light = np.zeros(3) # A
    intensity = -np.inf
    for x in range(len(indices_image_rows)):
        i = indices_image_rows[x]
        j = indices_image_columns[x]

        # if value is greater than intensity then it will be considered as atmospheric light
        if value[i][j] >= intensity:
            atmospheric_light = hazed_img[i][j]
            intensity = value[i][j]

    t = np.exp(-beta*blurred_depth_map)

    denom = np.clip(t,0.1,0.9)
    numer = hazed_img.astype("float") - atmospheric_light.astype("float")

    dehazed_image = copy.deepcopy(hazed_img).astype("float")

    for i in range(len(dehazed_image)):
        for j in range(len(dehazed_image[i])):
            dehazed_image[i][j] = numer[i][j]/denom[i][j]
        
    dehazed_image += atmospheric_light.astype("float")

    #cv2.imwrite("../media/hazy_img" + ".jpg", hazed_img)
    if os.path.isfile('D:\\SEM6\\SDP\\SDP_Django\\imagedehazing\\dehazing\\static\\images\\dehazed_img.jpg')== True:
        print("exist")
        os.remove('D:\\SEM6\\SDP\\SDP_Django\\imagedehazing\\dehazing\\static\\images\\dehazed_img.jpg')

    cv2.imwrite("D:\\SEM6\\SDP\\SDP_Django\\imagedehazing\\media\\output\\dehaze" + ".jpg", dehazed_image)
    cv2.imwrite("D:\\SEM6\\SDP\\SDP_Django\\imagedehazing\\dehazing\\static\\images\\dehazed_img" + ".jpg", dehazed_image)
    # J = ( (I - A)/t ) + A 

    pic.name = temp
    user.save()

    users = User.objects.all()
    p = users[len(users)-1].img
    print(p.url) 
    return render(request, 'output.html', {'pic':p.url})