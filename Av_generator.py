from mpdaf.obj import Cube, WCS, WaveCoord, bounding_box, image_angle_from_cd, image
from mpdaf.obj import Image
from mpdaf.obj import iter_ima
import numpy as np
import math
import astropy
from astropy.io import fits
import pandas as pd
import re
from io import StringIO
import pickle
import threadcount as tc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import ndimage

def avsmooth(Av_all):
    """
    A function created by Deanne Fisher that smooths out an A_v map before usage. Used in attenuation correctin function.
    """
    Av_smooth = Av_all.copy()
    AVrows = []
    wide = 5
    clip = 0.75
    rowclip = 3.5
    ### Series of steps that smooths out Av using first median rows
    ### then the local area.
    for m in range(Av_all.shape[0]): #for each row
        Av_row_median = np.nanmedian(Av_all[m,:]) #median of current row taken
        if Av_row_median < 0 :
            Av_row_median = 0.
        AVrows.append(Av_row_median)		
        for n in range(Av_all.shape[1]): #for all cols
            if Av_all[m,n] < 0. :
                Av_smooth[m,n] = 0
            if Av_all[m,n] > Av_row_median +rowclip :
                Av_smooth[m,n] = Av_row_median	
            if np.isnan(Av_all[m,n] ) == True :
                Av_smooth[m,n] = Av_row_median
            if 	Av_smooth[m,n] < np.median(Av_smooth[m-wide:m+wide,n-wide:n+wide]) - clip or Av_smooth[m,n] > np.median(Av_smooth[m-wide:m+wide,n-wide:n+wide])  + clip :
                Av_smooth[m,n] = np.median(Av_smooth[m-wide:m+wide,n-wide:n+wide])
    ### Plot the median Av against the rows
    rows = np.arange(0,Av_all.shape[0],1)
    # plt.figure()
    # plt.plot(rows, AVrows, ‘-’)
    # plt.xlabel("row index (pix)")
    # plt.ylabel("A_v value")
    # plt.savefig()
    return Av_smooth

def tc_line_centers(version="1.0"):
    """
    produces dictionaries containing different line central values (in angstroms)
    """
    if version=="1.0":
        catalog={"Hbeta":tc.lines.L_Hb4861.center,
                 "Halpha":6562.819,
                 "OIII_5007":tc.lines.L_OIII5007.center,
                 "N2_6583":6583.45, #(Value selected from https://physics.nist.gov/PhysRefData/ASD/lines_form.html)
                 "N2_6548":6548.05, #(Value selected from https://physics.nist.gov/PhysRefData/ASD/lines_form.html)
                 "SII_6716":6716.440, #(Value selected from https://physics.nist.gov/PhysRefData/ASD/lines_form.html)
                 "SII_6730":6730.816, #(Value selected from https://physics.nist.gov/PhysRefData/ASD/lines_form.html)
                 }
    if version=="2.0":
        catalog={"Hbeta":tc.lines.L_Hb4861.center,
                 "Halpha":6562,
                 "OIII_5007":tc.lines.L_OIII5007.center,
                 "N2_6583":6583.45, #(Value selected from https://physics.nist.gov/PhysRefData/ASD/lines_form.html)
                 "N2_6548":6548.05, #(Value selected from https://physics.nist.gov/PhysRefData/ASD/lines_form.html)
                 "SII_6716":6716.440, #(Value selected from https://physics.nist.gov/PhysRefData/ASD/lines_form.html)
                 "SII_6730":6730.816, #(Value selected from https://physics.nist.gov/PhysRefData/ASD/lines_form.html)
                 }
        
    return catalog

def temp_median_function(values,values_shape=None,radius=None): #median function
    """
    This is a median function used for DFisher's extinction correction functon that makes averages for nan pixels using the 5 closest non nan values
    """
    #following 5 lines reconstructs 2d array of input window of values
    #values_len=len(values)
    values_mod=np.insert(values, (radius*3)+1, np.nan, axis=None)          
    values_reshaped=np.reshape(values_mod, values_shape)
    # print("values_array:")
    # print(values_reshaped)
    for step in range(radius+1): #starting from the column that our target pixel sits in, we step outwards itteratively, until we find a subarray that contains the 5 closest pixels
        values_subsample=values_reshaped[:,radius-step:radius+step+1]
        # print("for step "+str(step)+":")
        # print(values_subsample)
        if np.count_nonzero(~np.isnan(values_subsample))>=5: #we stop making steps when we have 5 non NaN values in our subarray
            values_reshaped=values_subsample
            break
    #at maximum, we can only step out 5 pixels (horizontally) from our target pixel. After that, we stop taking steps
    if np.count_nonzero(~np.isnan(values_subsample))<5:
        return np.nan

    values_reshaped_radius=int((values_reshaped.shape[1]-1)/2)
    if np.count_nonzero(~np.isnan(values_reshaped))>5:#if we have more than 5 pixels in our subaray to make a median with, we cut out values on the edge until we only have 5 pixels
        values_reshaped[0,0]=np.nan
    if np.count_nonzero(~np.isnan(values_reshaped))>5:
        values_reshaped[0,values_reshaped_radius*2]=np.nan        
    if np.count_nonzero(~np.isnan(values_reshaped))>5:
        values_reshaped[2,0]=np.nan
    if np.count_nonzero(~np.isnan(values_reshaped))>5:
        values_reshaped[2,values_reshaped_radius*2]=np.nan
    if np.count_nonzero(~np.isnan(values_reshaped))>5:
        values_reshaped[1,0]=np.nan
    if np.count_nonzero(~np.isnan(values_reshaped))>5:
        values_reshaped[1,values_reshaped_radius*2]=np.nan
    # print("edited array:")
    # print(values_reshaped)
    # print("median:")
    # print(np.nanmedian(values_reshaped))
    #kjbfjkf
    return np.nanmedian(values_reshaped) #this takes the median of the remaining 5 closest pixels

def ext_law(wavel, Av):
    x = 1./(wavel/1.e4)
    y = x - 1.82
    ax =1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    bx = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    Awavel = Av*(ax +bx/3.1)
    return Awavel	


def Av_smooth_means_interpolator(values,radius=None): #median function
    """
    used to interpolate Av values for NaN values in a row-by-row mean Av profile. Replaces nans with 0 if interpolation doesn't work
    """
    #following 5 lines reconstructs 2d array of input window of values
    #values_len=len(values)
    values_mod=values          
    values_reshaped=values_mod
    # print("values_array:")
    # print(values_reshaped)
    default_replacement_value=0

    if np.isnan(values_reshaped[radius])==True:
        for step in range(radius+1): 
            positive_x_neighbour=np.nan
            values_subsample=values_reshaped[radius:radius+step+1]
            # print("for step "+str(step)+":")
            # print(values_subsample)
            if np.isnan(values_subsample[-1])==False:
                positive_x_neighbour=values_subsample[-1]
                positive_x_neighbour_pos=step
                break
        if np.isnan(positive_x_neighbour)==False:
            for step in range(radius+1): 
                negative_x_neighbour=np.nan
                values_subsample=values_reshaped[radius-step:radius+1]
                # print("for step "+str(step)+":")
                # print(values_subsample)
                if np.isnan(values_subsample[0])==False: 
                    negative_x_neighbour=values_subsample[0]
                    negative_x_neighbour_pos=-step
                    break
            if np.isnan(negative_x_neighbour)==False:
                temp_grad=(positive_x_neighbour-negative_x_neighbour)/(positive_x_neighbour_pos-negative_x_neighbour_pos)
                c=(-positive_x_neighbour_pos*temp_grad)+positive_x_neighbour
                return c
            else:
                return default_replacement_value
        else:
            return default_replacement_value
    else:
        return values_reshaped[radius]
    
def dfisher_internal_extinction_correction(ha,hb,linedata,ltype="Halpha",output_data="default",av_map_smoothed=None,show_av_map=False,ha_err=None,hb_err=None,sigma5_cutting=True,radius=2,upper_percent=0.03,Av_mean_profile_interpolation_radius=3,use_avsmooth=True):
    """
    This function is used to perform an internal extinction correction, using the Balmer decrement, for maps of optical emission line fluxes of sub 100pc resolved galaxies. This
    function was primarially built for working with flux maps made for galaxies in the GECKOS sample, which have had a 3x3 spatial binning applied. The maps that this function
    was tested on were produced using the python package Threadcount, developed by Prof Deanne Fisher.

    This function works by creating an A_v map from Halpha and Hbeta flux maps supplied to it. Only pixels which 5sigma Hbeta and Halpha detections are kept for this map. The
    function then used procedures to approximate the values of the pixels missing in the A_v map. A fuller description of the function's procedures can be found in the "notes" section of this
    docstring.

    To Use this function for a galaxy, one must supply flux maps of the galaxy's Hbeta and Halpha emission lines to the first 2 positional arguments respectively. The 3rd
    positional argument must be the flux map that you wish to do an internal extinction correciton for, and the variable "ltype" must specify the name of the line that
    this map corresponds to (options for this are "Halpha","OIII_5007","N2_6583","N2_6548","SII_6716", and "SII_6730"). One can also specify the line's wavelength (Angstroms) for ltype
    if prefered. One must also supply the parameters "ha_err" and "ha_err", which are the arrays containing the uncertainties of the 1st and 2nd positional arguments ("ha" and "hb") respectively.
    All other parameters are set to ideal default values that only need to be changed if prefered
    
    parameters
    ----------------------
    ha: tc obj
        tc results array containing Halpha flux data
    hb: tc obj
        tc results array containing Hbeta flux data
    linedata: numpy array
        tc results array containing the line data you want to de-attenuate
    ltype: str
        name of line you want to de-attenuate (calls the line's value from the function "tc_line_centers"). Alternatively, you can give the value (in angstroms) of the line you want to de-attenuate.
    output_data: str
        defines the data output by this function. Entering "default" (i.e.: the default output) will return the unattenuated version of linedata. Entering "extinction maps" will
        return a list containing he unattenuated version of linedata, the Av map generated, and the A_line map generated. Entering "extensive" returns a list of many intermediate data products (only used for past testing/ not recommended for use)
    av_map_smoothed: numpy array
        insert a smoothed Av map here if you already have one (if not, leave this parameter as "None", which is its default)
    show_av_map: Bool
        displays live copy of Av map (default=False)
    ha_err: np array
        Halpha flux error array
    hb_err: np array
        Hbeta flux error array
    sigma5_cutting: bool
        if True (default), Hbeta and Halpha input data are clipped so that only flux/err>=5 pixels remain.
    radius: int
        This is a value used for filling in nans around the edges of existing data. To do this procedure, a function called "temp_median_function" is used, which replaces the nans with
        a median of the 5 nearest non-nan pixels: these 5 pixels can only be selected, at maximum, 1 row and x columns away from the current nan. this parameter is used to define x (default=2)
    upper_percent: float
        this parameter is used in producing a row-by-row Av mean profile from the input data. For each row (row index=n), we flag pixels in that row that sit within the top x% of
        A_v pixels in the row range of n-5 to n+5. These flagged pixels are excluded when defining the row-by-row Av mean profile. the x% is defined as a decimal my this "upper_percent" parameter (default=0.03) 
    Av_mean_profile_interpolation_radius: int
        when interpolating nan values in the created row-by-row mean Av profile, this is the maximum distance (in elements) to which one can extend from the current nan in order to look for
        adjacent elements to perform interpolation with (default=3). If neibours cannot be found on either side of the current nan within this extent, we replace the nan with a 0.
    use_avsmooth: bool
        indicates if the function avsmooth should be used on Av map made from the input data. (Default=True)

    returns
    -------------

    If output_data="default", this function returns an extinction corrected version of the 3rd positional input to this function.
    If output_data="extinction maps". this returns a list cotaining an extinction corrected version of the 3rd positional input to this function, the Av map generated, and the A_line map generated

    notes
    ------------------

    -The function begins by masking out all pixels from the first two positional input arrays that contain either Halpha or Hbeta fluxes at less than a 5sigma detection threshold.
    This leads to the calculation of an A_v map (using the equation 2.5/1.163*3.1*np.log10((input_data/input_data_Hb)/2.87)). We then perform some median smoothing for this
    A_v map using the function "Avsmooth": this involves some nan value replacement. The result of this is masked again to ensure that only the pixels what were 
    initially below the 5sigma threshold remain masked out. We are left now with an A_v map stored in the array called "Av_smooth".

    -we then perform a procedure that allows us to create a row-by-row Av map mean profile, which will be used to help replace nan vlaues in the current Av map. Before making this profile, we
    pick out pixels in the Av map that may have been impacted by unrecognized noise, and thus may lead to unphysical jumps in the Av map mean profile.
    we flag pixels in each row that sit within the top x% of A_v pixels in the row range of n-5 to n+5 (n=current row index). These flagged pixels are excluded 
    when defining the row-by-row Av mean profile, assuming that these pixels may be outliers in the n-5 to n+5 range. the x% is defined as a decimal my this "upper_percent" parameter (default=0.03). We also replace nans in this profile with
    values linearly interpolated between the pixel's nearest non-nan neighbours. Once this is done, we have a map ("Av_smooth_means_map") where
    each row contains only the mean A_v value of the corresponding row in the Av_smooth map

    -following this, we replace nan values nearby non-nan pixels in the array "Av_smooth" with approximated A_v values, ensuring that any gaps in the Av_smooth map exisiting
    due to spatial masking are closed. To do this procedure, a function called "temp_median_function" is used to create a new map from Av_smooth. This map takes each Av_smooth pixel, and makes a pixel in the new map
    with a median of the 5 nearest non-nan pixels (from Av_smooth): these 5 pixels can only be selected, at maximum, 1 row and x columns away from the current pixel. The parameter "radius" is used to define x (default=2).
    The result of this procedure is an A_v map stored under the variable name "Av_smooth_radial_median_smoothing".

    -To make a finalised copy of Av_smooth, we replace existing nan values with corresponding values from the map "Av_smooth_radial_median_smoothing". Any still remaining
    nan values are replaced with corresponding values from the map "Av_smooth_means_map". This leaves no remaining nans.

    -The Av_smooth is then convereted into an extinction map for the line specified in "ltype", using the function "ext_law". This allows us to apply equation "line_ext = linedata*10**(0.4*A_line)" to produce an extinction corrected version of "linedata" (the 3rd positional input for this
    function)

    credits
    -----------------

    -Thanks to Prof. Deanne Fisher for making the base version of this code, as well as the functions "avsmooth" and "ext_law".
    -The program "Threadcount" which was used to create the emission line flux maps that this function was tested on, can also be attributed to Deanne Fisher (https://threadcount.readthedocs.io/en/latest/api.html)
    """
    #Deane attenuation correction code moddified
    input_data_Hb = hb  #Hbeta flux data
    input_data = ha  #Halpha flux data

    if sigma5_cutting:

        #NEW 21/5/2024
        #below making masks for accepting only 5 sigma pixels
        sigma_clip_mask0=np.where(np.isnan(ha_err/input_data)==True,np.nan,1)
        sigma_clip_mask1=np.where(ha_err/input_data>1/5,np.nan,1)
        sigma_clip_mask2=np.where(hb_err/input_data_Hb>1/5,np.nan,1)
        sigma_clip_mask3=np.where(np.isnan(hb_err/input_data_Hb)==True,np.nan,1)

        #NEW 21/5/2024
        #non_sigma5_pixels=sigma_clip_mask0*(np.where(np.isnan(sigma_clip_mask1)==True,1,np.nan))*(np.where(np.isnan(sigma_clip_mask2)==True,1,np.nan))*sigma_clip_mask3
        sigma5_pixels=sigma_clip_mask0*sigma_clip_mask1*sigma_clip_mask2*sigma_clip_mask3

        #NEW 21/5/2024
        #performing the sigma clipping
        input_data_Hb=input_data_Hb*sigma5_pixels
        input_data=input_data*sigma5_pixels

    if type(av_map_smoothed)==type(None): #if a smoothed_av_map hasn't already been given to this de-extinction function.
        Av_all = 2.5/1.163*3.1*np.log10((input_data/input_data_Hb)/2.87)

        if use_avsmooth:
            Av_smooth = avsmooth(Av_all)
            print(use_avsmooth)
        else:
            Av_smooth = Av_all

        #NEW 21/5/2024
        #once again, we ensure that only 5sigma pixels show up in the Av_smooth map
        Av_smooth=sigma5_pixels*Av_smooth
        Av_smooth_5sigma_selection_only=np.copy(Av_smooth)

        #below, we cut Av_smooth into 3 arrays: an array of the first fiew rows of data where all we have are nans (Av_smooth_cut_start), an array of the 
        # last fiew rows of data where all we have are nans (Av_smooth_cut_start), and the rows that fit between these (Av_smooth_cut)
        non_nan_pixels_per_row=np.sum(np.where(np.isnan(Av_smooth)==True,0,1),axis=1)
        pix_index1=np.where(non_nan_pixels_per_row>0)[0][0]
        pix_index2=np.where(non_nan_pixels_per_row>0)[0][-1]
        Av_smooth_cut=Av_smooth[pix_index1:pix_index2+1,:]
        Av_smooth_cut_start=Av_smooth[0:pix_index1,:]
        Av_smooth_cut_end=Av_smooth[pix_index2+1:Av_smooth.shape[0]+1,:]

        #the following block of code identifies pixels in Av_smooth_cut_end that should be flagged to be unused when calculating a row-by-row mean Av profile.
        Av_smooth_cut_mask=Av_smooth_cut.copy()
        for row_ind in range(Av_smooth_cut.shape[0]): #for each row in our current Av data
            temp_min_index=row_ind-5
            temp_max_index=row_ind+5+1
            #print(row_ind)
            if temp_min_index<0:
                temp_min_index=0
            if temp_max_index>Av_smooth_cut.shape[0]:
                temp_max_index=Av_smooth_cut.shape[0]
                #print("no")
            current_Av_smooth_cut_row=Av_smooth_cut[row_ind,:]#this stores our current row we're looking at
            current_Av_smooth_cut_region=Av_smooth_cut[temp_min_index:temp_max_index,:]#we select up to 5 rows aroud our current row (unless we're at our lowest or highest row index)
            region_list=np.reshape(current_Av_smooth_cut_region,current_Av_smooth_cut_region.shape[0]*current_Av_smooth_cut_region.shape[1])
            region_list=np.where(np.isnan(region_list)==True,0,region_list)#we convert nans to 0s.
            if len(region_list)!=0: #if we're not looking at a region with only NaN values...
                region_list=np.sort(region_list)
                #print(len(region_list))
                n_pixels_in_upper_percent=math.ceil(len(region_list)*upper_percent)#how many pixels would be included in a top x% of this region? (x=upper_percent)
                #print(n_pixels_in_upper_percent)
                #print(region_list)
                region_list_lower_values=region_list[0:-n_pixels_in_upper_percent]#cut out top x% of pixels
                #print(region_list_lower_values)
                if len(region_list_lower_values)>0:
                    temp_upper_av_value_threshold=np.nanmax(region_list_lower_values)#what is the threshold above which the top x% of the region's pixels occur for?
                    #print(temp_upper_av_value_threshold)
                    #print(current_Av_smooth_cut_row)
                    current_Av_smooth_cut_row_mask=np.where(current_Av_smooth_cut_row<=temp_upper_av_value_threshold,1,np.nan) #we turn our current row into a row for our now mask
                    Av_smooth_cut_mask[row_ind,:]=current_Av_smooth_cut_row_mask# we assign our new mask row to our mask array
                    #print(current_Av_smooth_cut_row_mask)
                else:
                    Av_smooth_cut_mask[row_ind,:]=current_Av_smooth_cut_row*np.nan
            else:
                Av_smooth_cut_mask[row_ind,:]=current_Av_smooth_cut_row*np.nan

        #we take the mask made in the previous steps, and stitch them together with Av_smooth_cut_start and Av_smooth_cut_end to make a full mask
        Av_smooth_mask_for_row_average_taking=np.append(np.append(Av_smooth_cut_start,Av_smooth_cut_mask,axis=0),Av_smooth_cut_end,axis=0)

        #we take a mean value profile of Av_smooth's rows (stored in "Av_smooth_means"), whilst also applying the "Av_smooth_mask_for_row_average_taking" mask we made just prior. This ensures that
        #the average of each row is not notably affected by pixels with very high Av values.
        Av_smooth_means=np.nanmean(Av_smooth*Av_smooth_mask_for_row_average_taking,axis=1)
        Av_smooth_means_map=Av_smooth.copy()

        #the following is used to interpolate nan values in "Av_smooth_means",
        Av_smooth_means_interpolator_footprint_radius=Av_mean_profile_interpolation_radius
        Av_smooth_means_interpolator_footprint=np.array(([1]*Av_mean_profile_interpolation_radius)+[1]+([1]*Av_mean_profile_interpolation_radius))
        Av_smooth_means_interpolated=ndimage.filters.generic_filter(Av_smooth_means,Av_smooth_means_interpolator,cval=np.NaN,mode="constant",footprint=Av_smooth_means_interpolator_footprint,extra_keywords={"radius":Av_smooth_means_interpolator_footprint_radius})
        Av_smooth_means=Av_smooth_means_interpolated
        for ele in range(len(Av_smooth_means)):
            #if np.nansum(np.where(np.isnan(Av_smooth[ele,:])==True,np.nan,1))>2: #this means we only add a row to our median map if that median is based on more than 2 pixels
            Av_smooth_means_map[ele,:]=Av_smooth_means[ele]


        #following that, we employ a function that replaces each pixel with the median value of pixels within a 5 pixel radius (ie: the median value of, at most, a 3 row X 11 column
        #square around the target pixel that doesn't include the target pixel in the calculation). The output is "Av_smooth_radial_median_smoothing"
        # def temp_median_function(values): #median function
        #     return np.nanmedian(values)
        generic_foorprint_shape=(2*radius)+1
        footprint1=np.ones([3,generic_foorprint_shape])
        footprint1[1,radius]=0 #footprint indicates the region around each pixel that a median should be calculated from (the zero at the center
        #represents the target pixel)
        Av_smooth_old=Av_smooth.copy()
        Av_smooth_radial_median_smoothing=ndimage.filters.generic_filter(Av_smooth, temp_median_function,cval=np.NaN,mode="constant",footprint=footprint1,extra_keywords={"values_shape":footprint1.shape,"radius":int(radius)})
        
        #now with our "Av_smooth_radial_median_smoothing" array, we replace nan values from our original Av_smooth map
        Av_smooth_intermediate=np.where(np.isnan(Av_smooth)==True,Av_smooth_radial_median_smoothing,Av_smooth)
        #we replace remaining NaN values with values from "Av_smooth_means_map"
        Av_smooth_intermediate_2=np.where(np.isnan(Av_smooth_intermediate)==True,Av_smooth_means_map,Av_smooth_intermediate)
        #Av_smooth_intermediate_2=np.where(np.isnan(Av_smooth_intermediate_2)==True,0,Av_smooth_intermediate_2)#this line replaces 
        Av_smooth=Av_smooth_intermediate_2

        #see where nan pixels are
        #make median map/ remember where nan pixels are 

    else: #if we have an Av_smooth map supplied as an input to this function (using the parameter "av_map_smoothed")....
        Av_smooth=av_map_smoothed


    if type(ltype)==type("string"):
        ltype_used=tc_line_centers(version="2.0")[ltype]
    else:
        ltype_used=ltype
    H_line = ext_law(ltype_used,Av_smooth)

    line_ext = linedata*10**(0.4*H_line)

    full_flux_map=line_ext


    if show_av_map:
        plt.figure()
        plt.imshow(Av_smooth)
        plt.colorbar()
        plt.show()

    if output_data=="default":
        output_data=full_flux_map
    if output_data=="extinction maps":
        output_data=[full_flux_map,Av_smooth,H_line]
    if output_data=="extensive":
        output_data=[full_flux_map,Av_smooth,H_line,sigma5_pixels,Av_smooth_radial_median_smoothing,Av_smooth_old,Av_smooth_means_map,Av_smooth_mask_for_row_average_taking,Av_smooth_5sigma_selection_only]
    return output_data