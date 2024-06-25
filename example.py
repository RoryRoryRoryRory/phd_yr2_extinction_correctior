import Av_generator
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib as mpl
mpl.rcParams["image.interpolation"]="None"

#this script will perform a Halpha emission line flux map internal extinction correction for galaxy ESO079-003 using the Balmer decrement.

#the following lines import our data for Halpha emission line flux maps (ha) and Halpha emission line flux maps (hb).
#ensure that the files listed below are in the directory that you run this script from.
#also ensure that the files Av_generator.py is in the directory that you run this script from: this file
#contains the functions needed for extinction correctiom, which are imported using the line "import Av_generator"
data_ha=fits.open("ESO079_003_Halpha_bin3x3.fits")
data_hb=fits.open("ESO079_003_hbeta_bin3x3.fits")
ha=data_ha["FLUX"].data
ha_err=data_ha["ERRFLUX"].data
hb=data_hb["FLUX"].data
hb_err=data_hb["ERRFLUX"].data

#the inputs for the above lines must be fits files


#below is an example usage of the internal extinction correction code. The 3rd positional argument is
#the data we want to do an extinction correction for, which is the Halpha emssion line flux array. We
#specify that this is the case using the input for "ltype".
ha_corrected=Av_generator.dfisher_internal_extinction_correction(ha,hb,ha,ltype="Halpha",ha_err=ha_err,hb_err=hb_err)
    
#note that that the following line has the same function as the one above
#ha_corrected=Av_generator.dfisher_internal_extinction_correction(ha,hb,ha,ltype=6562,ha_err=ha_err,hb_err=hb_err)

#you can look at the docstring of the above function using the following line of code
print(Av_generator.dfisher_internal_extinction_correction.__doc__)

#we can also get the A_v map and ltype specific extinction map (A_line) map using the following command
[ha_corrected,A_v,A_line]=Av_generator.dfisher_internal_extinction_correction(ha,hb,ha,ltype="Halpha",output_data="extinction maps",ha_err=ha_err,hb_err=hb_err)

#let's apply a mask to each result, based on where we have nans in our input Halpha array.
mask=np.where(np.isnan(ha)==True,np.nan,1)
ha_corrected=ha_corrected*mask
A_v=A_v*mask
A_line=A_line*mask

#we now plot the various produced maps
plt.figure()
plt.imshow(A_v,vmax=2,vmin=0,cmap="inferno",origin="lower")
cbar_temp=plt.colorbar()
cbar_temp.set_label("Av map",size="large")
plt.title("ESO079-003 Av map")
plt.show()

plt.figure()
plt.imshow(A_line,vmax=2,vmin=0,cmap="inferno",origin="lower")
cbar_temp=plt.colorbar()
cbar_temp.set_label("A_line map",size="large")
plt.title("ESO079-003 A_line map")
plt.show()

plt.figure()
plt.imshow(np.log10(ha_corrected),vmin=0.62,vmax=2.66,cmap="inferno",origin="lower")
cbar_temp=plt.colorbar()
s_alpha=chr(945)
cbar_temp.set_label(r"$log_{10}$($F_{H"+s_alpha+"} $)   "+"[$10^{-20}\u212B^{-1}cm^{-2}s^{-1}erg$]",size="large")
plt.title("ESO079-003 corrected Halpha flux")
plt.show()