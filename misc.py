import os

openslide_path = r"C:\Users\sarah\Documents\openslide-win64-20230414\bin"
os.environ['PATH'] = openslide_path + ";" + os.environ['PATH']
os.add_dll_directory(openslide_path)
import openslide

def get_base_magnification(wsi_img):
    try:
        base_magnification = float(wsi_img.properties['openslide.objective-power'])
    except KeyError:
        try:
            mpp_x = float(wsi_img.properties['openslide.mpp-x'])
            mpp_y = float(wsi_img.properties['openslide.mpp-y'])
            # The pixel size at 1x magnification is typically around 0.25 micrometers
            pixel_size_at_1x = 0.25
            base_magnification = 1 / (max(mpp_x, mpp_y) * pixel_size_at_1x)
        except KeyError:
            try:
                highest_res_dim = max(wsi_img.level_dimensions[0])
                lowest_res_dim = max(wsi_img.level_dimensions[-1])
                base_magnification = highest_res_dim / lowest_res_dim
            except:
                return None
    return base_magnification