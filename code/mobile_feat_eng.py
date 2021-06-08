def feat_eng(data):
    """
    Create new features combining several existing features
    """
    
    # Create a new variable 'generation' by combining 'three_g' and 'four_g'
    data["generation"] = ""
    for i in range(data.shape[0]):
        if data["three_g"][i] == 1 and data["four_g"][i] == 1:
            data["generation"][i] = 2   # 4G
        elif data["three_g"][i] == 1 and data["four_g"][i] == 0:
            data["generation"][i] = 1   # 3G
        else:
            data["generation"][i] = 0   # 2G
    
    # Create a new variable 'pixel_dimension' by multiplying 'px_width' and 'px_height'
    data["pixel_dimension"] = data["px_width"] * data["px_height"]
    
    # Create a new variable 'screen_dimension' by multiplying 'sc_w' and 'sc_h'
    data["screen_dimension"] = data["sc_w"] * data["sc_h"]
    
    # Create a new variable 'camera_pixels' by adding 'fc' and 'pc'
    data["camera_pixels"] = data["fc"] + data["pc"]