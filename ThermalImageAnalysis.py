import os
from PIL import Image
import pandas as pd 

folderPath = "C:\\Users\\tvrj\\OneDrive - Danmarks Tekniske Universitet\\Dokumenter\\Arbejde\\Data\\ThermalImages\\"
folders = ["2018.02.08_I0-1_DC_NUCF_5-12-8","2018.02.08_I0-3_DC_NUCF_5-12-8","2019.10.30_3P5-0_DC_2-0-0","2019.10.30_3P5-1_DC_2-0-0","2021.03.17_2P10-0_MIN4","2021.03.17_2P11-0_MIN4","2021.03.17_2P12-0_MIN4","2021.03.17_2P13-0_MIN4","2024.02.08_1P3-0_MIN4","2024.02.08_1P4-0_MIN4","2024.02.08_1P5-0_MIN4"]
testfolder = ["Tester"]

def analyze_images(folder):
    folder_path = folderPath+folder
    # List to store names of images with at least 3 non-dark pixels in a row, excluding boundary pixels
    images_with_colors = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            # Open the image file
            img = Image.open(os.path.join(folder_path, filename))
            # Convert image to RGB mode if not already
            img = img.convert("RGB")
            # Get image data
            data = img.load()
            width, height = img.size
            bright_pixels_found = False
            # Check for at least 3 non-dark pixels in a row in the image, excluding boundary pixels
            for y in range(1, height - 1):
                non_dark_count = 0
                for x in range(1, width - 1):
                    r, g, b = data[x, y]
                    if r > 150 or g > 150 or b > 150:  # Higher threshold for significantly bright colors
                        non_dark_count += 1
                        if non_dark_count >= 3:
                            bright_pixels_found = True
                            break
                    else:
                        non_dark_count = 0
                if bright_pixels_found:
                    break

            if bright_pixels_found:
                # Count all colors in the image
                color_count = {}
                for y in range(height):
                    for x in range(width):
                        r, g, b = data[x, y]
                        color = (r, g, b)
                        if color in color_count:
                            color_count[color] += 1
                        else:
                            color_count[color] = 1
                images_with_colors.append((filename, color_count))

    # Create a DataFrame to store the results
    dic = {"Image Name":[], "Color":[], "Count":[]}
    # df = pd.DataFrame(columns=["Image Name", "Color", "Count"])

    # Populate the DataFrame with the results
    for image_name, colors in images_with_colors:
        for color, count in colors.items():
            dic["Image Name"].append(image_name)
            dic["Color"].append(color)
            dic["Count"].append(count)
    df = pd.DataFrame(dic)
    # Export the DataFrame to an Excel file
    df.to_excel(folderPath+folder+".xlsx", index=False)

# for folder in folders:
#     analyze_images(folder)

def tiff2PNG(folder): 
    folder_path = folderPath+folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.tiff')):
            # Open the TIFF image
            tiff_image = Image.open(os.path.join(folder_path, filename))
            imagename = filename[:-5]
            # frames_in_image = getattr(tiff_image, "n_frames", 1)
            # print(tiff_image.info)

            # Convert the image to JPEG format
            png_image = tiff_image.convert("RGB")

            # Save the JPEG image
            png_image.save("{}\\{}.png".format(folder_path,imagename))
            # break

tiff2PNG(folders[1])

def show_tiff_image(file_path):
    # Open the image file
    with open(file_path, 'rb') as f:
        image_data = f.read()
    
    # Print the source code of the image
    print(image_data)
    
    # Display the image
    img = Image.open(file_path)
    img.show()

# show_tiff_image(folderPath+folders[1]+"\\2018.02.08_I0-3_DC_NUCF_2973 - 1.PNG")
