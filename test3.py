import pandas as pd
import cv2
import pytesseract
import re
import os
import requests
from PIL import Image
from io import BytesIO

# Define allowed units (for demonstration; refer to constants.py for actual units)
allowed_units = {
    "item_weight": ["gram", "kilogram", "ounce"],
    "item_dimensions": ["centimetre", "inch", "millimetre"],
    "item_volume": ["litre", "millilitre", "gallon"],
    # Add more entity types and corresponding units
}

# Download image from URL
def download_image(image_url, save_path="images"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    img_name = os.path.join(save_path, image_url.split("/")[-1])
    response = requests.get(image_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img.save(img_name)
        return img_name
    else:
        print(f"Failed to download image: {image_url}")
        return None

# Extract text from the image using OCR
def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Function to extract entity values using regex
def extract_entity_value(text, entity_name):
    entity_type = entity_name.split('_')[-1]  # Get entity type (e.g., weight, dimensions)
    if entity_name == "item_weight":
        match = re.search(r'(\d+(\.\d+)?)\s?(gram|kilogram|ounce)', text)
        if match and match.group(3) in allowed_units[entity_name]:
            return f"{match.group(1)} {match.group(3)}"
    elif entity_name == "item_dimensions":
        match = re.search(r'(\d+(\.\d+)?)\s?(centimetre|inch|millimetre)', text)
        if match and match.group(3) in allowed_units[entity_name]:
            return f"{match.group(1)} {match.group(3)}"
    elif entity_name == "item_volume":
        match = re.search(r'(\d+(\.\d+)?)\s?(litre|millilitre|gallon)', text)
        if match and match.group(3) in allowed_units[entity_name]:
            return f"{match.group(1)} {match.group(3)}"
    
    # Return empty string if no valid entity value is found
    return ""

# Load the training and test datasets
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

# Initialize predictions list
predictions = []

# Loop through each test record and extract entity values
for index, row in test_data.iterrows():
    image_url = row['image_link']
    entity_name = row['entity_name']
    
    # Download the image
    image_path = download_image(image_url)
    
    # Skip if image download fails
    if image_path is None:
        predictions.append([row['index'], ""])
        continue
    
    # Extract text from the image using OCR
    text = extract_text_from_image(image_path)
    
    # Extract the entity value based on the entity name
    entity_value = extract_entity_value(text, entity_name)
    
    # Append prediction in the format required
    predictions.append([row['index'], entity_value])

# Create the output DataFrame
output_df = pd.DataFrame(predictions, columns=['index', 'prediction'])

# Save the final output to CSV
output_df.to_csv('my_test_out.csv', index=False)

# Perform sanity check
# Assuming sanity.py is in the same directory as the script
os.system("python src/sanity.py --file my_test_out.csv")