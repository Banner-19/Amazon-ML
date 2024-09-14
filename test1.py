import requests
import pandas as pd
import pytesseract
from PIL import Image
from io import BytesIO
import re

# Load the CSV files
test_df = pd.read_csv('dataset\sample_test.csv')

# Allowed units from the Appendix
allowed_units = ['cm', 'inches', 'inch', 'mm', 'meter', 'g', 'kg', 'gram', 'milligram', 'ounce', 'lb', 'pound', 'ml', 'l', 'liter', 'cup']

# Function to download image from URL
def download_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

# Function to extract text from image using OCR (Tesseract)
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Logic to extract entity value based on entity name
def extract_entity_value(extracted_text, entity_name):
    extracted_value = None

    # Logic for height, width, depth (common dimensions)
    if entity_name in ['height', 'width', 'depth']:
        pattern = re.compile(r'(\d+\.?\d*)\s*(cm|inches|inch|mm|meter)')
        matches = pattern.findall(extracted_text.lower())
        if matches:
            extracted_value = f"{float(matches[0][0])} {matches[0][1]}"
    
    # Logic for item weight
    elif entity_name == 'item_weight':
        pattern = re.compile(r'(\d+\.?\d*)\s*(g|kg|gram|milligram|ounce|lb|pound)')
        matches = pattern.findall(extracted_text.lower())
        if matches:
            extracted_value = f"{float(matches[0][0])} {matches[0][1]}"
    
    # Logic for volume (if applicable)
    elif entity_name == 'item_volume':
        pattern = re.compile(r'(\d+\.?\d*)\s*(ml|l|liter|cup)')
        matches = pattern.findall(extracted_text.lower())
        if matches:
            extracted_value = f"{float(matches[0][0])} {matches[0][1]}"

    # Logic for maximum weight recommendation
    elif entity_name == 'maximum_weight_recommendation':
        pattern = re.compile(r'(\d+\.?\d*)\s*(g|kg|pound|lb|gram)')
        matches = pattern.findall(extracted_text.lower())
        if matches:
            extracted_value = f"{float(matches[0][0])} {matches[0][1]}"
    
    return extracted_value

# Create an empty list to store predictions
predictions = []

# Loop through the test.csv to download and process each image
for index, row in test_df.iterrows():
    image_url = row['image_link']
    entity_name = row['entity_name']
    
    # Download the image
    image = download_image(image_url)
    
    # Extract text from the image
    extracted_text = extract_text_from_image(image)
    
    # Apply logic to extract the specific entity value
    entity_value = extract_entity_value(extracted_text, entity_name)
    
    # Ensure the entity_value is in the required format
    if entity_value:
        predictions.append({'index': index, 'prediction': entity_value})
    else:
        predictions.append({'index': index, 'prediction': 'Not Found'})

# Convert the list of predictions to a DataFrame
predictions_df = pd.DataFrame(predictions)

# Save the predictions to a CSV file
predictions_df.to_csv('predictions.csv', index=False)
