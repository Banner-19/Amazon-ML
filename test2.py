import requests
import pandas as pd
import pytesseract
from PIL import Image
from io import BytesIO
# import sys
# sys.path.append('./src')
from utils import download_images, parse_string

# Load the CSV files
test_df = pd.read_csv('dataset/sample_test.csv')

# Function to download image from URL
def download_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

# Function to extract text from image using OCR (Tesseract)
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Logic to extract entity value based on entity name and allowed units
# def extract_entity_value(extracted_text, entity_name):
#     extracted_value = None

#     # Retrieve the valid units for the current entity
#     valid_units = entity_unit_map.get(entity_name, set())

#     # Regex pattern to match any value followed by a valid unit
#     unit_pattern = '|'.join(re.escape(unit) for unit in valid_units)  # Escape units for regex
#     pattern = re.compile(rf'(\d+\.?\d*)\s*({unit_pattern})')

#     # Find matches in the extracted text
#     matches = pattern.findall(extracted_text.lower())

#     if matches:
#         extracted_value = f"{float(matches[0][0])} {matches[0][1]}"  # Extract the first valid match
    
#     return extracted_value

# Create an empty list to store predictions
predictions = []

# Download and process images
image_links = test_df['image_link']
download_images(image_links, 'downloaded_images')

# Loop through the test.csv to download and process each image
for index, row in test_df.iterrows():
    image_url = row['image_link']
    entity_name = row['entity_name']
    
    # Download the image
    image = download_image(image_url)
    
    # Extract text from the image
    extracted_text = extract_text_from_image(image)
    
    # Apply logic to extract the specific entity value based on allowed units
    try:
        # Parse the extracted text using the parse_string function
        number, unit = parse_string(extracted_text)
        if number and unit:
            entity_value = f"{number} {unit}"
        else:
            entity_value = 'Not Found'
    except ValueError:
        entity_value = 'Not Found'

    # Append results to predictions list
    predictions.append({'index': index, 'prediction': entity_value})

# Convert the list of predictions to a DataFrame
predictions_df = pd.DataFrame(predictions)

# Save the predictions to a CSV file
predictions_df.to_csv('predictions1.csv', index=False)
