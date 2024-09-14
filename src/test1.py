import requests
import pandas as pd
import pytesseract
from PIL import Image
from io import BytesIO
import easyocr
import spacy
from utils import download_images, parse_string  # Assuming you have these implemented
from constants import entity_unit_map
import re

# Load the test CSV file
test_df = pd.read_csv('dataset/sample_test.csv')

# Initialize EasyOCR and spaCy models
reader = easyocr.Reader(['en'])
nlp = spacy.load("en_core_web_sm")

# Function to download an image from a URL
def download_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

# Function to extract text using EasyOCR
def extract_text_with_easyocr(image):
    results = reader.readtext(image, detail=0)
    return ' '.join(results)  # Join all the extracted text

# Function to detect numerical entities and units using spaCy NER
def extract_entities_with_spacy(text):
    doc = nlp(text)
    entity_values = []
    for ent in doc.ents:
        if ent.label_ == 'QUANTITY':  # Use QUANTITY to extract numeric values with units
            entity_values.append(ent.text)
    return entity_values

# Function to extract entity value based on entity name and allowed units using regex
def extract_entity_value(extracted_text, entity_name):
    extracted_value = None
    valid_units = entity_unit_map.get(entity_name, set())
    unit_pattern = '|'.join(re.escape(unit) for unit in valid_units)  # Escape units for regex
    pattern = re.compile(rf'(\d+\.?\d*)\s*({unit_pattern})')

    matches = pattern.findall(extracted_text.lower())
    if matches:
        extracted_value = f"{float(matches[0][0])} {matches[0][1]}"  # Extract first valid match
    return extracted_value

if __name__ == "__main__":
    # Create an empty list to store predictions
    predictions = []

    # Download and process images
    image_links = test_df['image_link']
    download_images(test_df['image_link'], 'downloaded_images')  # Assuming download_images is implemented

    # Loop through the test.csv to download and process each image
    for index, row in test_df.iterrows():
        image_url = row['image_link']
        entity_name = row['entity_name']
        
        # Download the image
        image = download_image(image_url)
        
        # Extract text using EasyOCR
        extracted_text = extract_text_with_easyocr(image)
        
        # Apply spaCy NER to detect entity and unit
        extracted_entities = extract_entities_with_spacy(extracted_text)
        
        # If no entities detected by NER, use regex-based extraction
        if not extracted_entities:
            entity_value = extract_entity_value(extracted_text, entity_name)
        else:
            entity_value = extracted_entities[0] if extracted_entities else 'Not Found'

        # Try to parse the entity value further using parse_string function (for additional logic)
        try:
            number, unit = parse_string(entity_value)
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
    predictions_df.to_csv('predictions_with_ml.csv', index=False)
