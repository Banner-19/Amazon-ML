import pandas as pd
import requests
from io import BytesIO
import re
import pytesseract
from PIL import Image
import keras_ocr
from constants import entity_unit_map

# Define short-to-full form unit mappings
short_to_full_unit_map = {
    'cm': 'centimetre',
    'm': 'metre',
    'mm': 'millimetre',
    'ft': 'foot',
    'in': 'inch',
    'yd': 'yard',
    'g': 'gram',
    'kg': 'kilogram',
    'mg': 'milligram',
    'ug': 'microgram',
    'oz': 'ounce',
    'lb': 'pound',
    't': 'ton',
    'kv': 'kilovolt',
    'v': 'volt',
    'w': 'watt',
    'kw': 'kilowatt',
    'ml': 'millilitre',
    'l': 'litre',
    'cl': 'centilitre',
    'dl': 'decilitre',
    'cu ft': 'cubic foot',
    'cu in': 'cubic inch',
    'fl oz': 'fluid ounce',
    'gal': 'gallon',
    'pt': 'pint',
    'qt': 'quart'
}

# Function to load image from a URL
def load_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None

# Function to extract text from image using OCR (Tesseract)
def extract_text_from_image_tesseract(image):
    if image is None:
        return ""
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Error during Tesseract OCR: {e}")
        return ""

# Function to extract text from image using OCR (Keras)
def extract_text_keras(image_url):
    try:
        pipeline = keras_ocr.pipeline.Pipeline()
        image = keras_ocr.tools.read(image_url)
        prediction_groups = pipeline.recognize([image])
        extracted_text = ' '.join([word for word, box in prediction_groups[0]])
        return extracted_text
    except Exception as e:
        print(f"Error using Keras OCR: {e}")
        return ""

# Combined extraction logic (Tesseract -> fallback to Keras)
def extract_text(image_url):
    if not image_url:
        print("No image URL provided for extraction.")
        return ""
    
    # Extract text using Tesseract
    image = load_image_from_url(image_url)
    extracted_text = extract_text_from_image_tesseract(image)
    
    # If Tesseract doesn't return text, fallback to Keras OCR
    if not extracted_text.strip():
        extracted_text = extract_text_keras(image_url)
    
    return extracted_text

# Logic to extract entity value based on entity name and allowed units
def extract_entity_value(extracted_text, entity_name):
    extracted_value = None
    extracted_text = extracted_text.lower()

    # Retrieve the valid units for the current entity
    valid_units = entity_unit_map.get(entity_name, set())

    # Replace short units with full form
    for short_unit, full_unit in short_to_full_unit_map.items():
        extracted_text = extracted_text.replace(short_unit, full_unit)

    # Regex pattern to match dimensions and single values with valid units
    unit_pattern = '|'.join(re.escape(unit) for unit in valid_units)
    # Pattern to capture both dimensions and single values
    pattern = re.compile(rf'(\d+)\*?(\d*)?\s*({unit_pattern})', re.IGNORECASE)

    # Find matches in the extracted text
    matches = pattern.findall(extracted_text)

    if matches:
        # If there are dimensions, assume the first dimension could be height
        for match in matches:
            height = match[0]  # First dimension
            unit = match[2]
            extracted_value = f"{height} {unit}"
            break  # Return the first match
    
    return extracted_value

# Function to process input CSV and generate output CSV
def process_csv(input_csv_path, output_csv_path):
    # Load the input CSV
    df = pd.read_csv(input_csv_path)
    
    # Initialize lists for output data
    indices = []
    predictions = []
    
    for index, row in df.iterrows():
        image_url = row['image_link']
        entity_name = row['entity_name']
        
        # Extract text from the image
        extracted_text = extract_text(image_url)
        print(f"Extracted Text: {extracted_text}")
        
        # Extract the entity value
        entity_value = extract_entity_value(extracted_text, entity_name)
        indices.append(row['index'])
        predictions.append(entity_value if entity_value else "Entity value could not be extracted.")
    
    # Create a DataFrame for the output CSV
    output_df = pd.DataFrame({
        'index': indices,
        'predictions': predictions
    })
    
    # Save to CSV
    output_df.to_csv(output_csv_path, index=False)
    print(f"Output CSV saved to {output_csv_path}")

# Example usage
if __name__ == "__main__":
    input_csv_path = "dataset/sample_test.csv"  # Replace with your input CSV path
    output_csv_path = "src/output.csv"  # Replace with your desired output CSV path
    process_csv(input_csv_path, output_csv_path)
