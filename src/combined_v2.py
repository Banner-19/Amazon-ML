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

# Function to load image from a local path
def load_image(image_path):
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
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
def extract_text_keras(image_path):
    try:
        pipeline = keras_ocr.pipeline.Pipeline()
        image = keras_ocr.tools.read(image_path)
        prediction_groups = pipeline.recognize([image])
        extracted_text = ' '.join([word for word, box in prediction_groups[0]])
        return extracted_text
    except Exception as e:
        print(f"Error using Keras OCR: {e}")
        return ""
# Combined extraction logic (Tesseract -> fallback to Keras)
def extract_text(image):
    if image is None:
        print("No image provided for extraction.")
        return ""
    
    extracted_text = extract_text_from_image_tesseract(image)
    if not extracted_text.strip():
        extracted_text = extract_text_keras(image)
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

    # Regex pattern to match any value followed by a valid unit
    unit_pattern = '|'.join(re.escape(unit) for unit in valid_units)
    pattern = re.compile(rf'(\d+\.?\d*)\s*({unit_pattern})')

    # Find matches in the extracted text
    matches = pattern.findall(extracted_text)

    if matches:
        extracted_value = f"{float(matches[0][0])} {matches[0][1]}"  # Extract the first valid match
    
    return extracted_value

# Example test function for a single image path
def test_entity_extraction(image_path, entity_name):
    # Load the image from the local path
    # image = load_image(image_path)
    # if image is None:
    #     print("Failed to load image.")
    #     return None
    
    # Extract text from the image
    extracted_text = extract_text(image_path)
    print(f"Extracted Text: {extracted_text}")
    
    # Apply logic to extract the specific entity value based on allowed units
    entity_value = extract_entity_value(extracted_text, entity_name)
    
    if entity_value:
        print(f"Final entity value: {entity_value}")
    else:
        print("Entity value could not be extracted.")
    return entity_value

# Test with an example
if __name__ == "__main__":
    image_path = "images1/61C+fwVD6dL.jpg"  # Replace with your image path
    entity_name = "width"  # Replace with the relevant entity you're testing
    test_entity_extraction(image_path, entity_name)
