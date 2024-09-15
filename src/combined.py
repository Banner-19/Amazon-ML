import pytesseract
from PIL import Image
import keras_ocr
import re
from constants import entity_unit_map  # Assuming you have the entity_unit_map

# Step 1: Create a short-to-full form unit mapping
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
    'ton': 'ton',
    'v': 'volt',
    'kv': 'kilovolt',
    'mv': 'millivolt',
    'w': 'watt',
    'kw': 'kilowatt',
    'ml': 'millilitre',
    'l': 'litre',
    'cl': 'centilitre',
    'dl': 'decilitre',
    'cup': 'cup',
    'pt': 'pint',
    'qt': 'quart',
    'fl oz': 'fluid ounce',
    'gal': 'gallon',
    'ig': 'imperial gallon',
    'cf': 'cubic foot',
    'ci': 'cubic inch'
}

# Function to replace short-form units with full forms in extracted text
def replace_short_units_with_full(text):
    for short_form, full_form in short_to_full_unit_map.items():
        text = re.sub(rf'\b{re.escape(short_form)}\b', full_form, text, flags=re.IGNORECASE)
    return text

# Function to extract text using Tesseract OCR
def extract_text_tesseract(image_path):
    try:
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image).strip()
        return extracted_text
    except Exception as e:
        print(f"Error using Tesseract: {e}")
        return ""

# Function to extract text using Keras OCR
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

# Function to extract entity value based on entity name and allowed units
def extract_entity_value(extracted_text, entity_name):
    extracted_value = None

    # Replace short-form units with full forms in extracted text
    preprocessed_text = replace_short_units_with_full(extracted_text)

    # Retrieve the valid units for the current entity
    valid_units = entity_unit_map.get(entity_name, set())

    # Regex pattern to match any value followed by a valid unit in full form
    unit_pattern = '|'.join(re.escape(unit) for unit in valid_units)  # Escape units for regex
    pattern = re.compile(rf'(\d+\.?\d*)\s*({unit_pattern})', re.IGNORECASE)

    # Find matches in the preprocessed text
    matches = pattern.findall(preprocessed_text.lower())

    if matches:
        extracted_value = f"{float(matches[0][0])} {matches[0][1]}"  # Extract the first valid match
    
    return extracted_value

# Main function to combine both OCR methods and extract entity value
def extract_text_and_entity(image_path, entity_name):
    # Step 1: Try Tesseract OCR
    extracted_text = extract_text_tesseract(image_path)
    print(f"Text extracted using Tesseract: '{extracted_text}'")
    
    # Step 2: If Tesseract fails (empty text), fallback to Keras OCR
    if not extracted_text:
        print("Tesseract returned empty text. Falling back to Keras OCR...")
        extracted_text = extract_text_keras(image_path)
        print(f"Text extracted using Keras: '{extracted_text}'")
    
    # Step 3: Extract entity value based on the extracted text and entity name
    entity_value = extract_entity_value(extracted_text, entity_name)
    
    if entity_value:
        print(f"Entity value extracted: '{entity_value}'")
    else:
        print("Entity value could not be extracted.")
    
    return entity_value

if __name__ == "__main__":
    # Path to the test image
    image_path = 'images1/614hn5uX9MS.jpg'  # Replace with your image path

    # Define the entity name for which you want to extract the value
    entity_name = "width"  # Example entity (make sure it's in your entity_unit_map)

    # Extract text and entity value using the combined method
    final_entity_value = extract_text_and_entity(image_path, entity_name)
    print(f"Final entity value: {final_entity_value}")
