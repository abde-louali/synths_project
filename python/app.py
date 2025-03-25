import os
from glob import glob
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from flask import Flask, request, jsonify
import zipfile
import tempfile
import re
import json

# Load the OCR model from docTR
model = ocr_predictor(pretrained=True)

# Keywords to search for
keywords = ["Prénom", "Nom", "Le candidat(e)"]


def normalize_value(value):
    """Clean and normalize extracted values."""
    value = value.replace(":", "").strip()
    return value


def reformat_name(name_info):
    """Reformat names into a consistent format."""
    if "Prénom" in name_info and "Nom" in name_info:
        return f"{normalize_value(name_info['Prénom'])} {normalize_value(name_info['Nom'])}"
    elif "Le candidat(e)" in name_info:
        full_name = normalize_value(name_info["Le candidat(e)"])
        parts = full_name.split()
        if len(parts) == 2:
            return f"{parts[1]} {parts[0]}"
        return full_name
    return None


def extract_names(text, keywords):
    """Extract names using keywords."""
    name_info = {}
    lines = text.split("\n")

    for line in lines:
        for keyword in keywords:
            if keyword in line:
                value = line.split(keyword)[-1].strip()
                name_info[keyword] = value
                break

    return name_info


def extract_capital_words(result):
    """Extract capitalized words from OCR result."""
    capital_words = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    word_text = word.value
                    if word_text.isupper() and len(word_text) > 2:
                        capital_words.append(word_text)
    return capital_words


def process_image(image_path):
    """Process individual image for name extraction."""
    try:
        # Load and process the image
        doc = DocumentFile.from_images(image_path)
        result = model(doc)
        extracted_text = result.render()

        # First attempt: Extract names using keywords
        name_info = extract_names(extracted_text, keywords)

        if name_info:
            return name_info
        else:
            # Fallback: Extract capitalized words
            capital_words = extract_capital_words(result)

            # Check if there are enough capitalized words
            if len(capital_words) >= 7:
                return {
                    "Prénom": capital_words[5],
                    "Nom": capital_words[6]
                }

            # Additional fallback: Look for patterns in the text
            lines = extracted_text.split('\n')
            for i, line in enumerate(lines):
                if ':' in line:
                    parts = line.split(':')
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    if 'nom' in key:
                        name_info['Nom'] = value
                    elif 'prénom' in key or 'prenom' in key:
                        name_info['Prénom'] = value

                if 'candidat' in line.lower() and i + 1 < len(lines):
                    name_info['Le candidat(e)'] = lines[i + 1].strip()

            if name_info:
                return name_info

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

    return None


def compare_names_in_folder(folder_path):
    """Compare names across images in a folder."""
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    results = []

    for subdir in subdirs:
        subdir_path = os.path.join(folder_path, subdir)
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            pattern = os.path.join(subdir_path, ext)
            image_paths.extend(glob(pattern))

        if not image_paths:
            continue

        # Extract CIN from folder name (assuming format like "BB567890_Name")
        cin = subdir.split('_')[0] if '_' in subdir else 'Unknown'

        student_result = {
            'cin': cin,
            'folder_name': subdir,
            'extracted_names': [],
            'is_correct': False,
            'verified_name': None,
            'files_processed': []
        }

        for image_path in image_paths:
            names = process_image(image_path)
            if names:
                formatted_name = reformat_name(names)
                if formatted_name:
                    student_result['extracted_names'].append(formatted_name)
                    student_result['files_processed'].append({
                        'file': os.path.basename(image_path),
                        'extracted_name': formatted_name
                    })

        if student_result['extracted_names']:
            first_name = student_result['extracted_names'][0]
            if all(name == first_name for name in student_result['extracted_names']):
                student_result['is_correct'] = True
                student_result['verified_name'] = first_name

        results.append(student_result)

    return json.dumps(results)


# Flask Application
app = Flask(__name__)


@app.route('/validate', methods=['POST'])
def validate_folder():
    """Flask route for folder validation."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    uploaded_file = request.files['file']
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, uploaded_file.filename)
    uploaded_file.save(zip_path)

    extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        results = []
        subdirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]

        for subdir in subdirs:
            subdir_path = os.path.join(extract_dir, subdir)
            image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                pattern = os.path.join(subdir_path, ext)
                image_paths.extend(glob(pattern))

            if not image_paths:
                continue

            extracted_names = []
            file_details = []

            # Process each image in the folder
            for image_path in image_paths:
                names = process_image(image_path)
                formatted_name = reformat_name(names) if names else None

                file_details.append({
                    "file": os.path.basename(image_path),
                    "extracted_name": formatted_name,
                    "raw_data": names
                })

                if formatted_name:
                    extracted_names.append(formatted_name)

            # Determine if all names match
            is_correct = False
            verified_name = None
            errors = []

            if extracted_names:
                first_name = extracted_names[0]
                is_correct = all(name == first_name for name in extracted_names)
                verified_name = first_name if is_correct else None

                # Generate errors for mismatches
                for detail in file_details:
                    if not detail["extracted_name"]:
                        errors.append({
                            "file": detail["file"],
                            "error": "No name could be extracted"
                        })
                    elif not is_correct and detail["extracted_name"] != first_name:
                        errors.append({
                            "file": detail["file"],
                            "error": f"Name mismatch: found '{detail['extracted_name']}'"
                        })

            # Get CIN from folder name
            cin = subdir.split('_')[0] if '_' in subdir else subdir

            results.append({
                "cin": cin,
                "is_correct": is_correct,
                "verified_name": verified_name,
                "errors": errors,
                "file_details": file_details
            })

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    app.run(host='localhost', port=81)