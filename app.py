"""
Flask web application for hand digit image recognition.
This module provides routes for uploading and predicting hand digit images.
"""

# Importing required libs
from flask import Flask, render_template, request
from PIL import UnidentifiedImageError
from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)

# Home route
@app.route("/")
def main():
    """Render the main page with image upload form."""
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """
    Process uploaded image and return prediction result.
    """
    try:
        if request.method == 'POST':
            img = preprocess_img(request.files['file'].stream)
            pred = predict_result(img)
            return render_template("result.html", predictions=str(pred))
        return render_template("result.html", err="Invalid request method.")
    except KeyError:
        error = "No file uploaded. Please select an image file."
        return render_template("result.html", err=error)
    except UnidentifiedImageError:
        error = "Invalid image file cannot be processed. Please upload a valid image."
        return render_template("result.html", err=error)
    except (OSError, IOError) as e:
        error = f"Error reading file: {str(e)}"
        return render_template("result.html", err=error)
    except (ValueError, RuntimeError) as e:
        error = f"Error processing image: {str(e)}"
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
