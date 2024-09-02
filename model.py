import cv2
import pytesseract
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    text = pytesseract.image_to_string(thresh)
    return text


def detect_ui_elements(image):
    ui_elements = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 20:
            element_type = classify_ui_element(image[y:y + h, x:x + w])
            ui_elements.append(f"{element_type} at ({x}, {y}) with size ({w}x{h})")
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return ui_elements, image


def classify_ui_element(element_image):
    gray = cv2.cvtColor(element_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(binary).strip()

    if text:
        return f"Text element ('{text}')"
    elif element_image.shape[0] < element_image.shape[1]:
        return "Possible button or input field"
    else:
        return "Unknown UI element"


def generate_test_cases(feature_description, screenshots):
    example_test_case = """
    Test Case: Source and Destination Selection
    Description: Verify that users can select source and destination cities for their journey.
    Pre-conditions:
    - The Red Bus app is installed and launched.
    - The user is on the home screen.
    Testing Steps:
    1. Tap on the "From" field.
    2. Type in the source city name (e.g., "Mumbai").
    3. Select the correct city from the dropdown list.
    4. Tap on the "To" field.
    5. Type in the destination city name (e.g., "Pune").
    6. Select the correct city from the dropdown list.
    Expected Result:
    - The "From" field displays the selected source city.
    - The "To" field displays the selected destination city.
    - The app allows proceeding to the next step (date selection).
    """

    try:
        messages = [
            {"role": "system",
             "content": "You are a QA expert specializing in mobile app testing. Generate detailed test cases for the Red Bus app features shown in the screenshots. Focus on the core features: source/destination/date selection, bus selection, seat selection, and pick-up/drop-off point selection. Also consider bonus features like offers, filters, and bus information if visible. Structure each test case with a description, pre-conditions, testing steps, and expected results."},
            {"role": "user",
             "content": f"Here's an example of a well-structured test case:\n{example_test_case}\nNow, generate test cases for the following Red Bus app features based on these details:\n{feature_description}"}
        ]

        for i, screenshot in enumerate(screenshots):
            messages.append({"role": "user", "content": f"Screenshot {i + 1} contents: {screenshot}"})

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1000,
            n=1,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating test cases: {str(e)}")
        return ""


def process_image_and_generate_test_cases(images, context):
    all_text = []
    all_ui_elements = []
    annotated_images = []

    for i, image in enumerate(images):
        text = extract_text_from_image(image)
        ui_elements, annotated_image = detect_ui_elements(image)
        all_text.append(f"Screenshot {i+1} Text:\n{text}")
        all_ui_elements.extend([f"Screenshot {i+1}: {element}" for element in ui_elements])
        annotated_images.append(annotated_image)

    feature_description = f"{context}\n\nExtracted Text:\n" + "\n\n".join(all_text) + "\n\nDetected UI Elements:\n" + "\n".join(all_ui_elements)
    test_cases = generate_test_cases(feature_description, all_text)
    return test_cases, annotated_images

if __name__ == "__main__":
    test_image_path = "screenshot.jpg"
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Failed to load image from {test_image_path}")
    else:
        ui_elements, annotated_image = detect_ui_elements(image)
        print("Detected UI Elements:")
        for element in ui_elements:
            print(element)
        cv2.imshow("Annotated Image", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

