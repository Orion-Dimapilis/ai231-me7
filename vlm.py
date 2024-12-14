import cv2
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import threading

# Load processor and model for BLIP image captioning
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Check if GPU is available and move model to GPU if so
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Open webcam (0 is default for the first webcam)
cap = cv2.VideoCapture(0)

# Global variable for the question
question = "What can you see?"

# Function to update the question from the terminal
def update_question():
    global question
    while True:
        user_input = input("Enter a new question: ")
        if user_input.strip():
            question = user_input.strip()

# Start a thread for updating the question
question_thread = threading.Thread(target=update_question, daemon=True)
question_thread.start()

# Process webcam frames continuously
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to a PIL Image for the model
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image and question
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)

    # Generate the answer
    outputs = model.generate(**inputs)

    # Decode the answer
    answer = processor.decode(outputs[0], skip_special_tokens=True)

    # Show the answer on the webcam feed
    cv2.putText(frame, f"Q: {question}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"A: {answer.split(':')[-1].strip()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame with the generated answer
    cv2.imshow("Webcam", frame)

    # Exit the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
