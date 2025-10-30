import cv2
import pytesseract
import torch
from docx import Document
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import sqlite3
import os
import pypandoc
from pymongo import MongoClient
from bson import ObjectId
from fer import FER

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
print("🔍 Tesseract Version:", pytesseract.get_tesseract_version())

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

client = MongoClient("mongodb+srv://abhinav3258:abhi3258@cluster0.h9z4jp3.mongodb.net/?appName=Cluster00")
db = client["ai_doc_generator"]
collection = db["documents"]

def capture_image_from_webcam(save_path="captured_image.jpg"):
    print("📷 Opening webcam... Press SPACE to capture, ESC to exit.")
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Cannot access webcam.")
        return None
    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break
        cv2.putText(frame, "Press SPACE to Capture | ESC to Exit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("AI Document Generator - Capture Mode", frame)
        key = cv2.waitKey(1)
        if key % 256 == 27:
            print("👋 Exiting camera.")
            break
        elif key % 256 == 32:
            cv2.imwrite(save_path, frame)
            print(f"✅ Image captured and saved as {save_path}")
            break
    cam.release()
    cv2.destroyAllWindows()
    return save_path if os.path.exists(save_path) else None

def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not found or invalid format.")
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

def generate_text(topic, style):
    style_prompts = {
        "Essay": f"Write a detailed essay on {topic} including introduction, body, and conclusion.",
        "Report": f"Write a formal report about {topic}, including objectives, data, and findings.",
        "Blog": f"Write an engaging and friendly blog post about {topic}. Use emojis and a casual tone.",
        "Story": f"Write a short creative story about {topic}. Make it imaginative.",
        "Summary": f"Summarize {topic} clearly and concisely.",
        "Poem": f"Write a short and beautiful poem about {topic}."
    }
    prompt = style_prompts.get(style, style_prompts["Essay"])
    print("🧠 Generating AI text...")
    response = model.generate_content([prompt])
    return response.text

def detect_objects(image_path):
    print("🔍 Running YOLOv5 object detection...")
    model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', verbose=False)
    results = model_yolo(image_path)
    detections = results.pandas().xyxy[0]['name'].tolist()
    return list(set(detections))

def generate_text_from_objects(objects):
    topic = ", ".join(objects)
    prompt = f"Write a descriptive report about an image containing {topic}. Include observations, context, and possible uses."
    response = model.generate_content([prompt])
    return topic, response.text

def detect_emotion_from_image(image_path):
    detector = FER(mtcnn=True)
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Invalid image path.")
        return None
    emotions = detector.detect_emotions(img)
    if not emotions:
        print("⚠️ No face detected in image.")
        return None
    top_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
    print(f"🧠 Detected Emotion: {top_emotion.capitalize()}")
    return top_emotion

def generate_text_from_mood(emotion):
    mood_prompts = {
        "happy": "Write a joyful and positive poem about a bright and cheerful day.",
        "sad": "Write a comforting paragraph about sadness and hope.",
        "angry": "Write a reflective note on dealing with anger and peace.",
        "surprise": "Write an exciting story full of unexpected twists.",
        "fear": "Write a short story showing how courage overcomes fear.",
        "disgust": "Write a thought-provoking article about finding beauty in imperfection.",
        "neutral": "Write a calm and balanced daily diary entry."
    }
    prompt = mood_prompts.get(emotion.lower(), "Write a creative piece about human emotions and feelings.")
    response = model.generate_content([prompt])
    return response.text

def create_docx(content, topic, style):
    doc = Document()
    doc.add_heading(f"{topic.title()} - {style}", 0)
    doc.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph("")
    doc.add_paragraph(content)
    filename = f"AI_{style}_{topic.replace(' ', '_')}.docx"
    doc.save(filename)
    print(f"📄 Document saved as {filename}")
    return filename

def convert_docx_to_format(input_file, output_format):
    output_file = os.path.splitext(input_file)[0] + f".{output_format}"
    pypandoc.convert_text(
        open(input_file, encoding="utf-8").read(),
        output_format,
        format='docx',
        outputfile=output_file,
        extra_args=['--standalone']
    )
    print(f"✅ File converted successfully: {output_file}")
    return output_file

def save_to_mongodb(topic, style, content, filename):
    doc_data = {
        "topic": topic,
        "style": style,
        "content": content,
        "filename": filename,
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    collection.insert_one(doc_data)
    print("✅ Document saved in MongoDB successfully.")

def save_mood_to_mongodb(emotion, content, filename):
    doc_data = {
        "type": "mood_document",
        "emotion": emotion,
        "content": content,
        "filename": filename,
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    collection.insert_one(doc_data)
    print("📦 Saved mood document to MongoDB successfully.")

def view_all_documents():
    docs = list(collection.find({}, {"topic": 1, "style": 1, "filename": 1, "created_at": 1, "emotion": 1}))
    if not docs:
        print("⚠️ No documents found in MongoDB.")
        return
    print("\n🗂️ Stored Documents:")
    print("-" * 80)
    for d in docs:
        print(f"🆔 ID: {d['_id']}")
        print(f"📘 Topic: {d.get('topic', 'N/A')}")
        print(f"📝 Style/Emotion: {d.get('style', d.get('emotion', 'N/A'))}")
        print(f"📄 File: {d['filename']}")
        print(f"🕒 Created At: {d['created_at']}")
        print("-" * 80)

def view_document_by_id():
    _id = input("Enter the MongoDB _id of the document: ").strip()
    try:
        doc = collection.find_one({"_id": ObjectId(_id)})
        if not doc:
            print("❌ Document not found.")
            return
        print("\n📄 Document Details:")
        print(f"📘 Topic/Emotion: {doc.get('topic', doc.get('emotion', 'N/A'))}")
        print(f"📝 Type: {doc.get('style', doc.get('type', 'N/A'))}")
        print(f"📄 File: {doc['filename']}")
        print(f"🕒 Created At: {doc['created_at']}")
        print("\n🧾 Content:\n" + "-"*60)
        print(doc["content"])
        print("-" * 60)
    except Exception as e:
        print(f"⚠️ Invalid ID or Error: {e}")

def delete_document_by_id():
    _id = input("Enter the MongoDB _id of the document to delete: ").strip()
    try:
        result = collection.delete_one({"_id": ObjectId(_id)})
        if result.deleted_count:
            print("✅ Document deleted successfully.")
        else:
            print("❌ Document not found.")
    except Exception as e:
        print(f"⚠️ Invalid ID or Error: {e}")

if __name__ == "__main__":
    print("🧠 === AI DOCUMENT GENERATOR + EMOTION DETECTION ===")
    while True:
        print("\nOptions:")
        print("1️⃣ Generate document from typed topic")
        print("2️⃣ Generate document from image text (OCR)")
        print("3️⃣ Generate AI Vision Report (YOLO + ML)")
        print("4️⃣ Capture from webcam and generate document")
        print("5️⃣ Detect emotion from image and create mood-based document")
        print("6️⃣ Convert existing DOCX to another format")
        print("7️⃣ View all stored documents")
        print("8️⃣ View document by ID")
        print("9️⃣ Delete document by ID")
        print("🔟 Exit")
        choice = input("\nEnter your choice: ").strip()
        if choice == "1":
            topic = input("Enter a topic: ").strip()
            style = input("Enter style (Essay, Report, Blog, Story, Summary, Poem): ").strip() or "Essay"
            text = generate_text(topic, style)
            filename = create_docx(text, topic, style)
            save_to_mongodb(topic, style, text, filename)
        elif choice == "2":
            path = input("Enter image file path: ").strip()
            if not os.path.exists(path):
                print("❌ File not found.")
                continue
            topic = extract_text_from_image(path)
            print(f"🧠 Extracted text: {topic}")
            text = generate_text(topic, "Report")
            filename = create_docx(text, topic, "Report")
            save_to_mongodb(topic, "Report", text, filename)
        elif choice == "3":
            path = input("Enter image file path: ").strip()
            if not os.path.exists(path):
                print("❌ File not found.")
                continue
            objects = detect_objects(path)
            if not objects:
                print("⚠️ No objects detected.")
                continue
            print(f"🧩 Detected objects: {objects}")
            topic, report = generate_text_from_objects(objects)
            filename = create_docx(report, topic, "AI Vision Report")
            save_to_mongodb(topic, "AI Vision Report", report, filename)
        elif choice == "4":
            captured_path = capture_image_from_webcam()
            if not captured_path:
                continue
            topic = extract_text_from_image(captured_path)
            if topic:
                text = generate_text(topic, "Report")
                filename = create_docx(text, topic, "Report")
                save_to_mongodb(topic, "Report", text, filename)
            else:
                objects = detect_objects(captured_path)
                topic, report = generate_text_from_objects(objects)
                filename = create_docx(report, topic, "Vision Report")
                save_to_mongodb(topic, "Vision Report", report, filename)
        elif choice == "5":
            import tkinter as tk
            from tkinter import filedialog
            import os
            print("🖼️ Choose an image file for emotion detection...")
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            pictures_dir = os.path.join(os.path.expanduser("~"), "Pictures")
            if not os.path.exists(pictures_dir):
                os.makedirs(pictures_dir)
            path = filedialog.askopenfilename(
                parent=root,
                title="Select Image for Emotion Detection",
                initialdir=pictures_dir,
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
            )
            root.destroy()
            if not path:
                print("❌ No file selected.")
                continue
            emotion = detect_emotion_from_image(path)
            if not emotion:
                print("😕 No emotion detected.")
                continue
            text = generate_text_from_mood(emotion)
            filename = create_docx(text, emotion, "Mood Document")
            save_mood_to_mongodb(emotion, text, filename)
        elif choice == "6":
            input_file = input("Enter .docx file path to convert: ").strip()
            if not os.path.exists(input_file):
                print("❌ File not found.")
                continue
            print("Available formats: pdf, txt, rtf, md, odt, html")
            fmt = input("Enter desired output format: ").strip().lower()
            convert_docx_to_format(input_file, fmt)
        elif choice == "7":
            view_all_documents()
        elif choice == "8":
            view_document_by_id()
        elif choice == "9":
            delete_document_by_id()
        elif choice == "10":
            print("👋 Exiting AI Document Generator.")
            break
        else:
            print("❌ Invalid choice. Try again.")
