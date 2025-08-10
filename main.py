from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import shutil

app = FastAPI()

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.post("/upload-image/")
async def upload_image(
    file: UploadFile = File(...),
    label: str = Form(...)
):
    try:
        label_folder = os.path.join(UPLOAD_FOLDER, label)
        os.makedirs(label_folder, exist_ok=True)

        file_location = os.path.join(label_folder, file.filename)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)

        return JSONResponse(content={
            "message": "Image uploaded successfully!",
            "filename": file.filename,
            "label": label
        })
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
