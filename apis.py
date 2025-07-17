from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
from Neural_Style_Transfer_Model.model_training import stylize_img
import warnings
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

warnings.filterwarnings(action = "ignore")

@app.post("/stylize_img_upload")
async def stylize_img_upload(content_img: UploadFile = File(...), style_img: UploadFile = File(...)):
    # Saving uploads to temp files
    content_path = f"temp_content_{content_img.filename}"
    style_path = f"temp_style_{style_img.filename}"
    with open(content_path, "wb") as f:
        f.write(await content_img.read())
    with open(style_path, "wb") as f:
        f.write(await style_img.read())

    # Running stylization
    saved_img_path = stylize_img(img_path=content_path, style_path=style_path)

    # Cleaning up input files
    os.remove(content_path)
    os.remove(style_path)

    if os.path.exists(saved_img_path):
        return FileResponse(saved_img_path, media_type="image/png")
    else:
        return {"error": "Image not found"}

if __name__ == "__main__":
    uvicorn.run("apis:app", host = "127.0.0.1", port = 8000, reload = True)