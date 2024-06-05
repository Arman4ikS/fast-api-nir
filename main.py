from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from second import processing
import base64

app = FastAPI()

#app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    html_content = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Загрузка файлов</title>
    </head>
    <body>
        <h2>Загрузка файлов</h2>
        <form action="/submit" method="post" enctype="multipart/form-data">
            <label for="testData">Test data:</label><br>
            <input type="file" id="testData" name="testData" accept=".txt"><br><br>
    
            <label for="trueRUL">True RUL:</label><br>
            <input type="file" id="trueRUL" name="trueRUL" accept=".txt"><br><br>
    
            <input type="submit" value="Отправить">
        </form>
    </body>
    </html>"""
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/submit/")
async def upload_files(testData: UploadFile = File(...), trueRUL: UploadFile = File(...)):
    testData_contents = await testData.read()
    trueRUL_contents = await trueRUL.read()
    buffer_bytes, preds = processing(testData_contents, trueRUL_contents)
    base64_encoded_image = base64.b64encode(buffer_bytes.getvalue()).decode()
    preds = preds.tolist()
    index_row = ''.join(f'<th>{i}</th>' for i in range(1, len(preds) + 1))
    data_row = ''.join(f'<td>{int(value)}</td>' for value in preds)

    html_content2 = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>График</title>
    </head>
    <body>
        <h2>График</h2>
        <img src="data:image/png;base64,{base64_encoded_image}" alt="График">
        <h2>NumPy Table</h2>
        <table border="1">
        <tr>{index_row}</tr>
        <tr>{data_row}</tr>
        </table>
    </body>
    </html>"""
    return HTMLResponse(content=html_content2, status_code=200)


# Запуск сервера
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)