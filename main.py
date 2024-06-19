from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from second import processing
import base64
import sqlite3
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

def create_table(database="results.db", table="test_FD001"):
    db = sqlite3.connect(database)
    c = db.cursor()
    c.execute(f"""CREATE TABLE {table[:-4]} (
        value integer          
    )""")
    db.commit()
    db.close()

def add_to_table(data, database, table):
    db = sqlite3.connect(database)
    c = db.cursor()
    c.executemany(f"INSERT INTO {table[:-4]} (value) VALUES (?)", [(value,) for value in data])
    db.commit()
    db.close()


@app.get("/", response_class=HTMLResponse)
async def read_index():
    html_content=f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Загрузка файлов</title>
    </head>
    <body>
        <div class="upload-container">
            <h2>Загрузка файлов</h2>
            <form action="/submit" method="post" enctype="multipart/form-data">
                <label for="testData">Test data:</label><br>
                <input type="file" id="testData" name="testData" accept=".txt" required><br><br>

                <label for="trueRUL">True RUL:</label><br>
                <input type="file" id="trueRUL" name="trueRUL" accept=".txt" required><br><br>

                <input type="submit" value="Отправить">
            </form>
        </div>
    </body>
    </html>"""
    return HTMLResponse(content=html_content,status_code=200)


@app.post("/submit/")
async def upload_files(testData: UploadFile = File(...), trueRUL: UploadFile = File(...)):
    testData_contents = await testData.read()
    trueRUL_contents = await trueRUL.read()
    buffer_bytes, preds = processing(testData_contents, trueRUL_contents)
    base64_encoded_image = base64.b64encode(buffer_bytes.getvalue()).decode()
    preds = preds.tolist()
    create_table("results.db", table=testData.filename)
    add_to_table(preds, "results.db", testData.filename)
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

@app.get("/history/{table}")
def generate_html_table_from_db(table: str):
    db = sqlite3.connect("results.db")
    c = db.cursor()
    c.execute(f"SELECT * FROM {table}")
    rows = c.fetchall()
    db.commit()
    db.close()
    # Генерируем HTML для таблицы
    html_content = "<table border='1'>\n"

    # Заголовки столбцов
    html_content += "<tr>"
    html_content += "<th>engine ID</th>"
    html_content += "<th>RUL</th>"
    html_content += "</tr>\n"
    i = 0
    for row in rows:
        html_content += "<tr>"
        i += 1
        for value in row:
            html_content += f"<td>{i}</td>"
            html_content += f"<td>{round(value)}</td>"
        html_content += "</tr>\n"
    html_content += "</table>"
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)