
import pickle
import io
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp

from fastapi import FastAPI, UploadFile, File  # FastAPI для создания API, UploadFile/File для загрузки файлов
from fastapi.responses import StreamingResponse  # Для потоковой передачи CSV файла
from pydantic import BaseModel  # Для валидации JSON схемы данных
from typing import List, Optional

app = FastAPI()  # Создаем экземпляр FastAPI приложения

# Загружаем артефакты один раз при старте (чтобы не перезагружать при каждом запросе)
with open('ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)  # OneHotEncoder для категориальных признаков

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)  # Обученная модель машинного обучения

with open('medians.pkl', 'rb') as f:
    medians = pickle.load(f)  # Медианы для заполнения пропусков

with open('col.pkl', 'rb') as f:
    cat_cols = pickle.load(f)  # Список категориальных колонок для OHE

with open('num_cols.pkl', 'rb') as f:
    num_cols = pickle.load(f)  # Список числовых колонок

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)  # StandardScaler для нормализации числовых признаков


# ── Pydantic-схемы (описывают структуру запроса) ──────────────────────────────────────────────

class Item(BaseModel):  # Схема для одного автомобиля
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):  # Схема для списка автомобилей
    objects: List[Item]


# ── Предобработка (очистка и преобразование данных) ───────────────────────────────────────────────

def parse_torque(torque_str):  # Функция парсинга крутящего момента (из строки в числа)
    if pd.isna(torque_str) or torque_str == 'NaN':  # Если пропуск или NaN
        return None, None  # Возвращаем два None

    s = torque_str.lower().replace(',', '')  # Приводим к нижнему регистру и убираем запятые
    torque_nm = None  # Крутящий момент в Ньютон-метрах

    kgm_match = re.search(r'(\d+(?:\.\d+)?)\s*kgm', s)  # Ищем значение в кгс·м
    if kgm_match:
        torque_nm = float(kgm_match.group(1)) * 9.8  # Переводим кгс·м в Н·м
    else:
        nm_match = re.search(r'(\d+(?:\.\d+)?)\s*nm', s)  # Ищем значение в Н·м
        if nm_match:
            torque_nm = float(nm_match.group(1))  # Оставляем как есть
        else:
            bracket_match = re.search(r'(\d+(?:\.\d+)?)\s*@.*\(kgm', s)  # Вариант: число @ ... (kgm)
            if bracket_match:
                torque_nm = float(bracket_match.group(1)) * 9.8  # Переводим

    max_rpm = None  # Максимальные обороты крутящего момента
    range_match = re.search(r'(\d+)-(\d+)\s*rpm', s)  # Диапазон: 2000-4000 rpm
    if range_match:
        max_rpm = int(range_match.group(2))  # Берем максимальное значение
    else:
        single_match = re.search(r'(\d+)\s*rpm', s)  # Одиночное значение: 4000 rpm
        if single_match:
            max_rpm = int(single_match.group(1))  # Берем его
        else:
            range_before = re.search(r'(\d+)-(\d+)\s*\(', s)  # Диапазон перед скобкой
            if range_before:
                max_rpm = int(range_before.group(2))  # Берем максимальное
            else:
                single_before = re.search(r'(\d+)\s*\(', s)  # Одиночное перед скобкой
                if single_before:
                    max_rpm = int(single_before.group(1))

    return torque_nm, max_rpm  # Возвращаем два числа


def preprocess(df: pd.DataFrame) -> sp.csr_matrix:  # Главная функция предобработки (возвращает разреженную матрицу)
    df = df.copy()  # Работаем с копией, чтобы не испортить оригинал

    df['name'] = df['name'].str.split().str[0]  # Извлекаем бренд (первое слово)

    # Очистка числовых колонок от единиц измерения и приведение к float
    df['mileage'] = (df['mileage'].str.replace(' kmpl', '')
                                  .str.replace(' km/kg', '')
                                  .astype('float'))  # Расход топлива
    df['engine'] = df['engine'].str.replace(' CC', '').astype('float')  # Объем двигателя
    df['max_power'] = (df['max_power'].str.replace(' bhp', '')
                                      .replace('', np.nan)
                                      .astype('float'))  # Мощность

    # Парсим torque и max_torque_rpm в две отдельные колонки
    df[['torque', 'max_torque_rpm']] = df['torque'].apply(
        lambda x: pd.Series(parse_torque(x))
    )

    # Заполняем пропуски медианами из обучающей выборки
    for col_name, median_val in medians.items():
        df[col_name] = df[col_name].fillna(median_val)

    # Приводим к целочисленному типу
    df['seats'] = df['seats'].astype('int')
    df['engine'] = df['engine'].astype('int')

    # Отладка — печатаем типы и значения перед сборкой матрицы (для диагностики)
    print("Типы перед сборкой матрицы:")
    print(df[num_cols].dtypes)
    print(df[num_cols].head())

    print("num_cols:", num_cols)
    print("cat_cols:", cat_cols)
    print(df[num_cols].head())

    # Стандартизируем числовые признаки (преобразуем в csr_matrix)
    X_num = sp.csr_matrix(scaler.transform(df[num_cols].values.astype(float)))
    # One-hot кодирование категориальных признаков
    X_cat = ohe.transform(df[cat_cols])

    return sp.hstack([X_num, X_cat])  # Объединяем горизонтально в одну матрицу


# ── Эндпоинты (API ручки) ───────────────────────────────────────────────────

@app.post("/predict_item")  # Эндпоинт для предсказания одного автомобиля (JSON запрос)
def predict_item(item: Item) -> float:  # Принимает Item, возвращает float (предсказанную цену)
    df = pd.DataFrame([item.dict()])  # Преобразуем JSON в DataFrame
    X = preprocess(df)  # Предобработка
    return float(model.predict(X)[0])  # Предсказываем и возвращаем цену


@app.post("/predict_items")  # Эндпоинт для предсказания CSV файла (много автомобилей)
def predict_items(file: UploadFile = File(...)) -> StreamingResponse:  # Загружаем CSV файл
    content = file.file.read()  # Читаем содержимое файла
    df = pd.read_csv(io.BytesIO(content))  # Парсим CSV в DataFrame

    X = preprocess(df)  # Предобработка всех строк
    df['predicted_price'] = model.predict(X)  # Добавляем колонку с предсказаниями

    output = io.StringIO()  # Создаем строковый буфер
    df.to_csv(output, index=False)  # Сохраняем DataFrame в CSV строку
    output.seek(0)  # Перемещаем курсор в начало буфера

    return StreamingResponse(  # Возвращаем CSV файл как ответ
        io.BytesIO(output.getvalue().encode()),  # Кодируем строку в байты
        media_type="text/csv",  # Тип содержимого
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}  # Имя скачиваемого файла
    )
