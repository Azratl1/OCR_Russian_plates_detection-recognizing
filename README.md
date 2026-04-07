# License Plate Recognition (YOLO + OCR)

Проект для автоматического обнаружения и распознавания автомобильных номерных знаков на изображениях. Решение построено как двухэтапный пайплайн: сначала модель детекции находит номерной знак, затем OCR-модель распознаёт текст на вырезанной области.

## Что делает проект

- обнаруживает номерной знак на изображении;
- вырезает область номерной рамки;
- выполняет предобработку изображения перед OCR;
- распознаёт текст номерного знака;
- применяет постобработку результата с учётом формата российских номеров.

## Архитектура

### Детекция номера

Для локализации номерного знака используется YOLO. Модель получает входное изображение и возвращает bounding box с оценкой уверенности. Для обучения подготовлен отдельный пайплайн на датасете в формате YOLO.

### Распознавание текста

Для OCR используется модель CRNN:

- сверточная часть извлекает признаки изображения номера;
- двунаправленная LSTM обрабатывает последовательность признаков;
- CTC-декодирование преобразует выход сети в текст.

## Pipeline

1. Загрузка изображения.
2. Детекция номерного знака.
3. Вырезание области номера.
4. Предобработка crop-изображения.
5. Распознавание текста OCR-моделью.
6. Постобработка результата.

## Структура проекта

```text
.
├── app.py
├── crnn_model.py
├── demo_my_images.py
├── evaluate_ocr.py
├── full_plate_pipeline.py
├── label_converter.py
├── ocr_augment.py
├── ocr_collate.py
├── ocr_dataset.py
├── prepare_ocr_dataset.py
├── prepare_yolo_dataset.py
├── test_dataloader.py
├── train_ocr.py
├── train_yolo.py
├── validate_yolo_dataset.py
├── requirements.txt
└── requirements-desktop.txt
```

## Датасеты

### Детекция

Для обучения детектора используется датасет с разметкой в формате YOLO:

```text
<class_id> <cx> <cy> <w> <h>
```

Подготовка локального конфига:

```bash
python prepare_yolo_dataset.py
```

Проверка корректности разметки:

```bash
python validate_yolo_dataset.py
```

### OCR

Для обучения OCR используется датасет изображений номерных знаков с текстовыми метками. CSV-файлы содержат путь к изображению и строку с номером:

```text
image_path,label
```

Подготовка CSV-сплитов:

```bash
python prepare_ocr_dataset.py
```

## Установка

Создай виртуальное окружение и установи зависимости:

```bash
pip install -r requirements.txt
```

Дополнительные зависимости для GUI вынесены отдельно:

```bash
pip install -r requirements-desktop.txt
```

## Обучение

### YOLO

```bash
python train_yolo.py
```

После обучения лучший вес обычно сохраняется в каталоге `runs_yolo/.../weights/best.pt`.

### OCR

```bash
python train_ocr.py
```

Лучший OCR-чекпоинт сохраняется в каталоге `ocr_checkpoints/`.

## Оценка и проверка

Проверка качества OCR на тестовой выборке:

```bash
python evaluate_ocr.py
```

Проверка загрузки батча OCR-датасета:

```bash
python test_dataloader.py
```

Визуальная проверка пайплайна на своих изображениях:

```bash
python demo_my_images.py --images-dir my_images
```

## Особенности проекта

- двухэтапный подход: детекция + OCR;
- постобработка под российский формат номерных знаков;
- аугментации для OCR: blur, noise, perspective, occlusion;
- отдельные сценарии подготовки и валидации датасетов;
- возможность дообучения моделей на собственных данных.

## Требования к весам

Для полноценной работы проекта нужны обученные веса:

- YOLO-веса детектора;
- OCR-веса `ocr_best.pt`.

Пути к весам можно указывать через аргументы скриптов или использовать значения по умолчанию, заданные в проекте.

## Назначение

Проект предназначен для учебных и исследовательских задач в области компьютерного зрения, распознавания объектов и OCR.
Ссылки на датасеты:
```bash
Для YOLO - https://www.kaggle.com/datasets/ronakgohil/license-plate-dataset
```
```bash
Для OCR - https://huggingface.co/datasets/AY000554/Car_plate_OCR_dataset
```
## Метрики обучения и пример работы приложения
<img width="974" height="487" alt="image" src="https://github.com/user-attachments/assets/f59ffc5c-69b0-41fc-ae18-5ccf49f4f879" /> Результаты обучения YOLO.
<img width="974" height="731" alt="image" src="https://github.com/user-attachments/assets/bfd680be-b680-4c32-a843-4e67f4708846" /> Матрица ошибок YOLO.
<img width="974" height="325" alt="image" src="https://github.com/user-attachments/assets/94b6ffbf-1443-4327-9c70-e845a313c139" /> Метрики OCR сети
<img width="974" height="300" alt="image" src="https://github.com/user-attachments/assets/cbd093e7-5061-478e-a21c-e7b5fb5a6ec6" /> Пример работы нейросети



