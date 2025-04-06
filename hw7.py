import torch  # Библиотека для работы с тензорами и нейронными сетями
import clip  # Библиотека CLIP (Contrastive Language-Image Pre-training) от OpenAI
from PIL import Image  # Библиотека для работы с изображениями
import requests  # Библиотека для отправки HTTP-запросов
from io import BytesIO  # Модуль для работы с бинарными данными в памяти


def setup_clip():
    """
    Настройка CLIP: проверка установки необходимых пакетов.
    Возвращает доступное устройство (CUDA для GPU или CPU).
    """
    # Проверяем, установлен ли CLIP, если нет - выводим сообщение
    try:
        import clip
    except ImportError:
        print("Installing CLIP...")  # Сообщение о необходимости установки CLIP

    # Определяем доступное устройство: GPU (CUDA) или CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")  # Выводим информацию об используемом устройстве

    return device  # Возвращаем устройство для дальнейшего использования


def load_image_from_url(url):
    """Загрузка изображения по URL-адресу."""
    response = requests.get(url)  # Отправляем HTTP-запрос для получения изображения
    return Image.open(BytesIO(response.content))  # Открываем изображение из полученных данных


def load_image_from_file(file_path):
    """Загрузка изображения из локального файла."""
    return Image.open(file_path)  # Открываем изображение из файла


def text_to_image_similarity(model, processor, text_queries, images, device):
    """
    Вычисление сходства между текстовыми запросами и изображениями.

    Аргументы:
        model: Модель CLIP
        processor: Препроцессор CLIP для обработки изображений
        text_queries: Список текстовых запросов
        images: Список изображений в формате PIL
        device: Устройство для вычислений (cuda/cpu)

    Возвращает:
        Оценки сходства между каждым текстовым запросом и каждым изображением
    """
    # Обработка текста
    text_inputs = clip.tokenize(text_queries).to(device)  # Токенизируем текст и переносим на устройство
    with torch.no_grad():  # Отключаем вычисление градиентов для экономии памяти
        text_features = model.encode_text(text_inputs)  # Кодируем текст в векторы признаков
        text_features /= text_features.norm(dim=-1, keepdim=True)  # Нормализуем векторы (делаем длину = 1)

    # Обработка изображений
    image_inputs = torch.stack([processor(img).to(device) for img in images])  # Обрабатываем и объединяем изображения
    with torch.no_grad():  # Отключаем вычисление градиентов
        image_features = model.encode_image(image_inputs)  # Кодируем изображения в векторы признаков
        image_features /= image_features.norm(dim=-1, keepdim=True)  # Нормализуем векторы

    # Вычисляем сходство между текстом и изображениями
    # @ - это матричное умножение, T - транспонирование
    # softmax превращает сходства в вероятности (в сумме дают 1)
    similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
    return similarity.cpu().numpy()  # Переносим результат на CPU и конвертируем в numpy массив


def main():
    # Настройка
    device = setup_clip()  # Определяем устройство для вычислений

    # Загружаем модель и препроцессор CLIP
    print("Loading CLIP model...")
    model, processor = clip.load("ViT-B/32", device=device)  # ViT-B/32 - вариант модели (Vision Transformer)

    # Пример использования с URL-адресами
    print("\nExample 1: Comparing text prompts to online images")

    # URL-адреса изображений
    image_urls = [
        "https://cdn.shopify.com/s/files/1/0086/0795/7054/files/Golden-Retriever.jpg?v=1645179525",  # Собака
        "https://miro.medium.com/v2/resize:fit:1400/1*tMKkGydXuiOBOb15srANvg@2x.jpeg"  # Закат
    ]

    # Загружаем изображения по URL
    try:
        images = [load_image_from_url(url) for url in image_urls]  # Загружаем каждое изображение
        print(f"Successfully loaded {len(images)} images")
    except Exception as e:
        print(f"Error loading images: {e}")  # Выводим ошибку, если изображения не загрузились
        print("Falling back to local images if available...")
        # Здесь можно добавить запасной вариант загрузки
        return

    # Текстовые запросы для сравнения с изображениями
    text_queries = ["a dog", "a car", "a sunset", "a person"]

    # Вычисляем сходство между текстом и изображениями
    similarities = text_to_image_similarity(model, processor, text_queries, images, device)

    # Выводим результаты
    print("\nSimilarity Results (%):")
    for i, text in enumerate(text_queries):
        print(f"\nText: '{text}'")
        for j, url in enumerate(image_urls):
            print(f"  Image {j + 1}: {similarities[i][j] * 100:.2f}%")  # Выводим процент сходства

    # Пример с классификацией изображений
    print("\nExample 2: Zero-shot image classification")

    # Изображение для классификации
    image = images[1]  # Используем второе изображение из предыдущего примера

    # Возможные метки для классификации
    labels = ["a photo of a dog", "a photo of a cat", "a photo of a car", "a photo of a sunset"]

    # Обрабатываем изображение
    image_input = processor(image).unsqueeze(0).to(device)  # Добавляем размерность для батча и переносим на устройство

    # Обрабатываем текст
    text_inputs = clip.tokenize(labels).to(device)  # Токенизируем метки

    # Вычисляем вероятности
    with torch.no_grad():  # Отключаем вычисление градиентов
        image_features = model.encode_image(image_input)  # Кодируем изображение
        text_features = model.encode_text(text_inputs)  # Кодируем текст

        # Нормализуем векторы признаков
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Вычисляем сходство и конвертируем в вероятности
        logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = logits_per_image.cpu().numpy()[0]  # Получаем вероятности как numpy массив

    # Выводим результаты
    print("\nClassification Results:")
    for i, label in enumerate(labels):
        print(f"{label}: {probs[i] * 100:.2f}%")  # Выводим вероятность для каждой метки


if __name__ == "__main__":
    main()  # Запускаем основную функцию при выполнении скрипта напрямую