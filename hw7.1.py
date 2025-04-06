import requests  # Библиотека для отправки HTTP-запросов
import io  # Модуль для работы с потоками ввода-вывода
from PIL import Image  # Библиотека для работы с изображениями
import os  # Модуль для взаимодействия с операционной системой
from dotenv import load_dotenv  # Библиотека для загрузки переменных окружения из файла .env


def setup_env():
    """Настройка окружения и проверка наличия токена Hugging Face"""
    load_dotenv()  # Загружаем переменные окружения из файла .env

    # Проверяем наличие токена Hugging Face
    hf_token = os.getenv("HF_TOKEN")  # Получаем токен из переменных окружения
    if not hf_token:  # Если токен не найден
        print("No Hugging Face token found. You need to:")
        print("1. Create a free account at huggingface.co")  # Инструкция: создать аккаунт
        print("2. Get your token at huggingface.co/settings/tokens")  # Инструкция: получить токен
        print("3. Create a .env file with HF_TOKEN=your_token_here")  # Инструкция: создать файл .env
        hf_token = input("Or enter your Hugging Face token now: ")  # Возможность ввести токен вручную

    return hf_token  # Возвращаем токен для дальнейшего использования

def generate_image(prompt, token, negative_prompt="", model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                   num_inference_steps=30):
    """
    Генерация изображения с помощью Stable Diffusion через API Hugging Face

    Аргументы:
        prompt (str): Текстовый запрос для генерации изображения
        token (str): Токен API Hugging Face
        negative_prompt (str): Описание того, что следует избегать в сгенерированном изображении
        model_id (str): Идентификатор модели на Hugging Face
        num_inference_steps (int): Количество шагов денойзинга (больше = лучше качество, но медленнее)

    Возвращает:
        PIL.Image: Сгенерированное изображение
    """
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"  # URL модели на Hugging Face
    headers = {
        "Authorization": f"Bearer {token}",  # Заголовок авторизации с токеном
        "Content-Type": "application/json"  # Тип содержимого запроса
    }

    payload = {  # Данные для отправки на сервер
        "inputs": prompt,  # Текстовый запрос
        "parameters": {  # Параметры генерации
            "negative_prompt": negative_prompt,  # Что избегать в изображении
            "num_inference_steps": num_inference_steps,  # Количество шагов генерации
        }
    }

    # Отправляем запрос к API Hugging Face
    print(f"Generating image for prompt: '{prompt}'")
    response = requests.post(API_URL, headers=headers, json=payload)  # POST-запрос с данными в формате JSON

    if response.status_code != 200:  # Проверка успешности запроса
        raise Exception(f"Error: {response.status_code}, {response.text}")  # Выбрасываем исключение при ошибке

    # Преобразуем ответ в изображение
    image = Image.open(io.BytesIO(response.content))  # Открываем изображение из бинарных данных ответа
    return image  # Возвращаем изображение


def save_image(image, filename="generated_image.png"):
    """Сохранение сгенерированного изображения в файл"""
    image.save(filename)  # Сохраняем изображение с указанным именем файла
    print(f"Image saved as {filename}")  # Выводим сообщение о сохранении
    return filename  # Возвращаем имя файла


def main():
    # Настройка окружения
    token = setup_env()  # Получаем токен Hugging Face

    # Генерируем изображения с разными запросами
    prompts = [
        "A programmer sits in a room in front of a computer.",
        "A programmer sits in a room in front of a computer. The computer has a keyboard with neon backlighting, and two monitors are hanging on the wall.",
        "A programmer sits in a modern room in front of a computer, with a keyboard featuring neon backlighting on the desk. Two monitors hang on the walls, reflecting bright colors. The room has a modern interior with minimalist furniture, several tech gadgets on the desk, and soft, cool lighting around. The atmosphere is peaceful and focused.",
    ]

    for i, prompt in enumerate(prompts):  # Перебираем все запросы
        try:
            # Генерируем изображение
            image = generate_image(
                prompt=prompt,  # Текстовый запрос
                token=token,  # Токен Hugging Face
                # negative_prompt="blurry, bad quality, distorted, ugly",  # Что избегать в изображении
                num_inference_steps=50  # Меньше для более быстрой генерации, больше для лучшего качества
            )

            # Сохраняем изображение
            filename = f"image_level_{i + 1}.png"  # Формируем имя файла с порядковым номером
            save_image(image, filename)  # Сохраняем изображение

            # Отображаем изображение, если запущено в интерактивной среде
            try:
                # Это будет работать в среде Jupyter Notebook
                from IPython.display import display  # Импортируем функцию для отображения в Jupyter
                display(image)  # Отображаем изображение
            except ImportError:  # Если не в Jupyter
                print(f"Image generated and saved as {filename}")  # Просто выводим информацию о сохранении

        except Exception as e:  # Обрабатываем возможные ошибки
            print(f"Error generating image for prompt '{prompt}': {e}")  # Выводим сообщение об ошибке


if __name__ == "__main__":
    main()  # Запускаем основную функцию при выполнении скрипта напрямую