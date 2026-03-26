# Лабораторная работа №2
## Структура проекта
<img width="516" height="393" alt="image" src="https://github.com/user-attachments/assets/4748ba5b-f872-4c64-88b4-e8e070c184b2" />

## Требования
- Python 3.8+
- Windows
- Jenkins for Windows (tested on [Jenkins 2.541.3](https://www.jenkins.io/download/#downloading-jenkins))
- pip, python3-venv
- OpenJDK (tested on [Temurin JDK 25.0.2](https://adoptium.net/temurin/releases/))
  
## Запуск
1. Склонируйте репозиторий
```
git clone https://github.com/devorkyan/mlops_practice.git
cd lab2
```
2. Установите зависимости
```
pip install -r requirements.txt
```
3. Установите [Jenkins for Windows](https://www.jenkins.io/doc/book/installing/windows/)
4. Создайте Item и выберете тип Pipeline
5. Настройки -> Pipeline -> Definition: Pipeline script from SCM -> SCM: Git -> Repository URL: https://github.com/devorkyan/mlops_practice.git -> Branches to build: */master
6. Настройки -> Pipeline -> Script Path: lab2/Jenkinsfile
7. Нажмите "Собрать сейчас"

## Данные
- Тип задачи - Бинарная классификация
- Количество объектов - 1000
- Данные по умолчанию создаются синтетически через sklearn.datasets.make_classification, скачивание данных из интернета при явном указании url
- 10 признаков: 7 информативных, 2 избыточных, 1 шумовой
- Добавлен шум + намеренные корреляции

## Модели
- Алгоритм - Логистическая регрессия/случайный лес
- Библиотека - scikit-learn
- Предобработка:
  - StandardScaler - нормализация признаков
  - Скалер обучается только на train-данных (предотвращает утечку)
- Метрика:
  - Accuracy - доля правильных предсказаний на тестовой выборке
- Типичный результат:
  - Model test accuracy is: 0.760
- Отчет в виде метрик сохраняется в формате json

## Этапы

data_creation.py - Генерация данных, разделение на train/test, сохранение в CSV

data_preprocessing.py - Нормализация признаков через StandardScaler

model_preparation.py - Обучение логистическая регрессия/случайный лес, сохранение в pickle

model_testing.py - Загрузка модели, предсказание, сохранение метрик, вывод accuracy

## Jenkinsfile
- Скрипт автоматически:
  - Определяет пути для Windows
  - Скачивает код из Git-репозитория в WORKSPACE
  - Создает изолированный Python, устанавливает зависимости
  - Последовательно запускает 4 этапа конвейера
- Особенности:
  - Всегда очищается workspace (удаляются временные файлы)
  - Логируется результат выполнения пайплайна
