#  Find Target Client — Прогнозирование конверсии на сайте «СберАвтоподписка»

> **Финальная работа по анализу и ML для реального бизнес-кейса от партнёра — «СберАвтоподписка»**

Этот проект направлен на решение реальной бизнес-задачи: **предсказание вероятности совершения целевого действия пользователем на сайте** «СберАвтоподписка».  
На основе данных Google Analytics мы строим модель, которая помогает выявлять "горячих" пользователей — тех, кто с высокой вероятностью оставит заявку, закажет звонок или совершит другое целевое действие.

---

##  Задача

**Цель проекта** — разработать ML-модель, способную с хорошей точностью предсказывать, совершит ли пользователь **целевое действие** в рамках одного визита (сессии).

### Что считается целевым действием?
События из `ga_hits.event_action`, включая:
- `sub_car_claim_click`
- `sub_car_claim_submit_click`
- `sub_open_dialog_click`
- `sub_custom_question_submit_click`
- `sub_call_number_click`
- `sub_callback_submit_click`
- `sub_submit_success`
- `sub_car_request_submit_click`

**Конверсия (CR)** — наличие хотя бы одного такого события в рамках одной сессии (`session_id`).

---

##  Данные

Используются два датасета из Google Analytics (last-click attribution):

### 1. `ga_sessions.csv` — информация о визитах
| Поле | Описание |
|------|---------|
| `session_id` | Уникальный ID сессии |
| `client_id` | ID пользователя |
| `visit_date`, `visit_time` | Дата и время визита |
| `utm_source`, `utm_medium`, `utm_campaign`, `utm_keyword` | Метки рекламы |
| `device_category`, `device_os`, `device_browser` | Информация об устройстве |
| `geo_country`, `geo_city` | Геолокация |
| `device_brand`, `device_model`, `device_screen_resolution` | Детали устройства |

### 2. `ga_hits-001.csv` — события в рамках сессий
| Поле | Описание |
|------|---------|
| `session_id` | Ссылка на визит |
| `hit_date`, `hit_time` | Время события |
| `hit_type`, `hit_page_path` | Тип и URL страницы |
| `event_category`, `event_action`, `event_label` | Категория и тип действия |
| `event_value` | Числовое значение (если есть) |

---

##  Решение

### Архитектура проекта

Проект состоит из двух основных компонентов:

1. **pipeline.py** — обработка данных и обучение модели
2. **main.py** — API для предсказаний с использованием обученной модели

### Этапы работы:

1. **Объединение данных**  
   Сессии из `ga_sessions.csv` объединяются с событиями из `ga_hits-001.csv` по `session_id`.

2. **Формирование целевой переменной**  
   Для каждой сессии определяется `target = 1`, если было хотя бы одно целевое действие, иначе `0`.

3. **Обработка данных**  
   - Удаление ненужных колонок и значений
   - Нормализация разрешения экрана
   - Классификация трафика (органический/платный)
   - Заполнение пропущенных значений в device_os
   - Создание географических признаков

4. **Обучение модели**  
   - Используются алгоритмы: `Logistic Regression`, `Random Forest`, `XGBoost`, `CatBoost`
   - Метрика: **ROC-AUC**, ориентир — **~0.65**
   - Валидация: 3-кратная кросс-валидация

5. **Сервис предсказания**  
   - Модель упакована в FastAPI
   - Принимает JSON с атрибутами сессии
   - Возвращает `prediction: 0|1`

---

##  Как запустить проект

### 1. Клонировать репозиторий

```bash
git clone https://github.com/alan-37/find_target_client.git
cd find_target_client 
```

### 2. Установить зависимости
```bash
pip install -r requirements.txt 
```

### 3. Подготовить данные
Поместите файлы (https://drive.google.com/drive/folders/1rA4o6KHH-M2KMvBLHp5DZ5gioF2q7hZw) с данными в папку data/:

data/ga_sessions.csv
data/ga_hits-001.csv

### 4. Обучить модель
```bash
python pipeline.py
```
После выполнения модель будет сохранена в model/event_action.dill

### 5. Запустить API (локально)

```bash
uvicorn main:app --reload --port 8000
```
Сервис будет доступен на http://localhost:8000

Доступные эндпоинты:
- GET /status — проверка статуса сервиса
- GET /version — информация о модели
- POST /predict — предсказание конверсии

Пример запроса к /predict:
```JSON
{
  "utm_source": "google",
  "utm_medium": "cpc",
  "utm_campaign": "autoloan",
  "utm_adcontent": "some_content",
  "utm_keyword": "car_subscription",
  "device_category": "desktop",
  "device_os": "Windows",
  "device_brand": "HP",
  "device_browser": "Chrome",
  "device_screen_resolution": "1920x1080",
  "geo_country": "Russia",
  "geo_city": "Москва",
  "visit_date": "2018-01-01",
  "visit_number": 1
}
```
Пример ответа:
```JSON
{
   "target": 1
}
```
#  Ключевые метрики модели
В проекте используется автоматический выбор лучшей модели из:

- Logistic Regression
- Random Forest
- CatBoost
- XGBoost

Метрика оценки: ROC-AUC

После обучения лучшая модель сохраняется в model/event_action.dill с метаданными, включая достигнутый ROC-AUC. 
