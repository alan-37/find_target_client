# Как приступить к работе?
1) Если вы хотите запустить pipeline.py, необходимо скачать датасет по ссылке(https://drive.google.com/drive/folders/1rA4o6KHH-M2KMvBLHp5DZ5gioF2q7hZw) в директорию data. После запуска pipeline.py, будет обновлен файл model/event_action.dill, который будет содержать: модель обучения, а также, неболую информацию о модели.
2) Для того, чтобы получить данные на проверку, можете запустить файл data/creat_verification_value.py. После этого будет создан файл data/find.csv, в котором можно будет взять значения для проверки модели
3) Для запуска сервиса необходимо в проекте в директории, где находится файл main.py, открыть терминал, активировать виртуальное оеружение и ввести команду:  uvicorn main:app --reload.
4) Для того, чтобы проверить модель, нужен Postman.В Postman необходимо выбрать метод POST и ввести: http://127.0.0.1:8000/predict. Далее открыть вкладку Body, а там выбрать raw. После этого вставить значения из find.csv. Как на примере:
   {
    "utm_source":"ZpYIoDJMcFzVoPFsHGJL",
    "utm_medium":"banner",
    "utm_campaign":"TmThBvoCcwkCZZUWACYq",
    "utm_adcontent":"LEoPHuyFvzoNfnzGgfcd",
    "utm_keyword":"vCIpmpaGBnIQhyYNkXqp",
    "device_category":"mobile",
    "device_os":"Android",
    "device_brand":"Samsung",
    "device_browser":"Chrome",
    "device_screen_resolution": "393x851",
    "geo_country":"Russia",
    "geo_city":"Saint Petersburg"
}
