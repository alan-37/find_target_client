import datetime
import dill as dill
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


#Удаление ненужных колонок
def filter_data(df):
    columns_to_drop = {'event_value', 'event_label', 'hit_referer','device_model', 'hit_type',
        'event_label', 'hit_time', 'visit_time', 'hit_date', 'hit_number',
        'hit_date','hit_page_path', 'event_category'}
    df_copy = df.copy()
    df_copy = df_copy.drop(columns_to_drop, axis=1)
    return df_copy

#Удаление ненужных значний в utm_adcontent
def filter_utm(df):
    df_drop_dupli = df.copy()
    df_new = df_drop_dupli.copy()
    l = []
    for _ in range(5):
        l.append(df_new.utm_adcontent.mode()[0])
        df_new = df_new.loc[df_new['utm_adcontent'] != df_new.utm_adcontent.mode()[0]]
    df_new = df_drop_dupli.loc[df_drop_dupli['utm_adcontent'].isin(l)]
    l = []
    for _ in range(15):
        l.append(df_new.utm_campaign.mode()[0])
        df_new = df_new.loc[df_new['utm_campaign'] != df_new.utm_campaign.mode()[0]]
    df_new = df_drop_dupli.loc[df_drop_dupli['utm_campaign'].isin(l)]
    return df_new


#Поиск таргетных значений
def find_target(df):
    find_list = ['sub_car_claim_click', 'sub_car_claim_submit_click',
                'sub_open_dialog_click', 'sub_custom_question_submit_click',
                'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
                'sub_car_request_submit_click']
    df_clean_all = df.copy()
    df_clean_all['target'] = df_clean_all.apply(lambda x:
                                                1 if x.event_action in find_list else 0, axis=1)
    df_clean_all = df_clean_all.drop('event_action', axis=1)
    return df_clean_all


#нормализация в  device_screen_resolution
def change_device_screen_resolution(df):
    df_clean_all = df.copy()
    df_clean_all['device_screen_resolution_w'] = df_clean_all.device_screen_resolution.apply(
        lambda x: int(x.split('x')[0]))
    df_clean_all['device_screen_resolution_h'] = df_clean_all.device_screen_resolution.apply(
        lambda x: int(x.split('x')[1]))
    df_clean_all = df_clean_all.drop('device_screen_resolution', axis=1)
    return df_clean_all


#Изменение предиктов utm_medium и utm_source
def change_utm(df):
    df_copy = df.copy()
    l_medium = ['organic', 'referral', '(none)']
    l_add_sn = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
                'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    #Органический трафик это значения типа:'organic', 'referral', '(none)', всё остальное это платный трафик
    df_copy['utm_medium'] = df_copy['utm_medium'].apply(lambda x:
                                                'organic' if x in l_medium else 'not_organic')
    #Реклама из соц. сетей хранится в списке l_add_sn
    df_copy['utm_source'] = df_copy['utm_source'].apply(lambda x:
                                                'add_sn' if x in l_add_sn else 'not_add')
    return df_copy


#Изменение пустых значений в device_os
def change_device_os(df):
    df_copy = df.copy()
    df2 = df_copy[df_copy.device_os.isnull()].copy()
    df2.loc[:, 'device_os'] = df2.apply(lambda x: 'iOS' if x.device_brand == 'Apple'
                                            else ('Windows' if x.device_category == 'desktop' else 'Android'), axis=1)
    df_copy[df_copy.device_os.isnull()] = df2
    return df_copy


#Добавление новых категориальных фич из geo_city и geo_country
def change_geo(df):
    df_copy = df.copy()
    # Переменая geo_mo_set хранит в себе названия городов Московской области
    geo_mo_set = {'Balashikha', 'Dedovsk', 'Dmitrov', 'Domodedovo', 'Ivanteyevka', 'Khimki', 'Kolomna', 'Krasnogorsk',
                  'Lytkarino', 'Lyubertsy', 'Mytishchi', 'Nakhabino', 'Naro-Fominsk', 'Odintsovo', 'Protvino',
                  'Pushkino', 'Ramenskoye', 'Reutov',
                  'Sergiyev Posad', 'Serpukhov', 'Stupino', 'Tomilino', 'Vidnoye', 'Voskresensk', 'Yegoryevsk'}
    #Создание новой колонки its_Russia для проверки из России ли клиент
    df_copy['its_Russia'] = df_copy.apply(lambda x: 1 if x.geo_country == 'Russia' else 0, axis=1)
    # Создание новой колонки its_Moscow для проверки из Москвы ли клиент
    df_copy['its_Moscow'] = df_copy.apply(lambda x: 1 if x.geo_city == 'Moscow' else 0, axis=1)
    # Создание новой колонки its_Saint_Petersburg для проверки из Питера ли клиент
    df_copy['its_Saint_Petersburg'] = df_copy.apply(lambda x: 1 if x.geo_city == 'Saint Petersburg' else 0,
                                                              axis=1)
    # Создание новой колонки its_MO для проверки из Московской области ли клиент
    df_copy['its_MO'] = df_copy.apply(lambda x: 1 if x.geo_city in geo_mo_set else 0, axis=1)
    df_copy = df_copy.drop(['geo_country', 'geo_city'], axis=1)
    return df_copy


#Удаление колонок visit_date и visit_number
def change_visit(df):
    df_copy = df.copy()
    df_copy = df_copy.drop(['visit_date', 'visit_number'], axis=1)
    return df_copy


#Открытие, объединение датасета, а также поиск таргетных значений
def loaf_df(source_ses='../data/ga_sessions.csv', source_hits='../data/ga_hits-001.csv'):
    df_ses = pd.read_csv(source_ses, low_memory=False)      #Открытые датасета data/ga_sessions.csv
    df_hits = pd.read_csv(source_hits)                      #Открытые датасета data/ga_hits-001.csv
    df_all = pd.merge(df_ses, df_hits, on=['session_id'])   #Объединение датасета df_ses и df_hits по ключу session_id
    df_clean = filter_data(df_all)                          #Удаление ненужных колонок
    df_clean = find_target(df_clean)                        #Поиск таргетных значений
    df_clean = df_clean.drop_duplicates()                   #Удаление дубликатов
    df_clean = filter_utm(df_clean)                         #Удаление ненужных значний в utm_adcontent
    df_clean = change_visit(df_clean)                       #Удаление колонок visit_date и visit_number
    df_clean = df_clean[df_clean.device_os != '(not set)']  #Удаление значения (not set) в device_os
    return df_clean


#Поиск моедли и сохранение модели в файл model/event_action.dill и сохранение датасета в файл model/out.csv
def main():
    #открытие датасета
    df = loaf_df()
    X = df.drop(['target', 'session_id', 'client_id'], axis=1)
    y = df['target']
    #Создание пайлайна для функций
    function_transformer = Pipeline(steps=[
        ('change_device_screen_resolution', FunctionTransformer(change_device_screen_resolution)),
        ('change_utm', FunctionTransformer(change_utm)),
        ('change_device_os', FunctionTransformer(change_device_os)),
        ('change_geo', FunctionTransformer(change_geo)),
    ])
    #Создание пайлайна для категориальных и числовых переменых
    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)
    #В числовых переменых пустые значения заполняются медианой, а затем происходит преоброзование при помощи StandardScaler()
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    #В категориальных переменых пустые значения заменяются на значение, которое встречается чаще,  а затем происходит преоброзование при помощи OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    #Объединение двух пайплайнов(numerical_transformer, categorical_transformer) в один
    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features),
    ])
    s = ((y == 0).sum() / (y == 1).sum()) ** 0.5
    #Создание set() с моделями для обучения
    models = (
        LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42),
        RandomForestClassifier(random_state=42),
        CatBoostClassifier(
            iterations=600,
            learning_rate=0.2,
            eval_metric='AUC',
            random_seed=42,
            auto_class_weights='SqrtBalanced'
        ),
        XGBClassifier(scale_pos_weight=s,
                      seed=42,
                      eta=0.2,
                      n_estimators=600,
                      early_stopping_rounds=10)


    )
    #Переменная best_score нужная для сохранения лучшей оценки модели
    best_score = 0
    #Переменная best_pipe нужная для сохранения лучшей модели
    best_pipe = None
    #В цикле проходим по моделям из models
    for model in models:
        #Сохраняем в переменую pipe пайплайн с текущей моделью цикла
        pipe = Pipeline(steps=[
            ('transform', function_transformer),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        #Рассчет оценки модели по методу roc_auc
        score = cross_val_score(pipe, X, y, cv=3, scoring='roc_auc')
        print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')
        #Ищем самое больше значние в score, и сохраняем это значение и этот пайплайн, если значение самое большое
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe
    #Применияем самую лучшую модель на датасете
    best_pipe.fit(X, y)
    #Сохранение части датасета для проверок
    df.sample(1000).to_csv('out.csv', index=False)
    #Выводим название модели и ее оценку
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')
    #Сохранение модели в файл event_action.dill
    with open('event_action.dill', 'wb') as file:
        dill.dump({'model': best_pipe,
                   'metadata': {
                       'name': 'Event action prediction model',
                       'author': 'Anton Korepanov',
                       'version': 1,
                       'date': datetime.datetime.now(),
                       'type': type(best_pipe.named_steps["classifier"]).__name__,
                       'roc_auc': best_score
                   }}, file)

#Запуск программы
if __name__ == '__main__':
    main()
