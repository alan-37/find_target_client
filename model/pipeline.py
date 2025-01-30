import datetime
import dill as dill
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector


#Удаление ненужных колонок и дубликатов
def filter_data(df):
    columns_to_drop = {'event_value', 'event_label', 'hit_referer','device_model', 'hit_type',
        'event_label', 'hit_time', 'visit_time', 'hit_date', 'hit_number',
        'hit_date','hit_page_path', 'event_category'}
    df_copy = df.copy()
    df_copy = df_copy.drop(columns_to_drop, axis=1)
    return df_copy


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


#Изменение предиктов utm_*
def change_utm(df):
    df_copy = df.copy()
    l_medium = ['organic', 'referral', '(none)']
    l_add_sn = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
                'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    df_copy['utm_medium'] = df_copy['utm_medium'].apply(lambda x:
                                                'organic' if x in l_medium else 'inorganic')
    df_copy['utm_source'] = df_copy['utm_source'].apply(lambda x:
                                                'add_sn' if x in l_add_sn else 'not_add')
    return df_copy


#Добавление значений в device_os
def change_device_os(df):
    df_copy = df.copy()
    df2 = df_copy[df_copy.device_os.isnull()]
    df2.loc[:, 'device_os'] = df2.apply(lambda x: 'iOS' if x.device_brand == 'Apple'
                                            else ('Windows' if x.device_category == 'desktop' else 'Android'), axis=1)
    df_copy[df_copy.device_os.isnull()] = df2
    return df_copy


#Добавление новых категориальных фич из geo_city и geo_country
def change_geo(df):
    df_copy = df.copy()
    geo_mo_set = {'Balashikha', 'Dedovsk', 'Dmitrov', 'Domodedovo', 'Ivanteyevka', 'Khimki', 'Kolomna', 'Krasnogorsk',
                  'Lytkarino', 'Lyubertsy', 'Mytishchi', 'Nakhabino', 'Naro-Fominsk', 'Odintsovo', 'Protvino',
                  'Pushkino', 'Ramenskoye', 'Reutov',
                  'Sergiyev Posad', 'Serpukhov', 'Stupino', 'Tomilino', 'Vidnoye', 'Voskresensk', 'Yegoryevsk'}
    df_copy['its_Russia'] = df_copy.apply(lambda x: 1 if x.geo_country == 'Russia' else 0, axis=1)
    df_copy['its_Moscow'] = df_copy.apply(lambda x: 1 if x.geo_city == 'Moscow' else 0, axis=1)
    df_copy['its_Saint_Petersburg'] = df_copy.apply(lambda x: 1 if x.geo_city == 'Saint Petersburg' else 0,
                                                              axis=1)
    df_copy['its_MO'] = df_copy.apply(lambda x: 1 if x.geo_city in geo_mo_set else 0, axis=1)
    df_copy = df_copy.drop(['geo_country', 'geo_city'], axis=1)
    return df_copy


#нормализация в visit_date
def change_visit(df):
    df_copy = df.copy()
    df_copy = df_copy.drop(['visit_date', 'visit_number'], axis=1)
    return df_copy


#Открытие, объединение датасета, а также поиск таргетных значений
def loaf_df(source_ses='../data/ga_sessions.csv', source_hits='../data/ga_hits-001.csv'):
    df_ses = pd.read_csv(source_ses, low_memory=False)
    df_hits = pd.read_csv(source_hits)
    df_all = pd.merge(df_ses, df_hits, on=['session_id'])
    df_clean = filter_data(df_all)
    df_clean = find_target(df_clean)
    df_clean = df_clean.drop_duplicates()
    df_clean = filter_utm(df_clean)
    df_clean = change_visit(df_clean)
    return df_clean


#Поиск моедли и сохранение модели в файл model/event_action.dill
# и сохранение датасета в файл model/out.csv
def main():
    df = loaf_df()
    X = df.drop(['target', 'session_id', 'client_id'], axis=1)
    y = df['target']
    function_transformer = Pipeline(steps=[
        ('change_device_screen_resolution', FunctionTransformer(change_device_screen_resolution)),
        ('change_utm', FunctionTransformer(change_utm)),
        ('change_device_os', FunctionTransformer(change_device_os)),
        ('change_geo', FunctionTransformer(change_geo)),
    ])

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features),
    ])

    models = (
        LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42),
        RandomForestClassifier(random_state=42),
        MLPClassifier(random_state=42)
    )

    best_score = 0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('transform', function_transformer),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe
    best_pipe.fit(X, y)
    X.to_csv('out.csv', index=False)
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')
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


if __name__ == '__main__':
    main()
