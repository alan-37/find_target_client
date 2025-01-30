import pandas as pd


def main():
    df_random = pd.read_csv('../model/out.csv').sample()
    df_random.to_csv('find.csv', index=False)
    print("Успех")


if __name__ == '__main__':
    main()