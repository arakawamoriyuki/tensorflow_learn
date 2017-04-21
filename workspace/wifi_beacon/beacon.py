import argparse

from beacon.datasets import fetch_datasets, save_datasets, load_datasets
from beacon.model import get_model, save_model


def sampling(**kwargs):
    X, y = fetch_datasets(label=kwargs['label'], size=kwargs['size'])
    save_datasets(X, y, path=kwargs['path'])
    training(model_path=kwargs['model_path'], datasets_path=kwargs['datasets_path'])

def training(**kwargs):
    X, y = load_datasets(path=kwargs['datasets_path'])
    model = get_model(path=kwargs['model_path'])
    model.fit(X, y)
    save_model(model, path=kwargs['model_path'])

def predict(**kwargs):
    X, _ = fetch_datasets(size=1)
    model = get_model(path=kwargs['model_path'])
    return model.predict(X[0])[0]

def percentage(**kwargs):
    X, _ = fetch_datasets(size=1)
    model = get_model(path=kwargs['model_path'])
    return {x: y for x, y in zip(model.classes_, model.predict_proba(X[0])[0])}

def labels(**kwargs):
    model = get_model(path=kwargs['model_path'])
    return model.classes_


def main(args):
    globals()[args.function](
        model_path=args.model,
        datasets_path=args.datasets,
        label=args.label,
        size=args.size
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('function',
        default='sampling',
        help='sampling|predict|percentage|labels')
    parser.add_argument('-m', '--model',
        default='models/model.pkl',
        help='model path')
    parser.add_argument('-d', '--datasets',
        default='datasets/datasets.csv',
        help='datasets path')
    parser.add_argument('-s', '--size',
        default=100,
        help='datasets fetch size')
    parser.add_argument('-l', '--label',
        default='default place',
        help='datasets label')
    args = parser.parse_args()
    main(args)





#