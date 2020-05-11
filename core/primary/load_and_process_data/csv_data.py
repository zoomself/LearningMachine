import tensorflow as tf
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

train_csv_file = "C:\\Users\\Administrator\\.keras\\datasets\\train.csv"
test_csv_file = "C:\\Users\\Administrator\\.keras\\datasets\\eval.csv"

CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']

numeric_columns = ["age", "fare"]
categorical_columns = ["sex", "class", "deck", "embark_town", "alone", "n_siblings_spouses", "parch"]
label_column = "survived"


def show_data():
    df = pd.read_csv(train_csv_file, comment="\n")
    print(df.head())
    for i, cc in enumerate(categorical_columns):
        plt.subplot(3, 3, i + 1)
        sb.barplot(x=cc, y=label_column, data=df)
    plt.show()

    for i, nc in enumerate(numeric_columns):
        plt.subplot(1, 2, i + 1)
        sb.distplot(df[nc])
    plt.show()


def create_data_sets(batch_size):
    _ds_train = tf.data.experimental.make_csv_dataset(
        file_pattern=train_csv_file,
        batch_size=batch_size,
        column_names=CSV_COLUMNS,
        num_epochs=1,
        label_name="survived"
    )

    _ds_test = tf.data.experimental.make_csv_dataset(
        file_pattern=test_csv_file,
        batch_size=batch_size,
        column_names=CSV_COLUMNS,
        num_epochs=1,
        label_name="survived"
    )
    return _ds_train, _ds_test


def create_deep_feature_columns():
    df = pd.read_csv(train_csv_file, comment="\n")
    fcs = []
    for cc in categorical_columns:
        ccv = tf.feature_column.categorical_column_with_vocabulary_list(cc, vocabulary_list=df[cc].unique())
        ic = tf.feature_column.indicator_column(categorical_column=ccv)
        fcs.append(ic)

    fcs.append(tf.feature_column.numeric_column("fare"))
    fcs.append(
        tf.feature_column.bucketized_column(source_column=tf.feature_column.numeric_column("age"),
                                            boundaries=[20, 40, 60, 80]))
    print(fcs)
    return fcs


def create_wide_feature_columns():
    df = pd.read_csv(train_csv_file, comment="\n")
    fcs = []
    for cc in categorical_columns:
        if not cc == "sex":
            fcs.append(tf.feature_column.categorical_column_with_vocabulary_list(cc, vocabulary_list=df[cc].unique()))

    cc_sex = tf.feature_column.categorical_column_with_vocabulary_list("sex", vocabulary_list=df["sex"].unique())
    fcs.append(cc_sex)
    bc_age = tf.feature_column.bucketized_column(source_column=tf.feature_column.numeric_column("age"),
                                                 boundaries=[20, 40, 60, 80])
    fcs.append(bc_age)
    fcs.append(
        tf.feature_column.bucketized_column(source_column=tf.feature_column.numeric_column("fare"),
                                            boundaries=[100, 200, 300, 400, 500]))

    fcs.append(tf.feature_column.crossed_column(keys=[cc_sex, bc_age], hash_bucket_size=100))

    print(tf.feature_column.make_parse_example_spec(fcs))

    return fcs


def create_keras_model():
    _model = tf.keras.models.Sequential([
        tf.keras.layers.DenseFeatures(feature_columns=create_deep_feature_columns()),
        tf.keras.layers.Dense(16, "relu"),
        tf.keras.layers.Dense(8, "relu"),
        tf.keras.layers.Dense(4, "relu"),
        tf.keras.layers.Dense(1, tf.keras.activations.sigmoid)
    ])
    _model.compile("adam", tf.keras.losses.binary_crossentropy, ["acc"])
    return _model


def create_estimator_model(is_deep=True):
    _model = None
    if is_deep:
        _model = tf.estimator.DNNClassifier(hidden_units=[16, 8, 4], feature_columns=create_deep_feature_columns())
    else:
        _model = tf.estimator.LinearClassifier(feature_columns=create_wide_feature_columns())
    return _model


def create_model(keras=True, is_deep=True):
    if keras:
        return create_keras_model()
    else:
        return create_estimator_model(is_deep)


def input_fn(is_train=True, batch_size=32):
    def fn():
        _ds_train, _ds_test = create_data_sets(batch_size)
        if is_train:
            return _ds_train
        else:
            return _ds_test

    return fn


if __name__ == '__main__':
    show_data()
    is_keras_model = False
    is_deep = False
    model = create_model(keras=is_keras_model, is_deep=is_deep)
    if is_keras_model:
        ds_train, ds_test = create_data_sets(32)
        model.fit(ds_train, epochs=100, validation_data=ds_test)
        model.summary()
    else:
        print("\n-------train------- ")
        model.train(input_fn=input_fn(), max_steps=100)
        print("\n-------evaluate------- ")
        evaluate_result = model.evaluate(input_fn=input_fn(False))
        print(evaluate_result)
        # result=model.predict(input_fn=input_fn(False))
        # print(result)
        # print(list(result))

        pred_dicts = list(model.predict(input_fn(False)))
        print(pred_dicts)
        probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
        probs.plot(kind='hist', bins=20, title='predicted probabilities')
        plt.show()
