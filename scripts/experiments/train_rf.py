from damseg_ml.train_rf import RandomForestModel

labels = [0, 1]
model = RandomForestModel(n_estimators=100, max_features="sqrt", seed=23)
model.load_train_data("data/sample/dataset_binary.csv")
model.load_test_data("data/sample/dataset_binary.csv", labels)
model.fit_model()
model.test_model()
print(model.metrics)
