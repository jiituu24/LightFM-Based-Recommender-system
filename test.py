print("Starting")
try:
    print("Importing")
    from utils.data_loader import load_data
    from lightfm import LightFM
    print("Loading data")
    df, matrix = load_data()
    print("Data loaded")
    model = LightFM(loss='logistic')
    print("Fitting")
    model.fit(matrix, epochs=10)
    print('fit done')
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()