
import pickle
import pandas as pd
import os
import argparse

categorical = ['PULocationID', 'DOLocationID']

def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def save_results(output_file, df_result):
    df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
    )
    print(f"Results saved to {output_file}")

def prepare_results(df, y_pred):
    df_results = pd.DataFrame()
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_results['prediction_duration'] = y_pred
    return df_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch processing for taxi trip duration prediction')
    parser.add_argument('--year', type=int, required=True, help='Year of the trip data')
    parser.add_argument('--month', type=int, required=True, help='Month of the trip data')
    args = parser.parse_args()
    year = args.year 
    month = args.month

    dv, model = load_model()
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f"Standard deviation of predictions: {y_pred.std():.2f}")
    print(f"Mean of predictions: {y_pred.mean():.2f}")

    df_results = prepare_results(df, y_pred)

    output_file = f'yellow_tripdata_{year:04d}-{month:02d}_predictions.parquet'
    save_results(output_file, df_results)


    # file_size = os.path.getsize(output_file) / (1024 * 1024)  # size in MB
    # print(f"Size of the parquet file: {file_size:.2f} MB")

