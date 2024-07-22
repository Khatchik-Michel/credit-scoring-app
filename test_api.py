import unittest
from unittest.mock import patch
import pandas as pd
import json
import requests

class TestFlaskAPI(unittest.TestCase):
    api_ip = 'http://127.0.0.1:5001'  #deuxieme IP si executer pas sur la même machine

    def test_change_model_endpoint_400(self):
        model_id = None
        # Create data for the POST request
        data = {"model_id": model_id}
        response = requests.post(f'{self.api_ip}/new_model', data=json.dumps(data), headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 400)

    def test_predict_endpoint_with_valid_data(self):
        X = pd.read_csv("./test/X_head", index_col=0)
        data = X.to_json(orient='records')
        response = requests.post(f'{self.api_ip}/predict', data=data, headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 200)

    def test_get_dataset_version(self):
        response = requests.get(f'{self.api_ip}/version')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "2.0")

    def test_get_dataset_threshold(self):
        response = requests.get(f'{self.api_ip}/threshold')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "0.1")

if __name__ == '__main__':
    unittest.main()
