import urllib.request
import json

from PIL import Image

def predict_from_api(api_url, image_url):
  request_header = {
      "Content-Type": "application/json",
  }
  request_data = json.dumps({
    "image_url": image_url
  })

  request = urllib.request.Request(api_url, data=request_data.encode(), method="POST", headers=request_header)
  with urllib.request.urlopen(request) as response:
    label = json.loads(response.read())["label"]

  return label

label = predict_from_api("http://127.0.0.1:5000/predict", "https://stmaaprodfwsite.blob.core.windows.net/assets/sites/1/2019/06/1-Scald-has-been-brought-under-control-by-footbathing-and-ensuring-grass-isnt-too-long-c-no-credit.jpg")

print(label)