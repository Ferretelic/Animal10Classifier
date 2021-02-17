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

label = predict_from_api("http://127.0.0.1:5000/predict", "https://images.theconversation.com/files/230552/original/file-20180803-41366-8x4waf.JPG?ixlib=rb-1.1.0&q=45&auto=format&w=926&fit=clip")

print(label)