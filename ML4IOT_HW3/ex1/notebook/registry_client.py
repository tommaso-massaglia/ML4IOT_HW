import argparse
import requests
import base64
import zlib

def iscompressed(data):
    result = True
    try:
        s = zlib.decompress(data)
    except zlib.error:
        result = False  
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--request', type=str, choices=['PUT', 'POST', 'GET'], help="What method to invoke", required=True)
parser.add_argument('--modelpath', type=str, required=False)
parser.add_argument('--modelname', type=str, required=False)
parser.add_argument('--predmodel', type=str, required=False, choices=['CNN', 'MLP'])
parser.add_argument('--tthres', type=float,default=0.1, required=False)
parser.add_argument('--hthres', type=float,default=0.2, required=False)
parser.add_argument('--duration', type=int,default=60, required=False)
args = parser.parse_args()

url = 'http://raspberrypi.local:8080'

if args.request=="POST":
    modelpath = args.modelpath
    model =  open(modelpath, "rb").read()
    if iscompressed(model):
        body = {'model': str(base64.b64encode(model)), 'modelname': args.modelname}
    else:
        body = {'model': str(base64.b64encode(zlib.compress(model,5))), 'modelname': args.modelname}
    r = requests.post(url, json=body)
    
if args.request=="GET":
    r = requests.get(url, "")
    
if args.request=='PUT':
    j = {
        "model":args.predmodel,
        "tthres":args.tthres,
        "hthres":args.hthres,
        "duration":args.duration,
    }
    r = requests.put(url, json=j)

if r.status_code == 200:
    body = r.json()
    print(body)
else:
    print('Error:', r.status_code)