import lzma
import base64
import struct
from functools import lru_cache
from urllib.parse import urlencode

import cv2
import torch
import numpy as np
from diffusers.models import AutoencoderKL

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, PlainTextResponse
import os

cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", cache_dir=cache_dir).to(device)
torch.set_grad_enabled(False)
app = FastAPI()

print(f"Running on: {device}")


def _encode(img_bytes: bytes) -> bytes:
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x, y = img.shape[:2]
    新img = img
    limit = 200000
    if x * y > limit:
        r = (x * y / limit) ** 0.5
        新img = cv2.resize(img, (int(y / r), int(x / r)), interpolation=cv2.INTER_AREA)
    新img = torch.from_numpy(新img.transpose((2, 0, 1))).float().div(255).unsqueeze(0).to(device)
    iimg = vae.encode(新img)
    v = iimg.latent_dist.mean
    iiimg = v.detach().cpu().numpy()
    q, w = np.percentile(iiimg, [0.1, 99.9])
    iiimg = (((np.clip(iiimg, q, w) - q) / (w - q)) * 255).astype('uint8')
    iiimg = iiimg.squeeze(0)  # CHW
    # print(len(iiimg.tobytes()))
    succ, comp = cv2.imencode(".webp", iiimg.reshape(-1, iiimg.shape[-1]), [cv2.IMWRITE_WEBP_QUALITY, 95])
    if not succ:
        raise ValueError("Latent Compression failed.")
    meta = struct.pack('ffII', q, w, iiimg.shape[1], iiimg.shape[2])
    # b = lzma.compress(meta + comp.reshape(-1).tobytes(), preset=9, format=lzma.FORMAT_ALONE)
    b = meta + comp.reshape(-1).tobytes()
    # print(len(b), len(lzma.compress(meta + iiimg.reshape(-1).tobytes(), preset=9, format=lzma.FORMAT_ALONE)))
    return b


@lru_cache(maxsize=128)
def _decode(e_bytes: bytes) -> bytes:
    # data = lzma.decompress(e_bytes)
    data = e_bytes
    q, w, x, y = struct.unpack('ffII', data[:16])
    pbytes = np.frombuffer(data[16:], dtype='uint8')
    iiimg = cv2.imdecode(pbytes, cv2.IMREAD_GRAYSCALE).reshape(4, x, y)
    v = (torch.from_numpy(iiimg).float().div(255).unsqueeze(0) * (w - q) + q).to(device)
    img_2 = vae.decode(v).sample
    img_2 = img_2.squeeze(0).mul(255).permute(1, 2, 0).detach().cpu().numpy()
    img_2 = np.clip(img_2, 0, 255).astype('uint8')
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR)
    return cv2.imencode('.webp', img_2)[1].tobytes()


@app.get("/")
async def 家():
    t = '''
    <form action="/upload" method="POST" enctype="multipart/form-data">
        <input type="file" id="myFile" name="file">
        <input type="submit">
    </form>
    '''
    return HTMLResponse(content=t)


@app.post('/upload')
async def 上(request: Request):
    form = await request.form()
    contents = await form['file'].read()
    b = _encode(contents)
    return PlainTextResponse(content='/image?' + urlencode({'q': base64.urlsafe_b64encode(b)}))


@app.get("/image")
async def 下(q: str):
    b = base64.urlsafe_b64decode(q)
    bb = _decode(b)
    return Response(content=bb, media_type="image/png")
