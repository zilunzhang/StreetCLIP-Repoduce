# StreetCLIP Reproduction

I am trying to reproduce the [StreetCLIP](https://arxiv.org/pdf/2302.00275.pdf).

- [Paper](https://arxiv.org/pdf/2302.00275.pdf)

- [Model](https://huggingface.co/geolocal/StreetCLIP)

- [App](https://huggingface.co/spaces/NemesisAlm/GeolocationCountryClassification)

- [Blog](https://osintteam.blog/geolocation-and-ai-with-streetclip-introduction-country-classification-and-building-a-web-e13bd0e6d857)

- [Youtube (Geoguesser)](https://www.youtube.com/watch?v=ts5lPDV--cU)

## Experiment Result

* Two-stage linear prob

  *  **"A Street View photo in {country}."**

  *  **"A Street View photo from {city}."**

### Data 

* [IM2GPS](http://graphics.cs.cmu.edu/projects/im2gps/): [Download](http://graphics.cs.cmu.edu/projects/im2gps/gps_query_imgs.zip)

* [IM2GPS3K](https://github.com/lugiavn/revisiting-im2gps/): [Download](http://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip)

* [Original Matlab implementation for inference](https://github.com/lugiavn/revisiting-im2gps/blob/master/main_im2gps_test.m)

* [Alternative implementation](https://github.com/TIBHannover/GeoEstimation/blob/master/classification/utils_global.py)

### Result on IM2GPS3K
* n=2997

|Model|Source|1KM|25KM|200KM|750KM|2,500KM|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|CLIP@ViT-L-14-336|[Paper](https://arxiv.org/pdf/2302.00275.pdf)|-|19.5 | 34.0|60.0 |78.1|
|CLIP@ViT-L-14-336|[OpenAI's CLIP-reproduce](https://github.com/openai/CLIP)|4.07|20.09|31.90|54.72|72.07|
|StreetCLIP@ViT-L-14-336|[Paper](https://arxiv.org/pdf/2302.00275.pdf)|-|22.4 |37.4|61.3 |80.4|
|StreetCLIP@ViT-L-14-336|[StreetCLIP-reproduce](https://huggingface.co/geolocal/StreetCLIP/tree/main)|4.24|21.79|34.73|55.52|74.84|
|CLIP@ViT-B-32|[OpenAI's CLIP](https://github.com/openai/CLIP)|1.67|8.88|14.65|32.87|53.72|
|CLIP@ViT-H-14|[OpenCLIP](https://github.com/mlfoundations/open_clip)|3.94|18.69|30.60|51.95|71.10|

### Result on IM2GPS
* n=237

|Model|Source|1KM|25KM|200KM|750KM|2,500KM|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|CLIP@ViT-L-14-336|[Paper](https://arxiv.org/pdf/2302.00275.pdf)|-|27.0 | 42.2|71.7| 86.9|
|CLIP@ViT-L-14-336|[OpenAI's CLIP-reproduce](https://github.com/openai/CLIP)|4.64|26.58|40.08|63.71|80.17|
|StreetCLIP@ViT-L-14-336|[Paper](https://arxiv.org/pdf/2302.00275.pdf)|-|28.3 | 45.1|74.7 |88.2|
|StreetCLIP@ViT-L-14-336|[StreetCLIP-reproduce](https://huggingface.co/geolocal/StreetCLIP/tree/main)|5.49|28.27|42.62|67.51|80.17|
|CLIP@ViT-B-32|[OpenAI's CLIP](https://github.com/openai/CLIP)|2.11|16.46|26.58|46.41|66.24|
|CLIP@ViT-H-14|[OpenCLIP](https://github.com/mlfoundations/open_clip)|5.49|29.54|44.30|65.82|79.75|




