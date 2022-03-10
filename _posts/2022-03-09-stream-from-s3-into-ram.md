---
layout: post
title: "ML on AWS Lambda: Stream models from S3 into RAM"
date: 2022-03-09 12:55:00 -0300
tags:
  - ml-ops
  - tech
---

## Introduction

I'd like to share a "pattern" that emerged in my work deploying ML inference Python code to AWS Lambda. To do this, first allow me to offer three quick observations that will help us tie this pattern together. 

### On Lambda and its storage quota
AWS Lambda can be a convenient way of deploying production code for many use cases. However, as can expected from such high-level services, it also comes with some limitations. In particular, it has a strict storage space quota: Lambda provides a filesystem on an ephemeral storage space for you to use under `/tmp` in each of your function executions, but it has a hard limit of 512 MB in size (see [docs](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)).

### On downloading objects from S3

When you need to get an object from S3, it often makes sense to just download it to disk. This seems intuitive and easy, and is a very common practice. You may even do this when you only need the object in RAM, by downloading it first to your filesystem and then loading it into your program from there:

```python
import boto3
s3_resource = boto3.resource('s3')

s3_object = s3_resource.Object('my-bucket', 'my_file.txt')
with open('temp_my_file.txt', 'w') as f:
    s3_object.download_fileobj(f)

# do something with temp_my_file.txt, remember to delete it if appropriate.
```

However, if you're going to download the S3 object to disk only to then load and use it in RAM, then you'd probably prefer it if there was a way to get the object into RAM directly.

### On using popular ML libraries  

Popular libraries that offer pre-trained models (such as [Hugging Face](https://huggingface.co/models), [OpenAI's CLIP](https://github.com/openai/CLIP) or [JaidedAI's EasyOCR](https://github.com/JaidedAI/EasyOCR)) usually use your filesystem to download models to and to use as cache. Also quite common is for single models or the conjunction of several models used in an inference program to exceed 512 MB.

## Piecing it together

Deploying ML inference code is a worthy use case for AWS Lambda. But, in light of popular libraries relying on disk space for getting you pre-trained models, if you use enough of these then your Lambdas are liable to eventually run into the 512 MB limit. This would also be the case if you don't download pre-trained models but you do rely on getting models (either pre-downloaded, trained by you or whatever the case may be) from S3 and you're used to interacting with S3 the way I showed before.

So, you're probably able to piece the idea together by yourself: if only there was a way to stream S3 objects directly into memory, we'd be good to go and on our way. In that case, whenever your Lambda-bound code needs to use models that are large enough, you could serialize them (e.g by using `pickle` or library-specific serialization options), upload them to S3 and then use this hypothetical method to load them directly into RAM without worrying too much about the storage space quota. This also would be true whether you're using pre-trained models such as the ones mentioned before, which you may pre-download and pickle, or models you train yourself.

I'm probably not spoiling anything if I tell you that there is such a way indeed. To show you how it may work with an example, let's assume we're using PyTorch and we've pickled and uploaded the `state_dict` of an instance of `Model` to S3 at `models-bucket/model_state_dict.pkl`.

Starting with the code from before, we add some imports:
```python
import io
import pickle

import boto3
s3_resource = boto3.resource('s3')
```

We then stream the pickled `state_dict` into a variable rather than save to disk:
```python
bytes_stream = io.BytesIO()
s3_object = s3_resource.Object('models-bucket', 'model_state_dict.pkl')
s3_object.download_fileobj(bytes_stream)
pickled_state_dict = bytes_stream.getvalue()
```

Finally, we load the model:
```python
state_dict = pickle.loads(pickled_state_dict)
model = Model()
model.load_state_dict(state_dict)
model.eval()

# Have fun with model
```

### A final note

Streaming objects from S3 into RAM is nothing new nor is it too much of an obscure functionality if you look at the docs closely enough. However, it seems worth highlighting in the particular context of deploying ML code to Lambda as this is an increasingly popular practice and this little pattern easily solves an issue that anyone starting to adopt it is bound to encounter.


