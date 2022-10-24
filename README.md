# Learn Amazon SageMaker

<a href="https://www.packtpub.com/product/learn-amazon-sagemaker/9781800208919?utm_source=github&utm_medium=repository&utm_campaign=9781800208919"><img src="https://static.packt-cdn.com/products/9781800208919/cover/smaller" alt="Learn Amazon SageMaker" height="256px" align="right"></a>

This is the code repository for [Learn Amazon SageMaker](https://www.packtpub.com/product/learn-amazon-sagemaker/9781800208919?utm_source=github&utm_medium=repository&utm_campaign=9781800208919), published by Packt.

**A guide to building, training, and deploying machine learning models for developers and data scientists**

## What is this book about?
Amazon SageMaker enables you to quickly build, train, and deploy machine learning (ML) models at scale, without managing any infrastructure. It helps you focus on the ML problem at hand and deploy high-quality models by removing the heavy lifting typically involved in each step of the ML process. This book is a comprehensive guide for data scientists and ML developers who want to learn the ins and outs of Amazon SageMaker.

You’ll understand how to use various modules of SageMaker as a single toolset to solve the challenges faced in ML. As you progress, you’ll cover features such as AutoML, built-in algorithms and frameworks, and the option for writing your own code and algorithms to build ML models. Later, the book will show you how to integrate Amazon SageMaker with popular deep learning libraries such as TensorFlow and PyTorch to increase the capabilities of existing models. You’ll also learn to get the models to production faster with minimum effort and at a lower cost. Finally, you’ll explore how to use Amazon SageMaker Debugger to analyze, detect, and highlight problems to understand the current model state and improve model accuracy.

By the end of this Amazon book, you’ll be able to use Amazon SageMaker on the full spectrum of ML workflows, from experimentation, training, and monitoring to scaling, deployment, and automation.

In this repo, you will find the code examples used in the book. I also include here parts of the code omitted in the book, such as the data visualization styling, additional formatting, etc.

This book covers the following exciting features: 
* Create and automate end-to-end machine learning workflows on Amazon Web Services (AWS)
* Become well-versed with data annotation and preparation techniques
* Use AutoML features to build and train machine learning models with AutoPilot
* Create models using built-in algorithms and frameworks and your own code
* Train computer vision and NLP models using real-world examples
* Cover training techniques for scaling, model optimization, model debugging, and cost optimization
* Automate deployment tasks in a variety of configurations using SDK and several automation tools

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/180020891X) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders.

The code will look like the following:
```
od = sagemaker.estimator.Estimator(
	container,
	role,
	train_instance_count=2,
	train_instance_type='ml.p3.2xlarge',
	train_use_spot_instances=True,
	train_max_run=3600,               # 1 hour
	train_max_wait=7200,              # 2 hours
	output_path=s3_output)

```

**Following is what you need for this book:**
This book is for software engineers, machine learning developers, data scientists, and AWS users who are new to using Amazon SageMaker and want to build high-quality machine learning models without worrying about infrastructure. Knowledge of AWS basics is required to grasp the concepts covered in this book more effectively. Some understanding of machine learning concepts and the Python programming language will also be beneficial. 

With the following software and hardware list you can run all code files present in the book (Chapter 1-13).

### Software and Hardware List

| Chapter  | Software required                                                                    | OS required                        |
| -------- | -------------------------------------------------------------------------------------| -----------------------------------|
| 1 - 13   |   Amazon Web Services                                						          | Windows, Mac OS X, and Linux (Any) |

All code examples in the book are based on SageMaker SDK v2 (https://sagemaker.readthedocs.io/en/stable/overview.html), which was released in August 2020. SDK v1 examples are also included for reference.


We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781800208919_ColorImages.pdf).


### Related products <Other books you may enjoy>
* Mastering Machine Learning on AWS [[Packt]](https://www.packtpub.com/product/mastering-machine-learning-on-aws/9781789349795) [[Amazon]](https://www.amazon.com/dp/1789349796)

* Hands-On Artificial Intelligence on Amazon Web Services [[Packt]](https://www.packtpub.com/product/hands-on-artificial-intelligence-on-amazon-web-services/9781789534146) [[Amazon]](https://www.amazon.com/dp/1789534143)

## Get to Know the Author
**Julien Simon** 
is a principal AI and machine learning developer advocate. He focuses on helping developers and enterprises to bring their ideas to life. He frequently speaks at conferences and blogs on AWS blogs and on Medium. Prior to joining AWS, Julien served for 10 years as CTO/VP of engineering in top-tier web start-ups where he led large software and ops teams in charge of thousands of servers worldwide. In the process, he fought his way through a wide range of technical, business, and procurement issues, which helped him gain a deep understanding of physical infrastructure, its limitations, and how cloud computing can help.


### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.
### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781800208919">https://packt.link/free-ebook/9781800208919 </a> </p>