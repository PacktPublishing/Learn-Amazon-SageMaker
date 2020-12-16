The purpose of this file is to track significant differences between the code present in the book and the code present in the repository.

Originally, all code examples have been written with SageMaker SDK v2.0.1. As this SDK keeps evolving, it's likely that breaking changes will happen. If you find that a notebook fails with a more recent SDK, please revert to SDK v2.0.1 by running 'pip -q install sagemaker==2.0.1' at the beginning of the notebook.

Feel free to open an issue and - even better - to submit a PR with a proposed fix :) I'll try to keep the notebooks as up to date as possible.

As of Dec. 16th 2020, I'm checking all examples with the latest SageMaker SDK (v2.19.0). Chapters that have been completely checked will have an 'sdk_version' file. I'm hoping to complete all checks in a matter of weeks.

The rest of this file will list issues I'm aware of, and how to work around them.

***
The content_type attribute in sagemaker.predictor.Predictor is now gone. Thus, lines similar to the one below will cause an error:
```
xgb_predictor.content_type = 'text/csv'
```
If you use built-in serializers and deserializers, simply remove the line as it's not needed anymore.

I don't have a solution for custom serializers yet, e.g. the PCA notebook in Chapter 4. Please use SDK v2.0.1 in the meantime.

--> FIXED, notebooks have been updated.

***

The current SageMaker SDK v2 doesn't have a serializer for 'application/x-image' content, making it impossible (?) to use predict(). The easiest option to predict with models that require this content type is to use boto3 instead, as shown in chapter 5 examples.

**
