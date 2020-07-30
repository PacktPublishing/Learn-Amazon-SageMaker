import time

from aws_cdk import (
    aws_sagemaker as sagemaker,
    core
)

class SagemakerEndpoint(core.Stack):
    def __init__(self, app: core.App, id: str, **kwargs) -> None:
        timestamp = '-'+time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        super().__init__(app, id, **kwargs)

        model = sagemaker.CfnModel(
            scope=self,
            id="my_model",
            execution_role_arn=self.node.try_get_context("role_arn"),
            containers=[
              {"image": self.node.try_get_context("image"),
              "modelDataUrl": self.node.try_get_context("model_data_url")} 
            ],
            model_name=self.node.try_get_context("model_name")+timestamp
        )

        epc = sagemaker.CfnEndpointConfig(
            scope=self,
            id="my_epc",
            production_variants=[
                    {"modelName": model.model_name,
                    "variantName": "variant-1",
                    "initialVariantWeight": 1.0,
                    "initialInstanceCount": 1,
                    "instanceType": self.node.try_get_context("instance_type")}
            ],
            endpoint_config_name=self.node.try_get_context("epc_name")+timestamp
        )

        epc.add_depends_on(model)

        ep = sagemaker.CfnEndpoint(
            scope=self,
            id="my_ep",
            endpoint_config_name=epc.endpoint_config_name,
            endpoint_name=self.node.try_get_context("ep_name")+timestamp
        )

        ep.add_depends_on(epc)


app = core.App()
SagemakerEndpoint(app, "SagemakerEndpoint", env={'region': 'us-east-1'})
app.synth()
