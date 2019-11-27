# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Utilities to use an Azure Machine Learning workspace to track the performance test results
over time.
"""

import logging
import os

from azureml.core import Workspace
from azureml.core.compute import AmlCompute
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml._base_sdk_common.common import resource_client_factory
from azure.mgmt.resource.resources.models import ResourceGroup


TENANT_ID = os.getenv("TENANT_ID")
SERVICE_PRINCIPAL_ID = os.getenv("SERVICE_PRINCIPAL_ID")
SERVICE_PRINCIPAL_PASSWORD = os.getenv("SERVICE_PRINCIPAL_PASSWORD")
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP_NAME = "fairlearnperf"
WORKSPACE_NAME = "fairlearnperf"
WORKSPACE_LOCATION = "canadacentral"
COMPUTE_TARGET_CONFIG = AmlCompute.provisioning_configuration(
    min_nodes=0,
    max_nodes=10,
    vm_size="STANDARD_DS2_V2",
    vm_priority="dedicated")

logger = logging.getLogger(__file__)


def get_workspace():
    logger.info("Logging in as service principal.")
    auth = ServicePrincipalAuthentication(TENANT_ID, SERVICE_PRINCIPAL_ID, SERVICE_PRINCIPAL_PASSWORD)
    logger.info("Successfully logged in as service principal.")

    logger.info("Ensuring resource group {} exists.".format(RESOURCE_GROUP_NAME))
    resource_management_client = resource_client_factory(auth, SUBSCRIPTION_ID)
    resource_group_properties = ResourceGroup(location=WORKSPACE_LOCATION)
    resource_management_client.resource_groups.create_or_update(WORKSPACE_NAME, resource_group_properties)
    logger.info("Ensured resource group {} exists.".format(RESOURCE_GROUP_NAME))

    logger.info("Ensuring workspace {} exists.".format(WORKSPACE_NAME))
    workspace = Workspace.create(name=WORKSPACE_NAME, auth=auth, subscription_id=SUBSCRIPTION_ID,
                                resource_group=RESOURCE_GROUP_NAME, location=WORKSPACE_LOCATION,
                                create_resource_group=False, exist_ok=True,
                                default_cpu_compute_target=COMPUTE_TARGET_CONFIG)
    logger.info("Ensured workspace {} exists.".format(WORKSPACE_NAME))
    return workspace
