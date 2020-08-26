# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from bokeh.plotting import figure, show
from gql import Client
from gql.transport.requests import RequestsHTTPTransport

SECRET_ENV_VAR = "GITHUB_GRAPHQL"


def get_client(token):
    # Note that for an Auth token, we have to have "token" in the header value
    transport = RequestsHTTPTransport(
        url='https://api.github.com/graphql',
        headers={'Authorization': "token {0}".format(token)})
    client = Client(
        transport=transport,
        fetch_schema_from_transport=True)
    return client


def get_month_string(target_date):
    return "{0}-{1:02}".format(target_date.year, target_date.month)


def create_no_fetch_data_plot():
    xs = ["Unable", "to", "fetch", "data"]
    ys = range(len(xs))
    p = figure(x_range=xs,
               plot_height=512,
               plot_width=832,
               title="Stars",
               toolbar_location=None,
               tools="")
    p.vbar(x=xs, top=ys, width=0.5)

    show(p)
