import React from "react";
import ReactDOM from "react-dom";
import axios from "axios";
import "babel-polyfill";

import { FairnessWizard } from "fairlearn-dashboard";


const RenderDashboard = (divId, data) => {
  let calculateMetrics = (postData) => {
    if (data.withCredentials) {
      var headers_data = {
        Accept:
          "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Content-Type": "application/json"
      };
      axios.defaults.withCredentials = true;
      var axios_options = { headers: headers_data, withCredentials: true };
      return axios
        .post(data.metricsUrl, JSON.stringify(postData), axios_options)
        .then((response) => {
          return response.data;
        })
        .catch(function (error) {
          throw new Error(error);
        });
    } else {
      return fetch(data.metricsUrl, {
        method: "post",
        body: JSON.stringify(postData),
        headers: {
          "Content-Type": "application/json"
        }
      })
        .then((resp) => {
          if (resp.status >= 200 && resp.status < 300) {
            return resp.json();
          }
          return Promise.reject(new Error(resp.statusText));
        })
        .then((json) => {
          if (json.error !== undefined) {
            throw new Error(json.error);
          }
          return Promise.resolve(json.data);
        });
    }
  };

  ReactDOM.render(
    <FairnessWizard
      dataSummary={{ featureNames: data.features, classNames: data.classes }}
      testData={data.dataset}
      predictedY={data.predicted_ys}
      trueY={data.true_y}
      modelNames={data.model_names}
      precomputedMetrics={data.precomputedMetrics}
      precomputedFeatureBins={data.precomputedFeatureBins}
      customMetrics={data.customMetrics}
      predictionType={data.predictionType}
      supportedBinaryClassificationAccuracyKeys={data.classification_methods}
      supportedRegressionAccuracyKeys={data.regression_methods}
      supportedProbabilityAccuracyKeys={data.probability_methods}
      locale={data.locale}
      key={new Date()}
      requestMetrics={calculateMetrics}
    />,
    document.getElementById(divId)
  );
};

export { RenderDashboard, FairnessWizard };
