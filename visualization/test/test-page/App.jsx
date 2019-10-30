import React from 'react';
import { FairnessWizard } from 'fairlearn-dashboard';
import { binaryClassifier } from '../__mock-data/binaryClassifier';
import {regression} from "../__mock-data/regression";

    class App extends React.Component {
      constructor(props) {
        super(props);
        this.state = {value: 0};
        this.handleChange = this.handleChange.bind(this);
        this.generateRandomScore = this.generateRandomScore.bind(this);
      }

      static choices = [
        {label: 'binaryClassifier', data: binaryClassifier},
        {label: 'regression', data: regression}
      ]

      messages = {
        'LocalExpAndTestReq': [{displayText: 'LocalExpAndTestReq'}],
        'LocalOrGlobalAndTestReq': [{displayText: 'LocalOrGlobalAndTestReq'}],
        'TestReq': [{displayText: 'TestReq'}],
        'PredictorReq': [{displayText: 'PredictorReq'}]
      }

      handleChange(event){
        this.setState({value: event.target.value});
      }

      generateRandomScore(data) {
        return Promise.resolve(data.map(x => Math.random()));
      }

      generateRandomMetrics(data, signal) {
        const binSize = Math.max(...data.binVector);
        const bins = new Array(binSize + 1).fill(0).map(x => Math.random())
        let promise = new Promise((resolve, reject) => {
          let timeout = setTimeout(() => {resolve({
            global: Math.random(),
            bins
          })}, 300);
          if (signal) {
            signal.addEventListener('abort', () => {
              clearTimeout(timeout);
              reject(new DOMException('Aborted', 'AbortError'));
            });
          }
        });
        return promise;
      }

      generateExplanatins(explanations, data, signal) {
        let promise = new Promise((resolve, reject) => {
          let timeout = setTimeout(() => {resolve(explanations)}, 300);
          signal.addEventListener('abort', () => {
            clearTimeout(timeout);
            reject(new DOMException('Aborted', 'AbortError'));
          });
        });

        return promise;
      }


      render() {
        const data = _.cloneDeep(App.choices[this.state.value].data);
        return (
          <div style={{backgroundColor: 'grey', height:'100%'}}>
            <label>
              Select dataset:
            </label>
            <select value={this.state.value} onChange={this.handleChange}>
              {App.choices.map((item, index) => <option key={item.label} value={index}>{item.label}</option>)}
            </select>
              <div style={{ width: '80vw', backgroundColor: 'white', margin:'50px auto'}}>
                  <div style={{ width: '740px'}}>
                      <FairnessWizard
                        modelInformation={{modelClass: 'blackbox'}}
                        dataSummary={{featureNames: data.featureNames, classNames: data.classNames}}
                        testData={data.augmentedData}
                        predictedY={data.predictedYs}
                        trueY={data.trueY}
                        supportedBinaryClassificationAccuracyKeys={["accuracy_score", "precision_score", "recall_score", "zero_one_loss"]}
                        supportedRegressionAccuracyKeys={["mean_absolute_error", "recall_score"]}
                        supportedPredictionAccuracyKeys={["mean_absolute_error", "recall_score"]}
                        stringParams={{contextualHelp: this.messages}}
                        requestMetrics={this.generateRandomMetrics.bind(this)}
                        key={new Date()}
                      />
                  </div>
              </div>
          </div>
        );
      }
    }

    export default App;