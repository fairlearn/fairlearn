export interface IDatasetSummary {
    featureNames?: string[];
    classNames?: string[];
    categoricalMap?: {[key: number]: string[]};
}

export type PredictionTypes = "classes" | "regression" | "probability" | "logOdds";

export interface IMetricResponse {
    global?: number;
    bins?: number[];
}

export interface IMetricRequest {
    metricKey: string;
    binVector: number[];
    modelIndex: number;
}

export interface IFairnessProps {
    dataSummary: IDatasetSummary;
    testData: any[][];
    predictionType?: PredictionTypes;
    // One array per each model;
    predictedY: number[][];
    trueY: number[];
    theme?: any;
    stringParams?: any;
    supportedClassificationAccuracyKeys: string[];
    supportedRegressionAccuracyKeys: string[];
    // The request hook
    requestMetrics: ( request: IMetricRequest, abortSignal?: AbortSignal) =>  Promise<IMetricResponse>;
}