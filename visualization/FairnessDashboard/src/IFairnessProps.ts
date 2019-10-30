export interface IDatasetSummary {
    featureNames?: string[];
    classNames?: string[];
    categoricalMap?: {[key: number]: string[]};
}

export enum PredictionTypes {
    binaryClassification = "binaryClassification",
    regression = "regression",
    probability = "probability"
}

export type PredictionType = 
    PredictionTypes.binaryClassification |
    PredictionTypes.probability |
    PredictionTypes.regression;

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
    supportedBinaryClassificationAccuracyKeys: string[];
    supportedRegressionAccuracyKeys: string[];
    supportedProbabilityAccuracyKeys: string[];
    // The request hook
    requestMetrics: ( request: IMetricRequest, abortSignal?: AbortSignal) =>  Promise<IMetricResponse>;
}